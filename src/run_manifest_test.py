import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.model.model_v1 import PenteNet
from src.run_manifest import source_fingerprint, write_run_manifest
from src.train.self_play import SelfPlayTrainerArgs


class RunManifestTest(unittest.TestCase):
    def test_source_fingerprint_changes_with_source_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            source = root / "src" / "module.py"
            source.write_text("value = 1\n", encoding="utf-8")
            initial = source_fingerprint(root)

            source.write_text("value = 2\n", encoding="utf-8")

            self.assertNotEqual(initial, source_fingerprint(root))

    def test_source_fingerprint_covers_native_sources_and_skips_build_trees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracked_files = {
                root / "native" / "CMakeLists.txt": "add_library(core)\n",
                root / "native" / "include" / "core.h": "int core();\n",
                root / "native" / "src" / "core.cpp": "int core() { return 1; }\n",
                root / "native" / "tests" / "core_test.cpp": "int main() {}\n",
                root / "native" / "bench" / "core_bench.cpp": "int main() {}\n",
                root / "native" / "torch" / "bindings.cpp": "void bind();\n",
                root / "native" / "torch" / "setup.py": "setup()\n",
                root / "native" / "torch" / "test_binding.py": "pass\n",
                root / "native" / "torch" / "test_extension.sh": "#!/bin/sh\n",
            }
            for path, contents in tracked_files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")

            initial = source_fingerprint(root)
            for path, contents in tracked_files.items():
                path.write_text(contents + "changed\n", encoding="utf-8")
                self.assertNotEqual(initial, source_fingerprint(root), path.as_posix())
                path.write_text(contents, encoding="utf-8")

            generated = root / "native" / "build" / "generated.cpp"
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_text("int generated;\n", encoding="utf-8")
            torch_generated = root / "native" / "torch" / "build" / "generated.cpp"
            torch_generated.parent.mkdir(parents=True, exist_ok=True)
            torch_generated.write_text("int generated;\n", encoding="utf-8")

            self.assertEqual(initial, source_fingerprint(root))

    def test_writes_reproducible_configuration_and_runtime_identity(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        device = torch.device("cpu")
        net = PenteNet(device, 5, game.get_action_size(), 1, 8, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            args = SelfPlayTrainerArgs(
                start_iteration=0,
                professional_games_training_iterations=0,
                self_play_training_iterations=1,
                temp_threshold=3,
                mcts_args=MCTSArgs(num_simulations=2),
                watch_training_raw_dataset_filepath="raw",
                watch_training_processed_dataset_filepath="processed",
                force_watch_training_raw_dataset_processing=False,
                checkpoint_dir=directory,
            )

            manifest = write_run_manifest(
                path,
                Path(__file__).parents[1],
                ["python", "main.py"],
                "run-id",
                0,
                device,
                False,
                net,
                args,
                {"seed": 3},
                Path(directory) / "metrics.jsonl",
            )

            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(manifest, loaded)
        self.assertEqual("run-id", loaded["training_run_id"])
        self.assertEqual(2, loaded["trainer"]["mcts_args"]["num_simulations"])
        self.assertEqual("cpu", loaded["runtime"]["device"])
        self.assertIn("source_fingerprint_sha256", loaded["repository"])
