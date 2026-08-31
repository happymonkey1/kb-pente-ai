from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import cast
import unittest
from unittest.mock import patch

import torch

from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.model.model_v1 import PenteNetConfig
from src.verification.self_play_benchmark_reports import (
    SelfPlayBenchmarkConfig,
    SelfPlayBenchmarkRatios,
    SelfPlayBenchmarkReport,
    SelfPlayBenchmarkSummary,
)


def _load_script():
    path = Path(__file__).parents[2] / "script" / "benchmark-native-self-play.py"
    spec = importlib.util.spec_from_file_location("benchmark_native_self_play", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load native self-play benchmark script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_script()


class _FakeModel:
    def __init__(self, board_size: int = 5) -> None:
        self.config = PenteNetConfig(
            board_size=board_size,
            action_size=board_size * board_size,
            num_res_blocks=1,
            num_channels=4,
            hidden_fc_size=8,
        )

    def get_parameter_count(self) -> int:
        return 1234


def _report(
    config: SelfPlayBenchmarkConfig, passed: bool = True
) -> SelfPlayBenchmarkReport:
    summary = SelfPlayBenchmarkSummary("python", 0, ())
    return SelfPlayBenchmarkReport(
        config=config,
        criteria=config.criteria(),
        raw_runs=(),
        python=summary,
        cpp=SelfPlayBenchmarkSummary("cpp", 0, ()),
        ratios=SelfPlayBenchmarkRatios(2.0, 2.0, 2.0),
        elapsed_seconds=0.25,
        passed=passed,
        failures=() if passed else ("criterion failed",),
    )


def _minimal_args(checkpoint: Path, *extra: str) -> list[str]:
    return [
        str(checkpoint),
        "--board-size",
        "5",
        "--ruleset",
        "freestyle",
        "--games",
        "1",
        "--max-active-games",
        "1",
        "--simulations",
        "1",
        "--repeats",
        "1",
        "--warmup-batches",
        "0",
        "--model-blocks",
        "1",
        "--model-channels",
        "4",
        "--model-hidden-size",
        "8",
        *extra,
    ]


def _write_history(path: Path) -> bytes:
    metrics = {
        "device_type": "cpu",
        "games": 8,
        "games_per_second": 2.0,
        "positions_per_second": 100.0,
        "leaf_evaluations_per_second": 200.0,
        "mean_inference_batch_size": 4.0,
        "p95_inference_batch_size": 5.0,
        "max_inference_batch_size": 6.0,
        "duplicate_leaf_rate": 0.1,
        "steady_state_mean_batch_occupancy": 0.9,
        "active_game_target": 8,
    }
    payload = (
        json.dumps(
            {
                "event": "training_iteration",
                "run_id": "history-run",
                "step": 4,
                "timestamp_unix": 1234.0,
                "metrics": metrics,
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    path.write_bytes(payload)
    return payload


class SelfPlayBenchmarkCliTest(unittest.TestCase):
    def test_import_parser_and_help_do_not_preflight(self) -> None:
        parser = _SCRIPT.build_parser()
        defaults = parser.parse_args(["checkpoint"])
        self.assertEqual(16, defaults.games)
        self.assertEqual(16, defaults.max_active_games)
        self.assertEqual(4, defaults.model_blocks)
        with patch.object(_SCRIPT, "load_native_extension") as preflight:
            parser.format_help()
            with self.assertRaises(SystemExit) as help_exit:
                _SCRIPT.main(["--help"])
            self.assertEqual(0, help_exit.exception.code)
            preflight.assert_not_called()

    def test_invalid_checkpoint_config_cuda_and_history_preflight_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pth.tar"
            history = root / "history.jsonl"
            checkpoint.write_bytes(b"checkpoint")
            history.write_text('{"broken"\n', encoding="utf-8")

            cases = (
                ([str(root / "missing.pth.tar")], "regular file"),
                (_minimal_args(checkpoint, "--games", "0"), "games must be positive"),
                (_minimal_args(checkpoint, "--model-blocks", "0"), "model_blocks must be positive"),
                (_minimal_args(checkpoint, "--historical-path", str(history)), "malformed JSON"),
            )
            for arguments, message in cases:
                with (
                    self.subTest(message=message),
                    patch.object(
                        _SCRIPT,
                        "load_native_extension",
                        side_effect=AssertionError("preflight ran before validation"),
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    _SCRIPT.main(arguments)

            with (
                patch.object(_SCRIPT.torch.cuda, "is_available", return_value=False),
                patch.object(
                    _SCRIPT,
                    "load_native_extension",
                    side_effect=AssertionError("preflight ran before validation"),
                ),
                self.assertRaisesRegex(ValueError, "without an available CUDA"),
            ):
                _SCRIPT.main(_minimal_args(checkpoint, "--gpu"))

    def test_load_model_propagates_architecture_and_expected_ruleset(self) -> None:
        args = _SCRIPT.build_parser().parse_args(
            [
                "checkpoint",
                "--board-size",
                "9",
                "--ruleset",
                "freestyle",
                "--model-blocks",
                "2",
                "--model-channels",
                "8",
                "--model-hidden-size",
                "32",
            ],
        )
        game = PenteGame(9, ruleset=PenteRuleset.FREESTYLE)
        calls: list[dict[str, object]] = []

        class FakePenteNet:
            def __init__(self, device: torch.device, **kwargs: object) -> None:
                calls.append({"device": device, **kwargs})

            @staticmethod
            def load_checkpoint_from_path(
                path: str,
                model: object,
                expected_ruleset: str,
            ) -> int:
                calls.append(
                    {
                        "path": path,
                        "model": model,
                        "expected_ruleset": expected_ruleset,
                    },
                )
                return 19

        with patch.object(_SCRIPT, "PenteNet", FakePenteNet):
            model, iteration = _SCRIPT._load_model(
                Path("checkpoint"), torch.device("cpu"), game, args
            )

        self.assertEqual(19, iteration)
        self.assertIsInstance(model, FakePenteNet)
        self.assertEqual(
            {
                "device": torch.device("cpu"),
                "board_size": 9,
                "action_size": 81,
                "num_res_blocks": 2,
                "num_channels": 8,
                "hidden_fc_size": 32,
            },
            calls[0],
        )
        self.assertEqual("freestyle", calls[1]["expected_ruleset"])

    def test_valid_run_preflights_once_calls_api_once_and_emits_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pth.tar"
            history = root / "history.jsonl"
            checkpoint.write_bytes(b"checkpoint")
            history_bytes = _write_history(history)
            model = _FakeModel()
            events: list[str] = []
            benchmark_calls: list[tuple[object, ...]] = []
            printed: list[str] = []

            def load_model(*arguments: object) -> tuple[_FakeModel, int]:
                events.append("load")
                return model, 7

            def benchmark(*arguments: object) -> SelfPlayBenchmarkReport:
                events.append("benchmark")
                benchmark_calls.append(arguments)
                return _report(cast(SelfPlayBenchmarkConfig, arguments[2]))

            with (
                patch.object(_SCRIPT, "load_native_extension", side_effect=lambda: events.append("preflight")),
                patch.object(_SCRIPT, "_load_model", side_effect=load_model),
                patch.object(_SCRIPT, "run_self_play_benchmark", side_effect=benchmark),
                patch.object(_SCRIPT, "_sha256", return_value="checkpoint-hash"),
                patch.object(_SCRIPT, "source_fingerprint", return_value="source-hash"),
                patch.object(_SCRIPT, "_git_commit", return_value="commit-hash"),
                patch.object(_SCRIPT, "_runtime_metadata", return_value={"device": "cpu", "finite": 1.0}),
                patch("builtins.print", side_effect=printed.append),
            ):
                result = _SCRIPT.main(
                    _minimal_args(
                        checkpoint,
                        "--historical-path",
                        str(history),
                        "--torch-threads",
                        "2",
                        "--minimum-native-games-per-second-ratio",
                        "1.5",
                    ),
                )

        self.assertEqual(0, result)
        self.assertEqual(["preflight", "load", "benchmark"], events)
        self.assertEqual(1, len(benchmark_calls))
        config = cast(SelfPlayBenchmarkConfig, benchmark_calls[0][2])
        self.assertEqual(2, config.torch_threads)
        self.assertEqual(1.5, config.minimum_native_games_per_second_ratio)
        output = json.loads(printed[0])
        self.assertEqual(str(checkpoint.resolve()), output["checkpoint"]["path"])
        self.assertEqual("checkpoint-hash", output["checkpoint"]["sha256"])
        self.assertEqual(7, output["checkpoint"]["iteration"])
        self.assertEqual(1234, output["model"]["parameter_count"])
        self.assertEqual("source-hash", output["repository"]["source_fingerprint_sha256"])
        self.assertEqual("commit-hash", output["repository"]["commit"])
        self.assertEqual("history-run", output["historical_reference"]["run_id"])
        self.assertEqual(
            hashlib.sha256(history_bytes).hexdigest(),
            output["historical_reference"]["source_sha256"],
        )
        self.assertEqual("freestyle", output["report"]["config"]["ruleset"])
        self.assertEqual(printed[0], _SCRIPT._stable_json(output))
        json.dumps(output, allow_nan=False)

    def test_failed_report_returns_one_without_changing_historical_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth.tar"
            history = Path(directory) / "history.jsonl"
            checkpoint.write_bytes(b"checkpoint")
            _write_history(history)
            model = _FakeModel()
            report_calls = 0
            printed: list[str] = []

            def benchmark(*arguments: object) -> SelfPlayBenchmarkReport:
                nonlocal report_calls
                report_calls += 1
                return _report(
                    cast(SelfPlayBenchmarkConfig, arguments[2]), passed=False
                )

            with (
                patch.object(_SCRIPT, "load_native_extension") as preflight,
                patch.object(_SCRIPT, "_load_model", return_value=(model, 1)),
                patch.object(_SCRIPT, "run_self_play_benchmark", side_effect=benchmark),
                patch.object(_SCRIPT, "_sha256", return_value="hash"),
                patch.object(_SCRIPT, "source_fingerprint", return_value="source"),
                patch.object(_SCRIPT, "_git_commit", return_value="commit"),
                patch.object(_SCRIPT, "_runtime_metadata", return_value={"device": "cpu"}),
                patch("builtins.print", side_effect=printed.append),
            ):
                result = _SCRIPT.main(
                    _minimal_args(
                        checkpoint,
                        "--historical-path",
                        str(history),
                    ),
                )

        self.assertEqual(1, result)
        preflight.assert_called_once_with()
        self.assertEqual(1, report_calls)
        output = json.loads(printed[0])
        self.assertEqual(
            "informational",
            output["historical_reference"]["comparison_kind"],
        )


if __name__ == "__main__":
    unittest.main()
