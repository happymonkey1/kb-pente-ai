from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping
import unittest


_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "script" / "sweep-native-cuda-utilization.sh"


def _run(
    *arguments: str,
    python: str = sys.executable,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_environment = os.environ.copy()
    if environment is not None:
        process_environment.update(environment)
    return subprocess.run(
        ["bash", str(_SCRIPT), "--python", python, *arguments],
        cwd=_ROOT,
        capture_output=True,
        env=process_environment,
        text=True,
        check=False,
    )


class NativeCudaSweepTest(unittest.TestCase):
    def test_help_and_dry_run_do_not_require_cuda(self) -> None:
        help_result = _run("--help")
        self.assertEqual(0, help_result.returncode)
        self.assertIn("--profiles", help_result.stdout)
        self.assertIn("--checkpoint", help_result.stdout)

        prefix = f"test-native-cuda-dry-{os.getpid()}"
        result = _run(
            "--dry-run",
            "--no-compile",
            "--prefix",
            prefix,
            "--profiles",
            "1:1 2:2",
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(2, result.stdout.count("  command:"))
        self.assertIn("--ruleset standard", result.stdout)
        self.assertIn("--board-size 19", result.stdout)
        self.assertIn("--model-blocks 6", result.stdout)
        self.assertIn("--model-channels 128", result.stdout)
        self.assertIn("--model-hidden-size 256", result.stdout)
        self.assertIn("--batch-games 512", result.stdout)
        self.assertIn("--learner-steps 1", result.stdout)
        self.assertIn("--minimum-batch-occupancy 0.80", result.stdout)
        self.assertIn("--active-games 2", result.stdout)
        self.assertIn("--native-search-threads 2", result.stdout)
        self.assertEqual(2, result.stdout.count("--native-search-cohorts 1"))
        self.assertIn("Dry run complete: 2 profiles validated", result.stdout)
        self.assertFalse(
            (_ROOT / f"pente-model-{prefix}-active1-workers1").exists()
        )
        self.assertFalse(
            (_ROOT / f"pente-model-{prefix}-active2-workers2").exists()
        )
        self.assertFalse(
            (_ROOT / "metrics" / f"{prefix}-active1-workers1.jsonl").exists()
        )
        self.assertFalse(
            (_ROOT / "replays" / f"{prefix}-active2-workers2.jsonl").exists()
        )

    def test_profile_validation_rejects_normalized_duplicates_and_invalid_budgets(self) -> None:
        duplicate = _run(
            "--dry-run",
            "--profiles",
            "512:8 512:8:1",
            "--prefix",
            f"test-native-cuda-duplicate-{os.getpid()}",
        )
        self.assertEqual(64, duplicate.returncode)
        self.assertIn("duplicate profile", duplicate.stderr)

        for profile, message in (
            ("2:2:3", "cohorts must be 1 or 2"),
            ("1:2:2", "active-games >= 2"),
            ("2:1:2", "native-workers >= 2"),
            ("513:2", "must not exceed 512"),
        ):
            result = _run(
                "--dry-run",
                "--profiles",
                profile,
                "--prefix",
                f"test-native-cuda-invalid-{os.getpid()}-{profile.replace(':', '-')}",
            )
            self.assertEqual(64, result.returncode, profile)
            self.assertIn(message, result.stderr, profile)

    def test_checkpoint_requires_paired_replay_and_explicit_final_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            checkpoint = temporary_path / "checkpoint.pth.tar"
            replay = temporary_path / "replay.pkl"
            checkpoint.touch()
            replay.touch()

            missing_target = _run(
                "--dry-run",
                "--checkpoint",
                str(checkpoint),
                "--resume-replay",
                str(replay),
                "--profiles",
                "1:1",
            )
            self.assertEqual(64, missing_target.returncode)
            self.assertIn("--final-iteration", missing_target.stderr)

            result = _run(
                "--dry-run",
                "--no-compile",
                "--checkpoint",
                str(checkpoint),
                "--resume-replay",
                str(replay),
                "--final-iteration",
                "8",
                "--profiles",
                "1:1",
                "--prefix",
                f"test-native-cuda-resume-{os.getpid()}",
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("--self-play-iterations 8", result.stdout)
            self.assertIn(f"--model {checkpoint}", result.stdout)
            self.assertIn(f"--resume-replay {replay}", result.stdout)

    def test_runs_profiles_in_order_and_rejects_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            fake_python = temporary_path / "fake-python"
            marker = temporary_path / "invocations.jsonl"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "import os\n"
                "from pathlib import Path\n"
                "import sys\n"
                "marker = Path(os.environ['SWEEP_TEST_MARKER'])\n"
                "code = None\n"
                "if len(sys.argv) > 1 and sys.argv[1] == '-c':\n"
                "    code = sys.argv[2]\n"
                "    kind = 'cuda-check' if 'torch.cuda.is_available' in code else 'native-check'\n"
                "    if kind == 'cuda-check':\n"
                "        print('available')\n"
                "else:\n"
                "    kind = 'main'\n"
                "with marker.open('a', encoding='utf-8') as stream:\n"
                "    stream.write(json.dumps({'kind': kind, 'argv': sys.argv[1:], 'code': code}) + '\\n')\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            prefix = f"test-native-cuda-run-{os.getpid()}"
            profile_text = "1:1 2:2:1 2:2:2"
            profiles = (
                (1, 1, 1, ""),
                (2, 2, 1, "-cohorts1"),
                (2, 2, 2, "-cohorts2"),
            )
            try:
                result = _run(
                    "--no-compile",
                    "--prefix",
                    prefix,
                    "--profiles",
                    profile_text,
                    python=str(fake_python),
                    environment={"SWEEP_TEST_MARKER": str(marker)},
                )
                self.assertEqual(0, result.returncode, result.stderr)
                records = [
                    json.loads(line)
                    for line in marker.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(
                    ["cuda-check", "native-check", "main", "main", "main"],
                    [record["kind"] for record in records],
                )
                native_code = records[1]["code"]
                self.assertIsInstance(native_code, str)
                assert isinstance(native_code, str)
                self.assertLess(
                    native_code.index("import torch"),
                    native_code.index("import kb_pente_native"),
                )
                actual_profiles = [
                    (
                        record["argv"][record["argv"].index("--active-games") + 1],
                        record["argv"][record["argv"].index("--native-search-threads") + 1],
                        record["argv"][record["argv"].index("--native-search-cohorts") + 1],
                    )
                    for record in records[2:]
                ]
                self.assertEqual(
                    [("1", "1", "1"), ("2", "2", "1"), ("2", "2", "2")],
                    actual_profiles,
                )
                for active, workers, cohorts, suffix in profiles:
                    name = f"{prefix}-active{active}-workers{workers}{suffix}"
                    self.assertTrue(
                        (_ROOT / f"pente-model-{name}").is_dir()
                    )
                    self.assertTrue(
                        (_ROOT / "metrics" / f"{name}.jsonl").is_file()
                    )
                    self.assertTrue(
                        (_ROOT / "replays" / f"{name}.jsonl").is_file()
                    )
                    log = (_ROOT / "logs" / f"{name}.log").read_text(encoding="utf-8")
                    self.assertIn(f"native-cohorts={cohorts}", log)

                rerun = _run(
                    "--no-compile",
                    "--prefix",
                    prefix,
                    "--profiles",
                    profile_text,
                    python=str(fake_python),
                    environment={"SWEEP_TEST_MARKER": str(marker)},
                )
                self.assertEqual(64, rerun.returncode)
                self.assertIn("output already exists", rerun.stderr)
                self.assertEqual(5, len(marker.read_text(encoding="utf-8").splitlines()))
            finally:
                for active, workers, _cohorts, suffix in profiles:
                    name = f"{prefix}-active{active}-workers{workers}{suffix}"
                    shutil.rmtree(
                        _ROOT / f"pente-model-{name}",
                        ignore_errors=True,
                    )
                    for output in (
                        _ROOT / "metrics" / f"{name}.jsonl",
                        _ROOT / "replays" / f"{name}.jsonl",
                        _ROOT / "logs" / f"{name}.log",
                    ):
                        output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
