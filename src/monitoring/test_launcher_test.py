from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from typing import BinaryIO, cast

from src.monitoring.test_launcher import (
    TestAlreadyRunningError,
    TestCatalogError,
    TestDefinition,
    TestLauncher,
    load_test_catalog,
)


class FakeProcess:
    def __init__(self, release: threading.Event) -> None:
        self.pid = 123
        self.waiting = threading.Event()
        self._release = release

    def wait(self) -> int:
        self.waiting.set()
        if not self._release.wait(timeout=2):
            raise TimeoutError("Fake process was not released")
        return 0


class TestLauncherTest(unittest.TestCase):
    def test_loads_validated_allowlisted_commands(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tests.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "tests": [
                            {
                                "id": "monitoring-tests",
                                "name": "Monitoring tests",
                                "description": "Run focused tests.",
                                "command": ["python", "-m", "unittest"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            definitions = load_test_catalog(path)

        self.assertEqual(
            (
                TestDefinition(
                    id="monitoring-tests",
                    name="Monitoring tests",
                    description="Run focused tests.",
                    command=("python", "-m", "unittest"),
                ),
            ),
            definitions,
        )

    def test_rejects_commands_that_are_not_argument_arrays(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tests.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "tests": [
                            {
                                "id": "unsafe",
                                "name": "Unsafe",
                                "command": "python -m unittest",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                TestCatalogError,
                "non-empty array of strings",
            ):
                load_test_catalog(path)

    def test_allows_only_one_mock_process_at_a_time(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            release = threading.Event()
            process = FakeProcess(release)
            calls: list[tuple[tuple[str, ...], Path]] = []

            def start_process(
                command: tuple[str, ...],
                working_root: Path,
                log_file: BinaryIO,
            ) -> FakeProcess:
                self.assertFalse(log_file.closed)
                calls.append((command, working_root))
                return process

            launcher = TestLauncher(
                [
                    TestDefinition(
                        id="focused",
                        name="Focused tests",
                        description="",
                        command=("python", "-m", "unittest"),
                    )
                ],
                working_root=root,
                log_root=root / "logs",
                process_factory=start_process,
            )

            run = launcher.launch("focused")
            self.assertTrue(process.waiting.wait(timeout=1))
            with self.assertRaisesRegex(TestAlreadyRunningError, "already running"):
                launcher.launch("focused")
            self.assertEqual(
                [(("python", "-m", "unittest"), root.resolve())],
                calls,
            )
            self.assertEqual("running", run["status"])
            active_run = cast(
                dict[str, object],
                launcher.snapshot()["active_run"],
            )
            self.assertEqual("running", active_run["status"])

            release.set()
            deadline = time.monotonic() + 1
            while launcher.snapshot()["active_run"] is not None:
                if time.monotonic() >= deadline:
                    self.fail("Mock process did not finish")
                time.sleep(0.001)

            recent_runs = cast(
                list[dict[str, object]],
                launcher.snapshot()["recent_runs"],
            )
            completed = recent_runs[0]
            self.assertEqual("passed", completed["status"])
            self.assertEqual(0, completed["exit_code"])


if __name__ == "__main__":
    unittest.main()
