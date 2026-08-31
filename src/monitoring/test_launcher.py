from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from typing import BinaryIO, Protocol, cast
import uuid


MAX_CONFIG_BYTES = 1024 * 1024
MAX_TESTS = 100
TEST_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
logger = logging.getLogger(__name__)


class TestCatalogError(ValueError):
    pass


class UnknownTestError(ValueError):
    pass


class TestAlreadyRunningError(RuntimeError):
    pass


class ProcessHandle(Protocol):
    """Minimal process contract used for asynchronous completion tracking."""

    def wait(self) -> int: ...


ProcessFactory = Callable[[tuple[str, ...], Path, BinaryIO], ProcessHandle]


@dataclass(frozen=True, slots=True)
class TestDefinition:
    id: str
    name: str
    description: str
    command: tuple[str, ...]

    def to_public_object(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "command": shlex.join(self.command),
        }


@dataclass(slots=True)
class TestRun:
    id: str
    test_id: str
    name: str
    command: str
    status: str
    started_at_unix: float
    finished_at_unix: float | None = None
    exit_code: int | None = None

    def to_public_object(self) -> dict[str, object]:
        return {
            "id": self.id,
            "test_id": self.test_id,
            "name": self.name,
            "command": self.command,
            "status": self.status,
            "started_at_unix": self.started_at_unix,
            "finished_at_unix": self.finished_at_unix,
            "exit_code": self.exit_code,
        }


class TestLauncher:
    def __init__(
        self,
        definitions: Sequence[TestDefinition],
        *,
        working_root: str | Path,
        log_root: str | Path,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        if not definitions:
            raise TestCatalogError("Test catalog must contain at least one test")

        resolved_working_root = Path(working_root).resolve()
        if not resolved_working_root.is_dir():
            raise TestCatalogError(
                f"Test working directory does not exist: {resolved_working_root}"
            )

        by_id = {definition.id: definition for definition in definitions}
        if len(by_id) != len(definitions):
            raise TestCatalogError("Test catalog contains duplicate identifiers")

        self._definitions = tuple(definitions)
        self._definitions_by_id = by_id
        self._working_root = resolved_working_root
        self._log_root = Path(log_root).resolve()
        self._process_factory = process_factory or _start_process
        self._runs: list[TestRun] = []
        self._active_run: TestRun | None = None
        self._lock = threading.Lock()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            active_run = (
                self._active_run.to_public_object()
                if self._active_run is not None
                else None
            )
            recent_runs = [run.to_public_object() for run in reversed(self._runs[-20:])]

        return {
            "enabled": True,
            "tests": [definition.to_public_object() for definition in self._definitions],
            "active_run": active_run,
            "recent_runs": recent_runs,
        }

    def launch(self, test_id: str) -> dict[str, object]:
        definition = self._definitions_by_id.get(test_id)
        if definition is None:
            raise UnknownTestError(f"Unknown test identifier: {test_id}")

        with self._lock:
            if self._active_run is not None:
                raise TestAlreadyRunningError(
                    f"{self._active_run.name} is already running"
                )

            self._log_root.mkdir(parents=True, exist_ok=True)
            run = TestRun(
                id=_new_run_id(),
                test_id=definition.id,
                name=definition.name,
                command=shlex.join(definition.command),
                status="starting",
                started_at_unix=time.time(),
            )
            log_file = (self._log_root / f"{run.id}.log").open("xb", buffering=0)
            try:
                process = self._process_factory(
                    definition.command,
                    self._working_root,
                    log_file,
                )
            except Exception:
                log_file.close()
                raise

            run.status = "running"
            self._runs.append(run)
            self._active_run = run

        watcher = threading.Thread(
            target=self._wait_for_process,
            args=(run, process, log_file),
            name=f"monitor-test-{run.id}",
            daemon=True,
        )
        watcher.start()
        return run.to_public_object()

    def _wait_for_process(
        self,
        run: TestRun,
        process: ProcessHandle,
        log_file: BinaryIO,
    ) -> None:
        try:
            exit_code = process.wait()
            finished_at_unix = time.time()
        except Exception:
            logger.exception("Launched test process wait failed")
            exit_code = None
            finished_at_unix = time.time()
        finally:
            log_file.close()

        with self._lock:
            run.exit_code = exit_code
            run.finished_at_unix = finished_at_unix
            if exit_code == 0:
                run.status = "passed"
            elif exit_code is None:
                run.status = "error"
            else:
                run.status = "failed"
            if self._active_run is run:
                self._active_run = None


def load_test_catalog(path: str | Path) -> tuple[TestDefinition, ...]:
    config_path = Path(path)
    try:
        size = config_path.stat().st_size
    except OSError as error:
        raise TestCatalogError(f"Could not read test catalog: {config_path}") from error
    if size > MAX_CONFIG_BYTES:
        raise TestCatalogError("Test catalog exceeds the 1 MiB size limit")

    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TestCatalogError(f"Could not parse test catalog: {config_path}") from error
    if not isinstance(value, Mapping):
        raise TestCatalogError("Test catalog must be a JSON object")
    if value.get("schema_version") != 1:
        raise TestCatalogError("Test catalog schema_version must be 1")

    raw_tests = value.get("tests")
    if not isinstance(raw_tests, list) or not raw_tests:
        raise TestCatalogError("Test catalog tests must be a non-empty array")
    if len(raw_tests) > MAX_TESTS:
        raise TestCatalogError(f"Test catalog cannot contain more than {MAX_TESTS} tests")

    definitions = tuple(_parse_definition(item) for item in raw_tests)
    if len({definition.id for definition in definitions}) != len(definitions):
        raise TestCatalogError("Test catalog contains duplicate identifiers")
    return definitions


def _parse_definition(value: object) -> TestDefinition:
    if not isinstance(value, Mapping):
        raise TestCatalogError("Each test must be a JSON object")

    test_id = value.get("id")
    name = value.get("name")
    description = value.get("description", "")
    command = value.get("command")
    if not isinstance(test_id, str) or TEST_ID_PATTERN.fullmatch(test_id) is None:
        raise TestCatalogError(
            "Test id must use lowercase letters, numbers, underscores, or hyphens"
        )
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise TestCatalogError("Test name must contain 1 to 100 characters")
    if not isinstance(description, str) or len(description) > 500:
        raise TestCatalogError("Test description cannot exceed 500 characters")
    if (
        not isinstance(command, list)
        or not command
        or len(command) > 100
        or any(
            not isinstance(argument, str) or not argument or len(argument) > 4096
            for argument in command
        )
    ):
        raise TestCatalogError("Test command must be a non-empty array of strings")

    return TestDefinition(
        id=test_id,
        name=name.strip(),
        description=description.strip(),
        command=tuple(cast(list[str], command)),
    )


def _start_process(
    command: tuple[str, ...],
    working_root: Path,
    log_file: BinaryIO,
) -> ProcessHandle:
    return subprocess.Popen(
        command,
        cwd=working_root,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"
