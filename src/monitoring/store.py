from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import base64
import json
import math
from pathlib import Path, PurePosixPath
import time
from typing import cast, Generic, TypeVar

from src.monitoring.device_summary import summarize_device
from src.monitoring.models import ArtifactIssue, ReplaySample, TelemetryRecord


DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RECORDS = 100_000
DEFAULT_ACTIVITY_WINDOW_SECONDS = 120.0


class ArtifactIdentifierError(ValueError):
    pass


class ArtifactNotFoundError(FileNotFoundError):
    pass


RecordT = TypeVar("RecordT")


@dataclass(frozen=True, slots=True)
class ParsedFile(Generic[RecordT]):
    records: tuple[tuple[int, RecordT], ...]
    issues: tuple[ArtifactIssue, ...]


@dataclass(frozen=True, slots=True)
class CachedFile(Generic[RecordT]):
    signature: tuple[int, int]
    size_bytes: int
    modified_at_unix: float
    parsed: ParsedFile[RecordT]


class JsonlStore(Generic[RecordT]):
    def __init__(
        self,
        root: str | Path,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> None:
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")
        if max_records < 1:
            raise ValueError("max_records must be positive")

        self.root = Path(root).resolve()
        self.max_file_bytes = max_file_bytes
        self.max_records = max_records
        self._cache: dict[Path, CachedFile[RecordT]] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _parse_record(self, value: object) -> RecordT:
        raise NotImplementedError

    def _discover(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        result: list[Path] = []
        for path in self.root.rglob("*.jsonl"):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved.is_file() and resolved.is_relative_to(self.root):
                result.append(resolved)
        return sorted(result, key=lambda path: path.relative_to(self.root).as_posix())

    def _identifier(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _path(self, identifier: str) -> Path:
        if not identifier or "\x00" in identifier:
            raise ArtifactIdentifierError("Artifact identifier is invalid")
        candidate = (self.root / identifier).resolve()
        if not candidate.is_relative_to(self.root) or candidate.suffix != ".jsonl":
            raise ArtifactIdentifierError("Artifact identifier is outside the configured root")
        if not candidate.is_file():
            raise ArtifactNotFoundError(f"Artifact does not exist: {identifier}")
        return candidate

    def _load(self, path: Path) -> CachedFile[RecordT]:
        try:
            before = path.stat()
        except OSError as error:
            raise ArtifactNotFoundError(f"Artifact cannot be read: {path.name}") from error
        signature = (before.st_size, before.st_mtime_ns)
        cached = self._cache.get(path)
        if cached is not None and cached.signature == signature:
            return cached

        parsed: ParsedFile[RecordT]
        if before.st_size > self.max_file_bytes:
            parsed = ParsedFile(
                records=(),
                issues=(
                    ArtifactIssue(
                        "file_too_large",
                        f"File exceeds the {self.max_file_bytes} byte monitoring limit",
                    ),
                ),
            )
            result = CachedFile(signature, before.st_size, before.st_mtime, parsed)
            self._cache[path] = result
            return result

        try:
            content = path.read_bytes()
        except OSError as error:
            raise ArtifactNotFoundError(f"Artifact cannot be read: {path.name}") from error
        if len(content) > self.max_file_bytes:
            parsed = ParsedFile(
                records=(),
                issues=(
                    ArtifactIssue(
                        "file_too_large",
                        f"File exceeds the {self.max_file_bytes} byte monitoring limit",
                    ),
                ),
            )
        else:
            parsed = self._parse_content(content)

        try:
            after = path.stat()
        except OSError:
            after = before
        after_signature = (after.st_size, after.st_mtime_ns)
        cache_signature = after_signature if after_signature == signature else signature
        result = CachedFile(
            signature=cache_signature,
            size_bytes=len(content),
            modified_at_unix=before.st_mtime,
            parsed=parsed,
        )
        self._cache[path] = result
        return result

    def _parse_content(self, content: bytes) -> ParsedFile[RecordT]:
        lines = content.splitlines()
        has_partial_tail = bool(content) and not content.endswith((b"\n", b"\r"))
        if has_partial_tail:
            lines = lines[:-1]

        records: list[tuple[int, RecordT]] = []
        issues: list[ArtifactIssue] = []
        for line_number, encoded in enumerate(lines, start=1):
            if not encoded.strip():
                continue
            if len(records) >= self.max_records:
                issues.append(
                    ArtifactIssue(
                        "record_limit",
                        f"File contains more than {self.max_records} monitored records",
                        line_number,
                    )
                )
                break
            try:
                value = json.loads(encoded)
                records.append((line_number, self._parse_record(value)))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                issues.append(ArtifactIssue("invalid_record", str(error), line_number))

        return ParsedFile(tuple(records), tuple(issues))
class TelemetryStore(JsonlStore[TelemetryRecord]):
    def __init__(
        self,
        root: str | Path,
        *,
        activity_window_seconds: float = DEFAULT_ACTIVITY_WINDOW_SECONDS,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> None:
        if activity_window_seconds <= 0:
            raise ValueError("activity_window_seconds must be positive")
        super().__init__(
            root,
            max_file_bytes=max_file_bytes,
            max_records=max_records,
        )
        self.activity_window_seconds = activity_window_seconds

    def _parse_record(self, value: object) -> TelemetryRecord:
        return TelemetryRecord.from_object(value)

    def list_runs(self, *, now: float | None = None) -> list[dict[str, object]]:
        observed_at = time.time() if now is None else now
        runs = [self._describe(path, observed_at) for path in self._discover()]
        runs.sort(
            key=lambda run: cast(float, run["modified_at_unix"]),
            reverse=True,
        )
        return runs

    def records(self, run_id: str, *, after: int, limit: int) -> dict[str, object]:
        if after < 0:
            raise ValueError("after cannot be negative")
        if not 1 <= limit <= 2_000:
            raise ValueError("limit must be between 1 and 2000")
        path = self._path(run_id)
        cached = self._load(path)
        records = cached.parsed.records
        reset_required = after > len(records)
        offset = 0 if reset_required else after
        selected = records[offset : offset + limit]
        return {
            "run_id": run_id,
            "offset": offset,
            "next_cursor": offset + len(selected),
            "total_records": len(records),
            "has_more": offset + len(selected) < len(records),
            "reset_required": reset_required,
            "records": [record.to_dict() for _, record in selected],
            "issues": [issue.to_dict() for issue in cached.parsed.issues],
        }

    def summary(self, run_id: str, *, now: float | None = None) -> dict[str, object]:
        observed_at = time.time() if now is None else now
        path = self._path(run_id)
        cached = self._load(path)
        description = self._describe_cached(path, cached, observed_at)
        event_counts: Counter[str] = Counter()
        latest_by_event: dict[str, dict[str, object]] = {}
        latest_metrics: dict[str, object] = {}
        numeric_values: dict[str, list[float]] = {}

        for _, record in cached.parsed.records:
            event_counts[record.event] += 1
            latest_by_event[record.event] = record.to_dict()
            for name, value in record.metrics.items():
                latest_metrics[name] = value
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_values.setdefault(name, []).append(float(value))

        statistics = {
            name: {
                "count": len(values),
                "latest": values[-1],
                "minimum": min(values),
                "maximum": max(values),
                "mean": math.fsum(value / len(values) for value in values),
            }
            for name, values in sorted(numeric_values.items())
        }
        return {
            **description,
            "device": summarize_device(record for _, record in cached.parsed.records),
            "event_counts": dict(sorted(event_counts.items())),
            "latest_by_event": latest_by_event,
            "latest_metrics": latest_metrics,
            "numeric_metrics": sorted(numeric_values),
            "statistics": statistics,
        }

    def _describe(self, path: Path, observed_at: float) -> dict[str, object]:
        return self._describe_cached(path, self._load(path), observed_at)

    def _describe_cached(
        self,
        path: Path,
        cached: CachedFile[TelemetryRecord],
        observed_at: float,
    ) -> dict[str, object]:
        records = cached.parsed.records
        last = records[-1][1] if records else None
        age_seconds = max(0.0, observed_at - cached.modified_at_unix)
        if cached.parsed.issues:
            status = "degraded"
        elif not records:
            status = "empty"
        elif age_seconds <= self.activity_window_seconds:
            status = "active"
        else:
            status = "idle"
        run_id = self._identifier(path)
        training_run_ids = {
            record.run_id for _, record in records if record.run_id is not None
        }
        training_run_id = (
            next(iter(training_run_ids)) if len(training_run_ids) == 1 else None
        )
        return {
            "id": run_id,
            "run_key": training_run_id or str(PurePosixPath(run_id).with_suffix("")),
            "training_run_id": training_run_id,
            "name": path.stem,
            "status": status,
            "size_bytes": cached.size_bytes,
            "modified_at_unix": cached.modified_at_unix,
            "age_seconds": age_seconds,
            "record_count": len(records),
            "last_event": last.event if last is not None else None,
            "last_step": last.step if last is not None else None,
            "last_timestamp_unix": last.timestamp_unix if last is not None else None,
            "issues": [issue.to_dict() for issue in cached.parsed.issues],
        }


@dataclass(frozen=True, slots=True)
class ReplayEntry:
    source: str
    line: int
    sample: ReplaySample

    @property
    def identifier(self) -> str:
        payload = f"{self.source}\x00{self.line}".encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    def metadata(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "source": self.source,
            **self.sample.metadata(),
        }


class ReplayStore(JsonlStore[ReplaySample]):
    def _parse_record(self, value: object) -> ReplaySample:
        return ReplaySample.from_object(value)

    def list_replays(self, *, run_id: str | None = None) -> dict[str, object]:
        replays: list[ReplayEntry] = []
        issues: list[dict[str, object]] = []
        for path in self._discover():
            source = self._identifier(path)
            cached = self._load(path)
            replays.extend(
                ReplayEntry(source, line, sample)
                for line, sample in cached.parsed.records
                if run_id is None or sample.run_id == run_id
            )
            issues.extend(
                {"source": source, **issue.to_dict()}
                for issue in cached.parsed.issues
            )
        replays.sort(key=lambda replay: replay.sample.recorded_at_unix, reverse=True)
        return {
            "replays": [replay.metadata() for replay in replays[:500]],
            "issues": issues,
        }

    def replay(self, replay_id: str) -> dict[str, object]:
        source, line = _decode_replay_id(replay_id)
        cached = self._load(self._path(source))
        for record_line, sample in cached.parsed.records:
            if record_line == line:
                return {
                    "id": replay_id,
                    "source": source,
                    **sample.to_dict(),
                }
        raise ArtifactNotFoundError("Replay sample does not exist")


def _decode_replay_id(replay_id: str) -> tuple[str, int]:
    if not replay_id:
        raise ArtifactIdentifierError("Replay identifier is invalid")
    padding = "=" * (-len(replay_id) % 4)
    try:
        payload = base64.b64decode(
            replay_id + padding,
            altchars=b"-_",
            validate=True,
        ).decode("utf-8")
        source, raw_line = payload.rsplit("\x00", 1)
        line = int(raw_line)
    except (UnicodeDecodeError, ValueError) as error:
        raise ArtifactIdentifierError("Replay identifier is invalid") from error
    if line < 1:
        raise ArtifactIdentifierError("Replay identifier is invalid")
    return source, line
