from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Mapping, Protocol


MetricValue = int | float | str | bool | None
TELEMETRY_SCHEMA_VERSION = 1


class MetricSink(Protocol):
    def emit(self, event: str, step: int, metrics: Mapping[str, MetricValue]) -> None:
        """Record one structured telemetry event."""


class NullMetricSink:
    def emit(self, event: str, step: int, metrics: Mapping[str, MetricValue]) -> None:
        return


class JsonlMetricSink:
    def __init__(self, path: str | Path, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        if run_id is not None:
            if not run_id:
                raise ValueError("Telemetry run identifier cannot be empty")
            self._validate_existing_run()
        self._lock = threading.Lock()

    def emit(self, event: str, step: int, metrics: Mapping[str, MetricValue]) -> None:
        record = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "timestamp_unix": time.time(),
            "run_id": self.run_id,
            "event": event,
            "step": step,
            "metrics": dict(metrics),
        }
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.write("\n")

    def _validate_existing_run(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("r", encoding="utf-8") as stream:
            first_line = next((line for line in stream if line.strip()), None)
        if first_line is None:
            return
        try:
            existing = json.loads(first_line)
        except json.JSONDecodeError as error:
            raise ValueError("Existing telemetry starts with invalid JSON") from error
        if existing.get("run_id") != self.run_id:
            raise ValueError("Telemetry file belongs to a different training run")


@dataclass
class InMemoryMetricSink:
    records: list[dict[str, object]]

    def __init__(self) -> None:
        self.records = []

    def emit(self, event: str, step: int, metrics: Mapping[str, MetricValue]) -> None:
        self.records.append({"event": event, "step": step, "metrics": dict(metrics)})
