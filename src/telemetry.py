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
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: str, step: int, metrics: Mapping[str, MetricValue]) -> None:
        record = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "timestamp_unix": time.time(),
            "event": event,
            "step": step,
            "metrics": dict(metrics),
        }
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.write("\n")


@dataclass
class InMemoryMetricSink:
    records: list[dict[str, object]]

    def __init__(self) -> None:
        self.records = []

    def emit(self, event: str, step: int, metrics: Mapping[str, MetricValue]) -> None:
        self.records.append({"event": event, "step": step, "metrics": dict(metrics)})
