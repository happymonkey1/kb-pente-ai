from __future__ import annotations

import json
from pathlib import Path
import threading

from src.monitoring.models import ReplaySample


class JsonlReplaySampleSink:
    """Append validated, safe replay samples for the monitoring dashboard."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, sample: ReplaySample) -> None:
        validated = ReplaySample.from_object(sample.to_dict())
        encoded = json.dumps(validated.to_dict(), sort_keys=True, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.write("\n")
