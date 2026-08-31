import json
import os
import tempfile
import unittest
from pathlib import Path

from src.telemetry import InMemoryMetricSink, JsonlMetricSink


class TelemetryTest(unittest.TestCase):
    def test_in_memory_sink_preserves_event_step_and_metrics(self) -> None:
        sink = InMemoryMetricSink()

        sink.emit("search", 4, {"root_children": 3})

        self.assertEqual(
            [{"event": "search", "step": 4, "metrics": {"root_children": 3}}],
            sink.records,
        )

    def test_jsonl_sink_writes_stable_valid_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "metrics.jsonl")
            sink = JsonlMetricSink(path)

            sink.emit("training", 2, {"loss": 1.25, "finite": True})

            with open(path, "r", encoding="utf-8") as stream:
                record = json.loads(stream.read())
        self.assertEqual(1, record["schema_version"])
        self.assertEqual("training", record["event"])
        self.assertEqual(2, record["step"])
        self.assertEqual({"finite": True, "loss": 1.25}, record["metrics"])

    def test_run_identity_prevents_mixed_telemetry_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            JsonlMetricSink(path, run_id="first").emit("training", 1, {})

            JsonlMetricSink(path, run_id="first").emit("training", 2, {})
            with self.assertRaisesRegex(ValueError, "different training run"):
                JsonlMetricSink(path, run_id="second")

            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(["first", "first"], [record["run_id"] for record in records])


if __name__ == "__main__":
    unittest.main()
