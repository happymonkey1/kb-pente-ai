from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from typing import Any, cast

from src.monitoring.models import ReplaySample
from src.monitoring.replay_writer import JsonlReplaySampleSink
from src.monitoring.store import (
    ArtifactIdentifierError,
    ReplayStore,
    TelemetryStore,
)


def _telemetry_record(step: int, loss: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "timestamp_unix": 100.0 + step,
        "event": "training_iteration",
        "step": step,
        "metrics": {"loss": loss, "games": step * 4},
    }


class TelemetryStoreTest(unittest.TestCase):
    def test_discovers_summarizes_and_pages_runs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "experiment" / "alpha.jsonl"
            path.parent.mkdir()
            path.write_text(
                "\n".join(
                    json.dumps(_telemetry_record(step, loss))
                    for step, loss in ((1, 3.0), (2, 2.0), (3, 1.0))
                )
                + "\n",
                encoding="utf-8",
            )
            store = TelemetryStore(root, activity_window_seconds=10_000)

            runs = cast(
                list[dict[str, Any]],
                store.list_runs(now=path.stat().st_mtime + 1),
            )
            summary = cast(
                dict[str, Any],
                store.summary("experiment/alpha.jsonl", now=path.stat().st_mtime + 1),
            )
            page = cast(
                dict[str, Any],
                store.records("experiment/alpha.jsonl", after=1, limit=1),
            )

        self.assertEqual(1, len(runs))
        self.assertEqual("experiment/alpha", runs[0]["run_key"])
        self.assertEqual("active", runs[0]["status"])
        self.assertEqual(3, summary["record_count"])
        self.assertEqual(1.0, summary["latest_metrics"]["loss"])
        self.assertEqual(
            {
                "count": 3,
                "latest": 1.0,
                "minimum": 1.0,
                "maximum": 3.0,
                "mean": 2.0,
            },
            summary["statistics"]["loss"],
        )
        self.assertEqual(2, page["next_cursor"])
        self.assertEqual(2, page["records"][0]["step"])
        self.assertTrue(page["has_more"])

    def test_ignores_partial_tail_and_reports_malformed_complete_line(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "training.jsonl"
            path.write_bytes(
                json.dumps(_telemetry_record(1, 1.0)).encode("utf-8")
                + b"\n{not-json}\n"
                + b'{"schema_version":1'
            )
            store = TelemetryStore(root)

            summary = cast(dict[str, Any], store.summary("training.jsonl"))

        self.assertEqual(1, summary["record_count"])
        self.assertEqual("degraded", summary["status"])
        self.assertEqual(1, len(summary["issues"]))
        self.assertEqual(2, summary["issues"][0]["line"])

    def test_rejects_paths_outside_the_configured_root(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training.jsonl").write_text("", encoding="utf-8")
            store = TelemetryStore(root)

            with self.assertRaises(ArtifactIdentifierError):
                store.summary("../training.jsonl")

    def test_resets_cursor_after_file_truncation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "training.jsonl"
            path.write_text(
                "\n".join(json.dumps(_telemetry_record(step, 1.0)) for step in range(3)) + "\n",
                encoding="utf-8",
            )
            store = TelemetryStore(root)
            self.assertEqual(3, store.records("training.jsonl", after=0, limit=10)["total_records"])

            path.write_text(json.dumps(_telemetry_record(5, 0.5)) + "\n", encoding="utf-8")
            page = cast(
                dict[str, Any],
                store.records("training.jsonl", after=3, limit=10),
            )

        self.assertTrue(page["reset_required"])
        self.assertEqual(0, page["offset"])
        self.assertEqual(5, page["records"][0]["step"])


class ReplayStoreTest(unittest.TestCase):
    def test_sink_and_store_round_trip_filtered_samples(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            sink = JsonlReplaySampleSink(path)
            first = ReplaySample.from_object(
                {
                    "schema_version": 1,
                    "run_id": "alpha",
                    "game_id": "game-1",
                    "recorded_at_unix": 10.0,
                    "board_size": 5,
                    "ruleset": "freestyle",
                    "actions": [0, 1, 5],
                    "winner": 1,
                    "win_reason": "line",
                }
            )
            second = ReplaySample.from_object(
                {
                    "schema_version": 1,
                    "run_id": "beta",
                    "game_id": "game-2",
                    "recorded_at_unix": 11.0,
                    "board_size": 5,
                    "ruleset": "freestyle",
                    "actions": [2, 3],
                    "winner": None,
                    "win_reason": None,
                }
            )
            sink.emit(first)
            sink.emit(second)
            store = ReplayStore(directory)

            listing = cast(dict[str, Any], store.list_replays(run_id="alpha"))
            replay_id = listing["replays"][0]["id"]
            replay = cast(dict[str, Any], store.replay(replay_id))

        self.assertEqual(1, len(listing["replays"]))
        self.assertEqual(first.to_dict()["actions"], replay["actions"])
        self.assertEqual("game-1", replay["game_id"])

    def test_sink_revalidates_directly_constructed_samples(self) -> None:
        invalid = ReplaySample(
            schema_version=1,
            run_id="alpha",
            game_id="invalid",
            recorded_at_unix=1.0,
            board_size=3,
            ruleset="freestyle",
            actions=(0,),
            winner=None,
            win_reason=None,
        )
        with TemporaryDirectory() as directory:
            sink = JsonlReplaySampleSink(Path(directory) / "samples.jsonl")

            with self.assertRaisesRegex(ValueError, "board_size"):
                sink.emit(invalid)


if __name__ == "__main__":
    unittest.main()
