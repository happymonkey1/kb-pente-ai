from __future__ import annotations

import unittest

from src.monitoring.models import ReplaySample, TelemetryRecord


class TelemetryRecordTest(unittest.TestCase):
    def test_parses_complete_scalar_metrics(self) -> None:
        record = TelemetryRecord.from_object(
            {
                "schema_version": 1,
                "timestamp_unix": 100.5,
                "event": "training_iteration",
                "step": 3,
                "metrics": {
                    "loss": 1.25,
                    "games": 8,
                    "phase": "self_play",
                    "healthy": True,
                    "optional": None,
                },
            }
        )

        self.assertEqual("training_iteration", record.event)
        self.assertEqual(3, record.step)
        self.assertEqual(1.25, record.metrics["loss"])

    def test_rejects_unsupported_schema_and_non_finite_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported telemetry schema"):
            TelemetryRecord.from_object(
                {
                    "schema_version": 2,
                    "timestamp_unix": 1.0,
                    "event": "event",
                    "step": 1,
                    "metrics": {},
                }
            )
        with self.assertRaisesRegex(ValueError, "finite scalar"):
            TelemetryRecord.from_object(
                {
                    "schema_version": 1,
                    "timestamp_unix": 1.0,
                    "event": "event",
                    "step": 1,
                    "metrics": {"loss": float("inf")},
                }
            )
        with self.assertRaisesRegex(ValueError, "finite scalar"):
            TelemetryRecord.from_object(
                {
                    "schema_version": 1,
                    "timestamp_unix": 1.0,
                    "event": "event",
                    "step": 1,
                    "metrics": {"count": 10**10_000},
                }
            )


class ReplaySampleTest(unittest.TestCase):
    def test_round_trips_a_replay_sample(self) -> None:
        payload = {
            "schema_version": 1,
            "run_id": "training",
            "game_id": "iteration-1-game-0",
            "recorded_at_unix": 100.0,
            "board_size": 5,
            "ruleset": "freestyle",
            "actions": [0, 1, 6],
            "winner": 1,
            "win_reason": "line",
        }

        sample = ReplaySample.from_object(payload)

        self.assertEqual(payload, sample.to_dict())

    def test_rejects_invalid_action_and_boolean_winner(self) -> None:
        base = {
            "schema_version": 1,
            "run_id": "training",
            "game_id": "game",
            "recorded_at_unix": 100.0,
            "board_size": 5,
            "ruleset": "freestyle",
            "actions": [0],
            "winner": None,
            "win_reason": None,
        }
        with self.assertRaisesRegex(ValueError, "outside the board"):
            ReplaySample.from_object({**base, "actions": [25]})
        with self.assertRaisesRegex(ValueError, "winner must be an integer"):
            ReplaySample.from_object({**base, "winner": True})


if __name__ == "__main__":
    unittest.main()
