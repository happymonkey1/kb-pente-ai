import unittest
from unittest.mock import patch

from src.telemetry import InMemoryMetricSink
from src.train.self_play_health import (
    SelfPlayHealthFailurePolicy,
    SelfPlayHealthThresholds,
    evaluate_self_play_health,
    validate_and_emit_self_play_health,
    validate_self_play_health,
)


class SelfPlayHealthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics: dict[str, int | float] = {
            "mcts_invalid_policy_fallbacks": 0,
            "mcts_zero_visit_fallbacks": 0,
            "steady_state_mean_batch_occupancy": 0.95,
            "mean_root_children_visited": 12.0,
            "search_collapse_rate": 0.01,
            "self_play_gpu_utilization_sampling_errors": 0,
        }
        self.thresholds = SelfPlayHealthThresholds(
            minimum_steady_state_batch_occupancy=0.8,
            minimum_mean_root_children_visited=4.0,
            maximum_search_collapse_rate=0.25,
            maximum_zero_visit_fallbacks=0,
        )

    def test_accepts_healthy_cohort(self) -> None:
        validate_self_play_health(self.metrics, self.thresholds)

    def test_reports_every_failed_invariant(self) -> None:
        unhealthy = {
            **self.metrics,
            "mcts_invalid_policy_fallbacks": 1,
            "mcts_zero_visit_fallbacks": 2,
            "steady_state_mean_batch_occupancy": 0.5,
            "mean_root_children_visited": 2.0,
            "search_collapse_rate": 0.5,
            "self_play_gpu_utilization_sampling_errors": 1,
        }

        with self.assertRaisesRegex(RuntimeError, "invalid_policy.*zero_visit.*occupancy"):
            validate_self_play_health(unhealthy, self.thresholds)

    def test_zero_visit_gate_can_be_disabled_for_tiny_smokes(self) -> None:
        self.metrics["mcts_zero_visit_fallbacks"] = 10

        validate_self_play_health(
            self.metrics,
            SelfPlayHealthThresholds(),
        )

    def test_evaluation_returns_failures_without_enforcement(self) -> None:
        report = evaluate_self_play_health(
            {**self.metrics, "mean_root_children_visited": 1.0},
            self.thresholds,
        )

        self.assertFalse(report.healthy)
        self.assertEqual(
            ("mean_root_children_visited=1 is below 4",),
            report.failures,
        )

    def test_warning_policy_emits_and_continues(self) -> None:
        sink = InMemoryMetricSink()

        report = validate_and_emit_self_play_health(
            sink,
            7,
            "cuda",
            8,
            1.5,
            {**self.metrics, "mean_root_children_visited": 1.0},
            self.thresholds,
            SelfPlayHealthFailurePolicy.WARN,
        )

        self.assertFalse(report.healthy)
        self.assertEqual(1, len(sink.records))
        metrics = sink.records[0]["metrics"]
        assert isinstance(metrics, dict)
        self.assertEqual("warn", metrics["health_failure_policy"])
        self.assertTrue(metrics["training_continues"])

    def test_error_policy_emits_then_raises(self) -> None:
        sink = InMemoryMetricSink()

        with self.assertRaisesRegex(RuntimeError, "mean_root_children_visited"):
            validate_and_emit_self_play_health(
                sink,
                7,
                "cuda",
                8,
                1.5,
                {**self.metrics, "mean_root_children_visited": 1.0},
                self.thresholds,
                SelfPlayHealthFailurePolicy.ERROR,
            )

        self.assertEqual(1, len(sink.records))
        metrics = sink.records[0]["metrics"]
        assert isinstance(metrics, dict)
        self.assertEqual("error", metrics["health_failure_policy"])
        self.assertFalse(metrics["training_continues"])

    def test_unexpected_health_evaluation_errors_propagate(self) -> None:
        sink = InMemoryMetricSink()

        with patch(
            "src.train.self_play_health.evaluate_self_play_health",
            side_effect=RuntimeError("search evaluator failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "search evaluator failed"):
                validate_and_emit_self_play_health(
                    sink,
                    7,
                    "cuda",
                    8,
                    1.5,
                    self.metrics,
                    self.thresholds,
                    SelfPlayHealthFailurePolicy.WARN,
                )

        self.assertEqual([], sink.records)


if __name__ == "__main__":
    unittest.main()
