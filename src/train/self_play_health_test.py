import unittest

from src.train.self_play_health import (
    SelfPlayHealthThresholds,
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


if __name__ == "__main__":
    unittest.main()
