import unittest

from src.evaluation.tactical import TacticalSuiteStats
from src.train.arena import ArenaStats
from src.verification.model_match import summarize_model_match


class ModelMatchVerificationTest(unittest.TestCase):
    def test_passes_stronger_candidate_without_tactical_regression(self) -> None:
        stats = ArenaStats(
            p1_wins=75,
            p2_wins=20,
            draws=5,
            avg_moves=50.0,
            player_one_color_wins=50,
            player_two_color_wins=45,
            p1_as_player_one_wins=40,
            p1_as_player_two_wins=35,
            p2_as_player_one_wins=10,
            p2_as_player_two_wins=10,
            opening_plies=4,
            unique_openings=50,
            paired_openings=50,
            p1_pair_wins=35,
            p1_pair_losses=10,
            pair_ties=5,
        )
        candidate_tactical = self._tactical(accuracy=1.0, mass=0.7)
        baseline_tactical = self._tactical(accuracy=0.8, mass=0.6)

        report = summarize_model_match(
            stats,
            elapsed_seconds=20.0,
            candidate_tactical=candidate_tactical,
            baseline_tactical=baseline_tactical,
        )

        self.assertTrue(report.passed, report.failures)
        self.assertGreater(report.candidate_paired_win_rate_95pct_lower, 0.5)
        self.assertEqual(50, report.unique_openings)

    def test_reports_strength_color_and_tactical_failures(self) -> None:
        stats = ArenaStats(
            p1_wins=4,
            p2_wins=6,
            draws=0,
            avg_moves=20.0,
            player_one_color_wins=6,
            player_two_color_wins=4,
            p1_as_player_one_wins=4,
            p1_as_player_two_wins=0,
            p2_as_player_one_wins=2,
            p2_as_player_two_wins=4,
            opening_plies=4,
            unique_openings=5,
            paired_openings=5,
            p1_pair_wins=1,
            p1_pair_losses=4,
            pair_ties=0,
        )

        report = summarize_model_match(
            stats,
            elapsed_seconds=1.0,
            candidate_tactical=self._tactical(accuracy=0.5, mass=0.3),
            baseline_tactical=self._tactical(accuracy=1.0, mass=0.6),
        )

        self.assertFalse(report.passed)
        self.assertEqual(5, len(report.failures))

    @staticmethod
    def _tactical(accuracy: float, mass: float) -> TacticalSuiteStats:
        return TacticalSuiteStats(
            cases=6,
            correct=round(6 * accuracy),
            accuracy=accuracy,
            mean_expected_policy_mass=mass,
            category_accuracy={"fixture": accuracy},
        )


if __name__ == "__main__":
    unittest.main()
