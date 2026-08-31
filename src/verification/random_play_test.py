import unittest

from src.train.arena import ArenaStats
from src.verification.random_play import RandomPlayCriteria, summarize_random_play


class RandomPlayVerificationTest(unittest.TestCase):
    def test_passes_statistically_decisive_color_balanced_result(self) -> None:
        stats = ArenaStats(
            p1_wins=75,
            p2_wins=20,
            draws=5,
            avg_moves=40.0,
            player_one_color_wins=52,
            player_two_color_wins=43,
            p1_as_player_one_wins=40,
            p1_as_player_two_wins=35,
            p2_as_player_one_wins=12,
            p2_as_player_two_wins=8,
            opening_plies=4,
            unique_openings=50,
            paired_openings=50,
            p1_pair_wins=35,
            p1_pair_losses=10,
            pair_ties=5,
        )

        report = summarize_random_play(stats, elapsed_seconds=10.0)

        self.assertTrue(report.passed, report.failures)
        self.assertGreater(report.model_decisive_win_rate_95pct_lower, 0.5)
        self.assertEqual(400.0, report.moves_per_second)
        self.assertEqual(40, report.model_as_player_one_wins)
        self.assertEqual(35, report.model_as_player_two_wins)
        self.assertEqual(50, report.unique_openings)

    def test_reports_sample_confidence_and_color_failures(self) -> None:
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

        report = summarize_random_play(
            stats,
            elapsed_seconds=1.0,
            criteria=RandomPlayCriteria(minimum_games=100),
        )

        self.assertFalse(report.passed)
        self.assertEqual(3, len(report.failures))


if __name__ == "__main__":
    unittest.main()
