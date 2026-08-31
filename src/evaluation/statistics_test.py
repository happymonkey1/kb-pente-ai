import unittest

from src.evaluation.statistics import elo_difference, wilson_interval


class EvaluationStatisticsTest(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self) -> None:
        lower, upper = wilson_interval(60, 100)

        self.assertLess(lower, 0.6)
        self.assertGreater(upper, 0.6)

    def test_empty_wilson_interval_is_uninformative(self) -> None:
        self.assertEqual((0.0, 1.0), wilson_interval(0, 0))

    def test_elo_is_zero_at_even_score_and_antisymmetric(self) -> None:
        self.assertAlmostEqual(0.0, elo_difference(0.5))
        self.assertAlmostEqual(elo_difference(0.75), -elo_difference(0.25))


if __name__ == "__main__":
    unittest.main()
