import unittest

import numpy as np

from src.evaluation.value_metrics import ValueMetricsAccumulator


class ValueMetricsAccumulatorTest(unittest.TestCase):
    def test_reports_calibration_and_complete_outcome_buckets(self) -> None:
        accumulator = ValueMetricsAccumulator(calibration_bins=10)
        accumulator.add(
            np.array([-0.8, 0.0, 0.8], dtype=np.float32),
            np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        )

        metrics = accumulator.finish()

        self.assertAlmostEqual(1.0 / 15.0, metrics.calibration_error)
        self.assertEqual(1, metrics.negative_outcomes)
        self.assertEqual(1, metrics.draw_outcomes)
        self.assertEqual(1, metrics.positive_outcomes)
        self.assertAlmostEqual(-0.8, metrics.negative_mean_prediction, places=6)
        self.assertAlmostEqual(0.0, metrics.draw_mean_prediction)
        self.assertAlmostEqual(0.8, metrics.positive_mean_prediction, places=6)

    def test_accumulates_multiple_batches(self) -> None:
        accumulator = ValueMetricsAccumulator()
        accumulator.add(np.array([-1.0]), np.array([-1.0]))
        accumulator.add(np.array([1.0]), np.array([1.0]))

        metrics = accumulator.finish()

        self.assertEqual(2, metrics.negative_outcomes + metrics.positive_outcomes)
        self.assertEqual(0.0, metrics.calibration_error)

    def test_rejects_shape_mismatch(self) -> None:
        accumulator = ValueMetricsAccumulator()

        with self.assertRaisesRegex(ValueError, "same shape"):
            accumulator.add(np.array([0.0]), np.array([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
