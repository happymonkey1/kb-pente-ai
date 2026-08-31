import tempfile
import unittest
from pathlib import Path

from src.telemetry import JsonlMetricSink
from src.verification.professional_learning import verify_professional_learning


class ProfessionalLearningVerificationTest(unittest.TestCase):
    def test_passes_when_every_predeclared_threshold_is_met(self) -> None:
        path = self._write_metrics(
            baseline=(6.0, 0.01, 0.04, 1.0),
            final=(5.4, 0.04, 0.10, 0.9),
        )

        report = verify_professional_learning(path)

        self.assertTrue(report.passed)
        self.assertEqual(100, report.examples)
        self.assertAlmostEqual(0.1, report.policy_cross_entropy_reduction)
        self.assertAlmostEqual(0.03, report.policy_top_one_gain)
        self.assertAlmostEqual(0.06, report.policy_top_five_gain)
        self.assertAlmostEqual(0.9, report.value_mse_ratio)
        self.assertEqual((), report.failures)

    def test_reports_each_failed_threshold(self) -> None:
        path = self._write_metrics(
            baseline=(6.0, 0.01, 0.04, 1.0),
            final=(5.9, 0.02, 0.06, 1.1),
        )

        report = verify_professional_learning(path)

        self.assertFalse(report.passed)
        self.assertEqual(4, len(report.failures))

    def test_rejects_mismatched_validation_sets(self) -> None:
        path = self._write_metrics(
            baseline=(6.0, 0.01, 0.04, 1.0),
            final=(5.0, 0.05, 0.10, 0.8),
            final_examples=99,
        )

        with self.assertRaisesRegex(ValueError, "counts differ"):
            verify_professional_learning(path)

    def _write_metrics(
        self,
        baseline: tuple[float, float, float, float],
        final: tuple[float, float, float, float],
        final_examples: int = 100,
    ) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "metrics.jsonl"
        sink = JsonlMetricSink(path)
        sink.emit(
            "professional_validation_baseline",
            0,
            self._metrics(100, baseline),
        )
        sink.emit("training_iteration", 1, self._metrics(final_examples, final))
        return path

    @staticmethod
    def _metrics(
        examples: int,
        values: tuple[float, float, float, float],
    ) -> dict[str, int | float]:
        cross_entropy, top_one, top_five, value_mse = values
        return {
            "professional_validation_examples": examples,
            "professional_validation_policy_cross_entropy": cross_entropy,
            "professional_validation_policy_top_one_accuracy": top_one,
            "professional_validation_policy_top_five_accuracy": top_five,
            "professional_validation_value_mse": value_mse,
        }


if __name__ == "__main__":
    unittest.main()
