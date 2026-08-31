import unittest

from src.verification.tiny_learning import run_tiny_learning_verification


class TinyLearningTest(unittest.TestCase):
    def test_tiny_model_overfits_32_examples(self) -> None:
        report = run_tiny_learning_verification()

        self.assertTrue(report.passed, report)
        self.assertEqual(32, report.examples)

    def test_rejects_non_positive_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            run_tiny_learning_verification(steps=0)


if __name__ == "__main__":
    unittest.main()
