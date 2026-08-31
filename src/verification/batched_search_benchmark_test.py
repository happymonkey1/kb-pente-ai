import unittest

import torch

from src.game.pente.rules import PenteRuleset
from src.verification.batched_search_benchmark import (
    BatchedSearchBenchmarkConfig,
    run_batched_search_benchmark,
)


class BatchedSearchBenchmarkTest(unittest.TestCase):
    def test_compares_reference_and_batched_search(self) -> None:
        report = run_batched_search_benchmark(
            BatchedSearchBenchmarkConfig(
                board_size=5,
                ruleset=PenteRuleset.FREESTYLE,
                games=2,
                simulations=3,
                repeats=1,
                warmup_batches=0,
                model_blocks=1,
                model_channels=4,
                model_hidden_size=8,
                minimum_speedup=0.0,
            ),
            torch.device("cpu"),
        )

        self.assertTrue(report.passed, report.failures)
        self.assertEqual(report.reference_leaf_evaluations, report.batched_leaf_evaluations)
        self.assertEqual(1.0, report.selected_action_agreement)
        self.assertLessEqual(report.maximum_policy_difference, 0.05)
        self.assertGreater(report.max_inference_batch_size, 1)
        self.assertGreater(report.reference_leaf_evaluations_per_second, 0.0)
        self.assertGreater(report.batched_leaf_evaluations_per_second, 0.0)


if __name__ == "__main__":
    unittest.main()
