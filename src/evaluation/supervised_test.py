import unittest

import numpy as np
import torch

from src.evaluation.supervised import (
    SupervisedEvaluationStats,
    evaluate_supervised_examples,
    supervised_evaluation_metrics,
)
from src.evaluation.value_metrics import ValueMetrics
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.model.model_v1 import PenteNet
from src.train.training_example import TrainingExample


class SupervisedEvaluationTest(unittest.TestCase):
    def test_reports_finite_held_out_metrics(self) -> None:
        torch.manual_seed(8)
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        position = game.init_board()
        policy = np.zeros(25, dtype=np.float32)
        policy[3] = 1.0
        examples = [TrainingExample(position, policy, 1.0)]

        stats = evaluate_supervised_examples(net, game, examples, batch_size=1)

        self.assertEqual(1, stats.examples)
        self.assertTrue(np.isfinite(stats.policy_cross_entropy))
        self.assertIn(stats.policy_top_one_accuracy, (0.0, 1.0))
        self.assertIn(stats.policy_top_five_accuracy, (0.0, 1.0))
        self.assertTrue(np.isfinite(stats.value_mse))
        self.assertFalse(net.training)
        expected: dict[str, int | float] = {
            "heldout_examples": stats.examples,
            "heldout_policy_cross_entropy": stats.policy_cross_entropy,
            "heldout_policy_top_one_accuracy": stats.policy_top_one_accuracy,
            "heldout_policy_top_five_accuracy": stats.policy_top_five_accuracy,
            "heldout_value_mse": stats.value_mse,
        }
        expected.update(stats.value_metrics.to_metrics("heldout_value"))
        self.assertEqual(expected, supervised_evaluation_metrics(stats, prefix="heldout"))

    def test_metric_prefix_cannot_be_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "prefix"):
            supervised_evaluation_metrics(
                SupervisedEvaluationStats(
                    1,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    ValueMetrics(0.0, 1, 0, 0, -1.0, 0.0, 0.0),
                ),
                prefix="",
            )


if __name__ == "__main__":
    unittest.main()
