import unittest

import numpy as np
import torch

from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.model.model_v1 import PenteNet
from src.train.learner import train_policy_value_model
from src.train.training_example import TrainingExample


class TrainPolicyValueModelTest(unittest.TestCase):
    def test_applies_declared_value_loss_weight(self) -> None:
        torch.manual_seed(23)
        device = torch.device("cpu")
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(device, 5, 25, 1, 8, 16)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.0)
        policy = np.zeros(25, dtype=np.float32)
        policy[0] = 1.0
        example = TrainingExample(game.init_board(), policy, 1.0)

        stats = train_policy_value_model(
            net,
            optimizer,
            device,
            game,
            [example],
            batch_size=1,
            augment=False,
            value_loss_weight=0.25,
        )

        self.assertEqual(0.25, stats.value_loss_weight)
        self.assertAlmostEqual(
            stats.total_policy_loss + 0.25 * stats.total_value_loss,
            stats.total_loss,
            places=6,
        )

    def test_rejects_negative_value_loss_weight(self) -> None:
        device = torch.device("cpu")
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(device, 5, 25, 1, 8, 16)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.0)

        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            train_policy_value_model(
                net,
                optimizer,
                device,
                game,
                [],
                batch_size=1,
                augment=False,
                value_loss_weight=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
