from __future__ import annotations

from typing import Any, cast
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.evaluation.value_metrics import ValueMetrics
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.telemetry import InMemoryMetricSink
from src.train.learner import ModelTrainingStats
from src.train.self_play import SelfPlayTrainer, SelfPlayTrainerArgs
from src.train.training_example import TrainingExample


class _FakeNet:
    def eval(self) -> _FakeNet:
        return self

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        del state


class SelfPlayEvaluationTest(unittest.TestCase):
    def test_training_forwards_backend_and_native_threads_to_evaluation(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = _FakeNet()
        args = SelfPlayTrainerArgs(
            start_iteration=0,
            professional_games_training_iterations=0,
            self_play_training_iterations=1,
            temp_threshold=0,
            mcts_args=MCTSArgs(num_simulations=1),
            watch_training_raw_dataset_filepath="unused",
            watch_training_processed_dataset_filepath="unused",
            force_watch_training_raw_dataset_processing=False,
            batch_size=1,
            batch_games=1,
            active_games=1,
            learner_steps_per_iteration=1,
            augment_training=False,
            should_checkpoint=False,
            should_use_arena=True,
            num_arena_games=1,
            eval_iteration_interval=1,
            search_backend="cpp",
            native_worker_threads=3,
        )
        trainer = SelfPlayTrainer(
            game,
            cast(Any, net),
            cast(Any, object()),
            torch.device("cpu"),
            args,
            InMemoryMetricSink(),
        )
        policy = np.full(game.get_action_size(), 1.0 / game.get_action_size())
        example = TrainingExample(game.init_board(), policy, 0.0)
        stats = ModelTrainingStats(
            total_loss=0.0,
            total_policy_loss=0.0,
            total_policy_kl=0.0,
            total_value_loss=0.0,
            total_value_absolute_error=0.0,
            total_value_bias=0.0,
            num_batches=1,
            value_metrics=ValueMetrics(
                calibration_error=0.0,
                negative_outcomes=0,
                draw_outcomes=1,
                positive_outcomes=0,
                negative_mean_prediction=0.0,
                draw_mean_prediction=0.0,
                positive_mean_prediction=0.0,
            ),
            value_loss_weight=1.0,
        )

        with (
            patch.object(
                trainer,
                "build_self_play_training_examples",
                return_value=([example], [], []),
            ),
            patch.object(trainer, "train_model", return_value=stats),
            patch("src.train.self_play.evaluate_training_iteration") as evaluate,
        ):
            trainer.train()

        evaluate.assert_called_once()
        self.assertEqual("cpp", evaluate.call_args.kwargs["search_backend"])
        self.assertEqual(3, evaluate.call_args.kwargs["native_worker_threads"])
        self.assertIs(trainer.game, evaluate.call_args.args[0])
        self.assertIs(trainer.previous_net, evaluate.call_args.args[1])
        self.assertIs(trainer.net, evaluate.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
