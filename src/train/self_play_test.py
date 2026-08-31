from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from typing import Sequence

import numpy as np
import torch

from src.game.game import Game, TerminalResult
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.model.model_v1 import PenteNet
from src.telemetry import InMemoryMetricSink
from src.train.self_play import SelfPlayTrainer, SelfPlayTrainerArgs, finalize_training_examples
from src.train.self_play_generation import SelfPlayGenerator
from src.train.self_play_metrics import collect_self_play_metrics
from src.train.replay_buffer import ReplayBuffer
from src.train.training_example import TrainingExample


class FirstLegalEvaluator:
    def __init__(self, game: PenteGame) -> None:
        self.game = game

    def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        policies, values = self.evaluate_batch([position])
        return policies[0], float(values[0])

    def evaluate_batch(
        self,
        positions: Sequence[PenteBoard],
    ) -> tuple[np.ndarray, np.ndarray]:
        policies = np.zeros(
            (len(positions), self.game.get_action_size()),
            dtype=np.float64,
        )
        for index, position in enumerate(positions):
            legal = np.flatnonzero(
                self.game.get_valid_moves(position, position.current_player)
            )
            policies[index, legal[0]] = 1.0
        return policies, np.zeros(len(positions), dtype=np.float64)


class SelfPlayTest(unittest.TestCase):
    def test_training_example_rejects_occupied_policy_target(self) -> None:
        position = PenteBoard.new_board(5).apply_move(Game.PLAYER_ONE, (0, 0))
        policy = np.zeros(25, dtype=np.float32)
        policy[0] = 1.0

        with self.assertRaisesRegex(ValueError, "occupied"):
            TrainingExample(position, policy, 0.0)

    def test_final_values_follow_side_to_move_perspective(self) -> None:
        first = PenteBoard.new_board(5)
        second = first.apply_move(Game.PLAYER_ONE, (0, 0))
        first_policy = np.full(25, 1.0 / 25, dtype=np.float32)
        second_policy = np.full(25, 1.0 / 24, dtype=np.float32)
        second_policy[0] = 0.0

        examples = finalize_training_examples(
            [(first, first_policy), (second, second_policy)],
            TerminalResult.win(Game.PLAYER_ONE),
        )

        self.assertEqual(1.0, examples[0].value)
        self.assertEqual(-1.0, examples[1].value)

    def test_draw_values_are_zero(self) -> None:
        position = PenteBoard.new_board(5)
        policy = np.full(25, 1.0 / 25, dtype=np.float32)

        examples = finalize_training_examples([(position, policy)], TerminalResult.draw())

        self.assertEqual(0.0, examples[0].value)

    def test_training_restores_train_mode_and_emits_finite_stats(self) -> None:
        torch.manual_seed(2)
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        position_one = game.init_board()
        position_two, _ = game.apply_action(position_one, position_one.current_player, 0)
        policy_one = np.full(25, 1.0 / 25, dtype=np.float32)
        policy_two = np.full(25, 1.0 / 24, dtype=np.float32)
        policy_two[0] = 0.0
        examples = [
            TrainingExample(position_one, policy_one, 1.0),
            TrainingExample(position_two, policy_two, -1.0),
        ]

        with tempfile.TemporaryDirectory() as directory:
            args = SelfPlayTrainerArgs(
                start_iteration=0,
                professional_games_training_iterations=0,
                self_play_training_iterations=1,
                temp_threshold=3,
                mcts_args=MCTSArgs(num_simulations=2),
                watch_training_raw_dataset_filepath="unused",
                watch_training_processed_dataset_filepath="unused",
                force_watch_training_raw_dataset_processing=False,
                batch_size=2,
                batch_games=1,
                checkpoint_dir=directory,
                should_checkpoint=False,
                augment_training=False,
            )
            trainer = SelfPlayTrainer(
                game,
                net,
                optimizer,
                torch.device("cpu"),
                args,
                InMemoryMetricSink(),
            )
            net.eval()

            stats = trainer.train_model(examples)

        self.assertTrue(net.training)
        self.assertEqual(1, stats.num_batches)
        self.assertTrue(np.isfinite(stats.total_loss))

    def test_self_play_batches_games_and_preserves_example_contract(self) -> None:
        torch.manual_seed(3)
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        args = SelfPlayTrainerArgs(
            start_iteration=0,
            professional_games_training_iterations=0,
            self_play_training_iterations=1,
            temp_threshold=3,
            mcts_args=MCTSArgs(
                num_simulations=2,
                root_dirichlet_alpha=0.3,
            ),
            watch_training_raw_dataset_filepath="unused",
            watch_training_processed_dataset_filepath="unused",
            force_watch_training_raw_dataset_processing=False,
            batch_size=2,
            batch_games=4,
            active_games=2,
            should_checkpoint=False,
            seed=4,
        )
        trainer = SelfPlayTrainer(game, net, optimizer, torch.device("cpu"), args)

        examples, games, batches = trainer.build_self_play_training_examples()

        self.assertEqual(4, len(games))
        self.assertEqual(sum(len(played.examples) for played in games), len(examples))
        self.assertGreater(max(batch.max_inference_batch_size for batch in batches), 1)
        self.assertLessEqual(max(batch.root_count for batch in batches), 2)
        self.assertEqual(2, max(batch.root_count for batch in batches))
        self.assertFalse(net.training)
        for played in games:
            for example in played.examples:
                action_probabilities = example.policy
                occupied = example.position.board.reshape(-1) != 0
                self.assertAlmostEqual(0.0, float(action_probabilities[occupied].sum()))
                expected = 0.0 if played.winner is None else float(played.winner * example.position.current_player)
                self.assertEqual(expected, example.value)

    def test_professional_training_emits_baseline_before_updated_metrics(self) -> None:
        torch.manual_seed(18)
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        policy = np.zeros(25, dtype=np.float32)
        policy[0] = 1.0
        example = TrainingExample(game.init_board(), policy, 1.0)
        sink = InMemoryMetricSink()
        args = SelfPlayTrainerArgs(
            start_iteration=0,
            professional_games_training_iterations=1,
            self_play_training_iterations=0,
            temp_threshold=0,
            mcts_args=MCTSArgs(num_simulations=2),
            watch_training_raw_dataset_filepath="unused",
            watch_training_processed_dataset_filepath="unused",
            force_watch_training_raw_dataset_processing=False,
            batch_size=1,
            batch_games=1,
            should_checkpoint=False,
            augment_training=False,
            learner_steps_per_iteration=1,
        )
        trainer = SelfPlayTrainer(
            game,
            net,
            optimizer,
            torch.device("cpu"),
            args,
            sink,
        )

        def load_examples() -> list[TrainingExample]:
            trainer.professional_validation_examples = [example]
            return [example]

        with patch.object(
            trainer,
            "load_professional_training_examples",
            side_effect=load_examples,
        ):
            trainer.train()

        self.assertEqual(
            ["professional_validation_baseline", "training_iteration"],
            [record["event"] for record in sink.records],
        )
        baseline = sink.records[0]["metrics"]
        updated = sink.records[1]["metrics"]
        assert isinstance(baseline, dict)
        assert isinstance(updated, dict)
        self.assertEqual(1, baseline["professional_validation_examples"])
        self.assertEqual(1, updated["professional_validation_examples"])

    def test_resume_requires_replay_or_explicit_professional_seed(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            args = SelfPlayTrainerArgs(
                start_iteration=1,
                professional_games_training_iterations=0,
                self_play_training_iterations=2,
                temp_threshold=0,
                mcts_args=MCTSArgs(num_simulations=2),
                watch_training_raw_dataset_filepath="unused",
                watch_training_processed_dataset_filepath="unused",
                force_watch_training_raw_dataset_processing=False,
                checkpoint_dir=directory,
                should_checkpoint=False,
                training_run_id="resume-test",
                expected_replay_generation=1,
            )

            with self.assertRaisesRegex(ValueError, "requires a compatible replay"):
                SelfPlayTrainer(
                    game,
                    net,
                    optimizer,
                    torch.device("cpu"),
                    args,
                )

    def test_resume_can_explicitly_seed_replay_from_professional_data(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        policy = np.zeros(25, dtype=np.float32)
        policy[0] = 1.0
        example = TrainingExample(game.init_board(), policy, 1.0)
        sink = InMemoryMetricSink()
        with tempfile.TemporaryDirectory() as directory:
            args = SelfPlayTrainerArgs(
                start_iteration=1,
                professional_games_training_iterations=0,
                self_play_training_iterations=1,
                temp_threshold=0,
                mcts_args=MCTSArgs(num_simulations=2),
                watch_training_raw_dataset_filepath="unused",
                watch_training_processed_dataset_filepath="unused",
                force_watch_training_raw_dataset_processing=False,
                checkpoint_dir=directory,
                should_checkpoint=False,
                max_training_examples=2,
                seed_replay_from_professional=True,
                training_run_id="seed-test",
                expected_replay_generation=1,
            )
            trainer = SelfPlayTrainer(
                game,
                net,
                optimizer,
                torch.device("cpu"),
                args,
                sink,
            )

            with patch.object(
                trainer,
                "load_professional_training_examples",
                return_value=[example],
            ):
                trainer.train()

        self.assertEqual([example], trainer.replay.examples())
        self.assertEqual("replay_seed", sink.records[0]["event"])

    def test_initial_checkpoint_has_matching_empty_replay_snapshot(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            args = SelfPlayTrainerArgs(
                start_iteration=0,
                professional_games_training_iterations=0,
                self_play_training_iterations=0,
                temp_threshold=0,
                mcts_args=MCTSArgs(num_simulations=2),
                watch_training_raw_dataset_filepath="unused",
                watch_training_processed_dataset_filepath="unused",
                force_watch_training_raw_dataset_processing=False,
                checkpoint_dir=directory,
            )
            trainer = SelfPlayTrainer(
                game,
                net,
                optimizer,
                torch.device("cpu"),
                args,
            )

            trainer.train()

            replay = ReplayBuffer.load(
                f"{directory}/replay-latest.pkl",
                expected_training_run_id=trainer.training_run_id,
                expected_board_size=5,
                expected_ruleset="freestyle",
            )
            self.assertTrue(
                (Path(directory) / net.get_checkpoint_file_name(0)).exists()
            )
            self.assertTrue((Path(directory) / "replay-0.pkl").exists())

        self.assertEqual(0, replay.snapshot_generation)
        self.assertEqual(0, len(replay))

    def test_resume_prefers_checkpoint_generation_over_newer_latest_replay(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            replay = ReplayBuffer(10)
            replay.save(
                f"{directory}/replay-1.pkl",
                generation=1,
                training_run_id="run-id",
                board_size=5,
                ruleset="freestyle",
            )
            replay.save(
                f"{directory}/replay-latest.pkl",
                generation=5,
                training_run_id="run-id",
                board_size=5,
                ruleset="freestyle",
            )
            args = SelfPlayTrainerArgs(
                start_iteration=1,
                professional_games_training_iterations=0,
                self_play_training_iterations=2,
                temp_threshold=0,
                mcts_args=MCTSArgs(num_simulations=2),
                watch_training_raw_dataset_filepath="unused",
                watch_training_processed_dataset_filepath="unused",
                force_watch_training_raw_dataset_processing=False,
                checkpoint_dir=directory,
                max_training_examples=10,
                training_run_id="run-id",
                expected_replay_generation=1,
            )

            trainer = SelfPlayTrainer(
                game,
                net,
                optimizer,
                torch.device("cpu"),
                args,
            )

        self.assertEqual(1, trainer.replay.snapshot_generation)

    def test_self_play_metrics_emit_collapsed_search_alarm_and_batch_tail(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        generator = SelfPlayGenerator(
            game,
            FirstLegalEvaluator(game),
            MCTSArgs(num_simulations=16, root_noise_epsilon=0.0),
            temp_threshold=0,
            rng=np.random.default_rng(17),
        )

        games, batches = generator.play_games(1)
        metrics = collect_self_play_metrics(games, batches, elapsed_seconds=1.0)

        self.assertGreater(metrics["search_collapse_eligible_roots"], 0)
        self.assertGreater(metrics["search_collapsed_roots"], 0)
        self.assertEqual(1, metrics["search_collapse_detected"])
        self.assertGreater(metrics["search_collapse_rate"], 0.0)
        self.assertLessEqual(metrics["search_collapse_rate"], 1.0)
        self.assertEqual(1, metrics["min_inference_batch_size"])
        self.assertEqual(1.0, metrics["median_inference_batch_size"])
        self.assertEqual(1.0, metrics["p95_inference_batch_size"])
        self.assertEqual(1, metrics["active_game_target"])
        self.assertEqual(1.0, metrics["steady_state_mean_batch_occupancy"])


if __name__ == "__main__":
    unittest.main()
