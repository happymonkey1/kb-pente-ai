from __future__ import annotations

import unittest

import numpy as np
import torch

from src.game.pente.pente_game import PenteGame
from src.game.pente.pente_board import PenteBoard
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.mcts.native_backend import NativeSearchBackend
from src.model.model_v1 import PenteNet
from src.train.arena import Arena
from src.train.native_player import NativeMCTSPlayer
from src.train.player import Player
from src.train.self_play_generation import SelfPlayGenerator
from src.train.self_play_metrics import collect_self_play_metrics


def _draw_root():
    stones = torch.tensor(
        [
            [1, 1, -1, -1, 1],
            [-1, -1, 1, 1, -1],
            [1, 1, -1, -1, 1],
            [-1, -1, 1, 1, -1],
            [1, -1, 1, -1, 0],
        ],
        dtype=torch.int8,
    )
    return stones, torch.zeros(2, dtype=torch.int16)


class NativeBackendExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extension = __import__("kb_pente_native")
        cls.game = PenteGame(board_size=5, ruleset=PenteRuleset.FREESTYLE)
        cls.net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        cls.net.eval()

    def make_backend(self, simulations: int = 2, capacity: int = 2) -> NativeSearchBackend:
        return NativeSearchBackend(
            self.game,
            self.net,
            MCTSArgs(num_simulations=simulations, root_noise_epsilon=0.0),
            max_active_games=capacity,
            worker_threads=2,
            seed=103,
            pin_memory=False,
            extension=self.extension,
        )

    def test_duplicate_roots_multiwave_completion_and_stable_staging(self) -> None:
        backend = self.make_backend()
        root = self.game.init_board()
        first = backend.add_root(root, temperature=0.0)
        second = backend.add_root(root, temperature=0.0)

        waves = []
        while not backend.complete():
            waves.append(backend.evaluate_wave())
        self.assertEqual([wave.size for wave in waves], [1, 1])
        self.assertEqual([wave.raw_size for wave in waves], [2, 2])
        self.assertEqual(backend.slot_telemetry(first), backend.slot_telemetry(second))
        self.assertEqual(backend.slot_telemetry(first).mean_inference_batch_size, 1.0)
        self.assertEqual(
            backend.deduplication_telemetry()["cumulative"]["eliminated_duplicate_evaluations"],
            2,
        )
        policy = backend.root_policy(first)
        self.assertEqual((25,), policy.shape)
        self.assertEqual(np.float32, policy.dtype)
        self.assertAlmostEqual(1.0, float(policy.sum()), places=5)
        timing = backend.timing_telemetry()
        self.assertGreaterEqual(timing["inference"]["calls"], 2)
        self.assertEqual(2, timing["worker"]["worker_threads"])

    def test_root_advance_terminal_conversion_and_remove(self) -> None:
        backend = self.make_backend(simulations=1, capacity=1)
        root = self.game.init_board()
        slot = backend.add_root(root, temperature=0.0)
        while not backend.complete():
            backend.evaluate_wave()
        backend.advance_root(slot, 0, temperature=0.0)
        self.assertFalse(backend.slot_complete(slot))
        backend.evaluate_wave()
        self.assertTrue(backend.slot_complete(slot))

        draw_stones, draw_captures = _draw_root()
        draw = PenteBoard(
            draw_stones.numpy(),
            draw_captures.numpy(),
            current_player=1,
            ply=24,
        )
        backend.replace_root(slot, draw, temperature=0.0)
        backend.evaluate_wave()
        backend.advance_root(slot, 24, temperature=0.0)
        self.assertEqual(backend.root_terminal(slot).status.value, "draw")
        self.assertTrue(backend.slot_complete(slot))
        backend.remove(slot)
        self.assertEqual(backend.active_count, 0)

    def test_terminal_advance_updates_cached_completion_immediately(self) -> None:
        class TerminalEvaluator:
            device = torch.device("cpu")

            def evaluate_features(
                self,
                inputs: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                policies = torch.zeros((inputs.shape[0], 25), dtype=torch.float32)
                policies[:, 24] = 1.0
                return policies, torch.zeros(inputs.shape[0], dtype=torch.float32)

        backend = NativeSearchBackend(
            self.game,
            TerminalEvaluator(),
            MCTSArgs(num_simulations=1, root_noise_epsilon=0.0),
            max_active_games=1,
            worker_threads=1,
            pin_memory=False,
            extension=self.extension,
        )
        draw_stones, draw_captures = _draw_root()
        root = PenteBoard(
            draw_stones.numpy(),
            draw_captures.numpy(),
            current_player=1,
            ply=24,
        )
        slot = backend.add_root(root, temperature=0.0)
        selected = backend.evaluate_wave()
        self.assertEqual(1, selected.size)

        backend.advance_root(slot, 24, temperature=0.0)

        self.assertTrue(backend.slot_complete(slot))
        self.assertEqual(backend.root_terminal(slot).status.value, "draw")

    def test_removed_slot_completion_preserves_index_error(self) -> None:
        backend = self.make_backend(simulations=1, capacity=1)
        slot = backend.add_root(self.game.init_board(), temperature=0.0)
        backend.evaluate_wave()
        backend.remove(slot)

        with self.assertRaises(IndexError):
            backend.slot_complete(slot)

    def test_observe_action_updates_root_mirror_and_continues_search(self) -> None:
        backend = self.make_backend(simulations=1, capacity=1)
        root = self.game.init_board()
        slot = backend.add_root(root, temperature=0.0)
        expected, _ = self.game.apply_action(root, root.current_player, 0)

        stats = backend.observe_action(
            slot,
            0,
            temperature=0.0,
            add_root_noise=False,
        )
        self.assertFalse(stats["reused_subtree"])
        self.assertEqual(backend._roots[slot].state_key(), expected.state_key())
        self.assertEqual(backend._batch_sizes[slot], [])
        self.assertFalse(backend.slot_complete(slot))

        backend.evaluate_wave()
        self.assertTrue(backend.slot_complete(slot))
        policy = backend.root_policy(slot)
        self.assertAlmostEqual(float(policy.sum()), 1.0)
        self.assertEqual(float(policy[0]), 0.0)

    def test_self_play_matches_python_on_deterministic_freestyle(self) -> None:
        class DualBoundaryEvaluator:
            device = torch.device("cpu")

            def __init__(self, game: PenteGame) -> None:
                self.game = game

            def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
                policies, values = self.evaluate_batch([position])
                return policies[0], float(values[0])

            def evaluate_batch(
                self,
                positions: list[PenteBoard],
            ) -> tuple[np.ndarray, np.ndarray]:
                action_size = self.game.get_action_size()
                return (
                    np.full(
                        (len(positions), action_size),
                        1.0 / action_size,
                        dtype=np.float32,
                    ),
                    np.zeros(len(positions), dtype=np.float32),
                )

            def evaluate_features(
                self,
                inputs: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                rows = inputs.shape[0]
                action_size = self.game.get_action_size()
                return (
                    torch.full(
                        (rows, action_size),
                        1.0 / action_size,
                        dtype=torch.float32,
                    ),
                    torch.zeros(rows, dtype=torch.float32),
                )

        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        evaluator = DualBoundaryEvaluator(game)
        args = MCTSArgs(num_simulations=1, root_noise_epsilon=0.0)
        python_generator = SelfPlayGenerator(
            game,
            evaluator,
            args,
            temp_threshold=0,
            rng=np.random.default_rng(71),
            search_backend="python",
        )

        def native_factory(*factory_args, **factory_kwargs):
            return NativeSearchBackend(
                *factory_args,
                extension=self.extension,
                pin_memory=False,
                **factory_kwargs,
            )

        native_generator = SelfPlayGenerator(
            game,
            evaluator,
            args,
            temp_threshold=0,
            rng=np.random.default_rng(71),
            search_backend="cpp",
            native_worker_threads=2,
            _native_backend_factory=native_factory,
        )
        python_games, python_batches = python_generator.play_games(2, 2)
        native_games, native_batches = native_generator.play_games(2, 2)

        self.assertEqual(
            [(played.actions, played.winner, played.win_reason) for played in python_games],
            [(played.actions, played.winner, played.win_reason) for played in native_games],
        )
        for python_game, native_game in zip(python_games, native_games):
            self.assertEqual(len(python_game.examples), len(native_game.examples))
            for python_example, native_example in zip(
                python_game.examples,
                native_game.examples,
            ):
                np.testing.assert_array_equal(
                    python_example.position.board,
                    native_example.position.board,
                )
                np.testing.assert_array_equal(
                    python_example.position.captures,
                    native_example.position.captures,
                )
                self.assertEqual(
                    python_example.position.current_player,
                    native_example.position.current_player,
                )
                self.assertEqual(python_example.position.ply, native_example.position.ply)
                self.assertEqual(
                    python_example.position.last_action,
                    native_example.position.last_action,
                )
                np.testing.assert_allclose(
                    python_example.policy,
                    native_example.policy,
                    rtol=0.0,
                    atol=1e-6,
                )
                self.assertAlmostEqual(python_example.value, native_example.value)

        self.assertEqual(len(python_batches), len(native_batches))
        self.assertEqual(
            sum(batch.evaluation_requests for batch in python_batches),
            sum(batch.evaluation_requests for batch in native_batches),
        )
        self.assertEqual(
            sum(batch.unique_evaluations for batch in python_batches),
            sum(batch.unique_evaluations for batch in native_batches),
        )
        self.assertEqual(
            sum(batch.selected_leaves for batch in python_batches),
            sum(batch.selected_leaves for batch in native_batches),
        )
        self.assertEqual(
            sum(batch.evaluator_calls for batch in python_batches),
            sum(batch.evaluator_calls for batch in native_batches),
        )
        for batch in native_batches:
            self.assertEqual(batch.native_worker_threads, 2)
            self.assertGreaterEqual(batch.native_select_seconds, 0.0)
            self.assertGreaterEqual(batch.native_deduplication_seconds, 0.0)
            self.assertGreaterEqual(batch.native_feature_encode_seconds, 0.0)
            self.assertGreaterEqual(batch.native_backup_seconds, 0.0)

    def test_pentenet_self_play_drains_native_slots(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        net.eval()
        backends = []

        def native_factory(*factory_args, **factory_kwargs):
            backend = NativeSearchBackend(
                *factory_args,
                extension=self.extension,
                pin_memory=False,
                **factory_kwargs,
            )
            backends.append(backend)
            return backend

        generator = SelfPlayGenerator(
            game,
            net,
            MCTSArgs(num_simulations=1, root_noise_epsilon=0.0),
            temp_threshold=0,
            rng=np.random.default_rng(79),
            search_backend="cpp",
            native_worker_threads=2,
            _native_backend_factory=native_factory,
        )
        games, _ = generator.play_games(1, 1)

        self.assertEqual(1, len(games))
        self.assertEqual(1, len(backends))
        self.assertEqual(0, backends[0].active_count)
        for example in games[0].examples:
            legal = game.get_valid_moves(
                example.position,
                example.position.current_player,
            )
            self.assertEqual(1.0, float(example.policy.sum()))
            self.assertEqual(0.0, float(example.policy[legal == 0].sum()))

    def test_native_arena_player_reuses_a_bounded_tree(self) -> None:
        class FirstLegalPlayer(Player):
            def play(
                self,
                game: PenteGame,
                board: PenteBoard,
                player: int,
                debug: bool = False,
            ) -> int:
                del debug
                return int(np.flatnonzero(game.get_valid_moves(board, player))[0])

        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            torch.device("cpu"),
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=4,
            hidden_fc_size=8,
        )
        net.eval()
        args = MCTSArgs(num_simulations=1, root_noise_epsilon=0.0)
        backends: list[NativeSearchBackend] = []

        def native_factory(*factory_args, **factory_kwargs):
            backend = NativeSearchBackend(
                *factory_args,
                extension=self.extension,
                pin_memory=False,
                **factory_kwargs,
            )
            backends.append(backend)
            return backend

        native = NativeMCTSPlayer(
            net,
            game,
            args,
            seed=83,
            native_worker_threads=2,
            _native_backend_factory=native_factory,
        )
        arena = Arena(FirstLegalPlayer(), native, game)
        stats = arena.play_games(1)

        self.assertGreater(stats.avg_moves, 0.0)
        self.assertEqual(1, len(backends))
        self.assertEqual(1, backends[0].capacity)
        native.reset()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_pinned_transfer_and_wait_timing(self) -> None:
        device = torch.device("cuda")
        net = PenteNet(
            device,
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        net.eval()
        backend = NativeSearchBackend(
            self.game,
            net,
            MCTSArgs(num_simulations=1, root_noise_epsilon=0.0),
            max_active_games=1,
            worker_threads=1,
            seed=103,
            pin_memory=True,
            extension=self.extension,
        )
        backend.add_root(self.game.init_board(), temperature=0.0)
        submission = backend.submit_wave()
        wave = backend.wait_wave(submission)
        self.assertEqual(
            (submission.token, submission.size, submission.raw_size),
            (wave.token, wave.size, wave.raw_size),
        )
        self.assertEqual(1, wave.size)
        timing = backend.inference_timing()
        self.assertEqual(1, timing["calls"])
        self.assertGreaterEqual(timing["host_to_device_seconds"], 0.0)
        self.assertGreaterEqual(timing["model_inference_seconds"], 0.0)
        self.assertGreaterEqual(timing["device_to_host_seconds"], 0.0)
        self.assertGreaterEqual(timing["inference_wait_seconds"], 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_two_cohort_self_play_scheduler(self) -> None:
        torch.manual_seed(103)
        device = torch.device("cuda")
        game = PenteGame(board_size=5, ruleset=PenteRuleset.FREESTYLE)
        net = PenteNet(
            device,
            board_size=5,
            action_size=25,
            num_res_blocks=1,
            num_channels=8,
            hidden_fc_size=16,
        )
        net.eval()
        generator = SelfPlayGenerator(
            game,
            net,
            MCTSArgs(num_simulations=2, root_noise_epsilon=0.0),
            temp_threshold=0,
            rng=np.random.default_rng(103),
            search_backend="cpp",
            native_worker_threads=2,
            native_search_cohorts=2,
        )

        games, batches = generator.play_games(2, 2)
        torch.cuda.synchronize(device)

        self.assertEqual(2, len(games))
        self.assertTrue(batches)
        self.assertTrue(all(game.root_telemetry for game in games))
        metrics = collect_self_play_metrics(games, batches, elapsed_seconds=1.0)
        self.assertEqual(2, metrics["native_search_cohorts"])
        self.assertEqual(2, metrics["active_game_target"])
        self.assertEqual(1, metrics["inference_batch_target"])
        self.assertEqual(2, metrics["native_worker_threads"])
        self.assertEqual(1, metrics["native_worker_threads_per_cohort"])
        self.assertEqual(2, metrics["native_pipeline_max_in_flight"])
        self.assertEqual(
            metrics["native_pipeline_submissions"],
            metrics["native_pipeline_waits"],
        )
        self.assertGreater(metrics["native_pipeline_submissions"], 0)
        self.assertEqual(0, metrics["mcts_invalid_policy_fallbacks"])
        self.assertEqual(0, metrics["mcts_zero_visit_fallbacks"])

        for played in games:
            for example in played.examples:
                legal = game.get_valid_moves(
                    example.position,
                    example.position.current_player,
                )
                self.assertAlmostEqual(1.0, float(example.policy.sum()), places=5)
                self.assertAlmostEqual(
                    0.0,
                    float(example.policy[legal == 0].sum()),
                    places=5,
                )


if __name__ == "__main__":
    unittest.main()
