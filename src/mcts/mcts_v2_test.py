import unittest
from typing import Sequence

import numpy as np

from src.game.game import Game, GameStatus
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTS, MCTSArgs


class UniformEvaluator:
    def __init__(self, action_size: int, value: float = 0.0) -> None:
        self.policy = np.full(action_size, 1.0 / action_size, dtype=np.float64)
        self.value = value
        self.calls = 0

    def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        self.calls += 1
        return np.array(self.policy, copy=True), self.value

    def evaluate_batch(self, positions: Sequence[PenteBoard]) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        policies = np.repeat(self.policy[np.newaxis, :], len(positions), axis=0)
        values = np.full(len(positions), self.value, dtype=np.float64)
        return policies, values


class ImmediateWinEvaluator(UniformEvaluator):
    def __init__(self, game: PenteGame) -> None:
        super().__init__(game.get_action_size())
        self.game = game

    def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        self.calls += 1
        legal = np.flatnonzero(self.game.get_valid_moves(position, position.current_player))
        winning_actions = []
        for action in legal:
            next_position, _ = self.game.apply_action(position, position.current_player, int(action))
            result = self.game.check_game_end(next_position)
            if result.winner == position.current_player:
                winning_actions.append(int(action))
        policy = np.zeros(self.game.get_action_size(), dtype=np.float64)
        selected = winning_actions if winning_actions else [int(action) for action in legal]
        policy[selected] = 1.0 / len(selected)
        return policy, 0.0


class MCTSTest(unittest.TestCase):
    BOARD_SIZE = 5

    def setUp(self) -> None:
        self.game = PenteGame(self.BOARD_SIZE, ruleset=PenteRuleset.FREESTYLE)

    def test_uniform_zero_evaluator_visits_multiple_root_children(self) -> None:
        evaluator = UniformEvaluator(self.game.get_action_size())
        mcts = MCTS(self.game, evaluator, MCTSArgs(num_simulations=32), np.random.default_rng(1))
        root = self.game.init_board()

        policy = mcts.get_action_prob(root, temp=1)
        telemetry = mcts.telemetry(root)

        self.assertGreater(telemetry.root_children_visited, 1)
        self.assertGreater(telemetry.root_visit_entropy, 0.0)
        self.assertLess(telemetry.root_max_visit_share, 1.0)
        self.assertAlmostEqual(1.0, float(policy.sum()))
        self.assertEqual(32, telemetry.simulations)
        self.assertEqual(25, telemetry.root_legal_actions)
        self.assertEqual(31, telemetry.root_edge_visits)
        self.assertTrue(telemetry.root_collapse_eligible)
        self.assertFalse(telemetry.root_search_collapsed)
        self.assertEqual(1.0, telemetry.mean_inference_batch_size)

    def test_collapsed_search_is_reported_for_a_non_forced_root(self) -> None:
        evaluator = UniformEvaluator(self.game.get_action_size())
        evaluator.policy[:] = 0.0
        evaluator.policy[0] = 1.0
        mcts = MCTS(
            self.game,
            evaluator,
            MCTSArgs(num_simulations=16),
            np.random.default_rng(12),
        )
        root = self.game.init_board()

        mcts.get_action_prob(root)
        telemetry = mcts.telemetry(root)

        self.assertEqual(25, telemetry.root_legal_actions)
        self.assertEqual(1, telemetry.root_children_visited)
        self.assertTrue(telemetry.root_collapse_eligible)
        self.assertTrue(telemetry.root_search_collapsed)

    def test_forced_opening_is_not_reported_as_collapsed(self) -> None:
        game = PenteGame(self.BOARD_SIZE, ruleset=PenteRuleset.STANDARD)
        evaluator = UniformEvaluator(game.get_action_size())
        mcts = MCTS(
            game,
            evaluator,
            MCTSArgs(num_simulations=16),
            np.random.default_rng(13),
        )
        root = game.init_board()

        mcts.get_action_prob(root)
        telemetry = mcts.telemetry(root)

        self.assertEqual(1, telemetry.root_legal_actions)
        self.assertEqual(1, telemetry.root_children_visited)
        self.assertFalse(telemetry.root_collapse_eligible)
        self.assertFalse(telemetry.root_search_collapsed)

    def test_unvisited_legal_action_remains_selectable(self) -> None:
        evaluator = UniformEvaluator(self.game.get_action_size())
        mcts = MCTS(self.game, evaluator, MCTSArgs(num_simulations=3), np.random.default_rng(2))
        root = self.game.init_board()

        mcts.get_action_prob(root)

        root_key = self.game.to_string(root)
        visited = [action for action in range(self.game.get_action_size()) if (root_key, action) in mcts.nsa]
        self.assertEqual(2, len(visited))

    def test_illegal_actions_never_receive_visits(self) -> None:
        root = self.game.init_board()
        root, _ = self.game.apply_action(root, root.current_player, 0)
        evaluator = UniformEvaluator(self.game.get_action_size())
        mcts = MCTS(self.game, evaluator, MCTSArgs(num_simulations=24), np.random.default_rng(3))

        policy = mcts.get_action_prob(root)

        root_key = self.game.to_string(root)
        self.assertNotIn((root_key, 0), mcts.nsa)
        self.assertEqual(0.0, policy[0])

    def test_value_backup_changes_perspective_once_per_ply(self) -> None:
        evaluator = UniformEvaluator(self.game.get_action_size(), value=0.5)
        evaluator.policy[:] = 0.0
        evaluator.policy[0] = 1.0
        mcts = MCTS(self.game, evaluator, MCTSArgs(num_simulations=2), np.random.default_rng(4))
        root = self.game.init_board()

        mcts.get_action_prob(root)

        root_key = self.game.to_string(root)
        self.assertAlmostEqual(-0.5, mcts.qsa[(root_key, 0)])

    def test_terminal_draw_is_cached_without_evaluation(self) -> None:
        stones = np.array(
            (
                (1, 1, -1, -1, 1),
                (-1, -1, 1, 1, -1),
                (1, 1, -1, -1, 1),
                (-1, -1, 1, 1, -1),
                (1, -1, 1, -1, 1),
            ),
            dtype=np.int8,
        )
        root = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=Game.PLAYER_TWO)
        evaluator = UniformEvaluator(self.game.get_action_size())
        mcts = MCTS(self.game, evaluator, MCTSArgs(num_simulations=4))

        selection = mcts.select_leaf(root)
        mcts.expand_and_backup(selection)

        self.assertEqual(GameStatus.DRAW, selection.terminal_result.status)
        self.assertEqual(0, evaluator.calls)
        self.assertEqual(0, mcts.evaluator_calls)

    def test_immediate_win_receives_maximum_visit_policy(self) -> None:
        stones = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        stones[2, 0:4] = Game.PLAYER_ONE
        stones[0, 0] = Game.PLAYER_TWO
        stones[0, 2] = Game.PLAYER_TWO
        stones[1, 4] = Game.PLAYER_TWO
        stones[3, 1] = Game.PLAYER_TWO
        root = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=Game.PLAYER_ONE)
        winning_action = 2 * self.BOARD_SIZE + 4
        evaluator = UniformEvaluator(self.game.get_action_size())
        evaluator.policy[:] = 0.001
        evaluator.policy[winning_action] = 1.0
        evaluator.policy /= evaluator.policy.sum()
        mcts = MCTS(self.game, evaluator, MCTSArgs(num_simulations=24), np.random.default_rng(5))

        policy = mcts.get_action_prob(root, temp=0)

        self.assertEqual(1.0, policy[winning_action])
        root_key = self.game.to_string(root)
        self.assertAlmostEqual(1.0, mcts.qsa[(root_key, winning_action)])

    def test_immediate_capture_win_is_selected(self) -> None:
        stones = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        stones[2, 0] = Game.PLAYER_ONE
        stones[2, 1:3] = Game.PLAYER_TWO
        stones[0, 0] = Game.PLAYER_ONE
        root = PenteBoard(
            stones,
            np.array((4, 0), dtype=np.int16),
            current_player=Game.PLAYER_ONE,
        )
        winning_action = 2 * self.BOARD_SIZE + 3
        evaluator = ImmediateWinEvaluator(self.game)
        mcts = MCTS(self.game, evaluator, MCTSArgs(num_simulations=24), np.random.default_rng(9))

        policy = mcts.get_action_prob(root, temp=0)

        self.assertEqual(1.0, policy[winning_action])

    def test_search_blocks_opponents_immediate_line_win(self) -> None:
        game = PenteGame(7, ruleset=PenteRuleset.FREESTYLE)
        stones = np.zeros((7, 7), dtype=np.int8)
        stones[3, 1:5] = Game.PLAYER_TWO
        stones[3, 0] = Game.PLAYER_ONE
        stones[0, 0] = Game.PLAYER_ONE
        stones[1, 2] = Game.PLAYER_ONE
        stones[5, 6] = Game.PLAYER_ONE
        root = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=Game.PLAYER_ONE)
        blocking_action = 3 * 7 + 5
        evaluator = ImmediateWinEvaluator(game)
        mcts = MCTS(game, evaluator, MCTSArgs(num_simulations=128), np.random.default_rng(10))

        policy = mcts.get_action_prob(root, temp=0)

        self.assertEqual(1.0, policy[blocking_action])

    def test_root_noise_is_seeded_and_does_not_mutate_base_priors(self) -> None:
        root = self.game.init_board()
        args = MCTSArgs(
            num_simulations=48,
            root_noise_epsilon=0.75,
            root_dirichlet_alpha=0.3,
        )
        first = MCTS(
            self.game,
            UniformEvaluator(self.game.get_action_size()),
            args,
            np.random.default_rng(6),
        )
        second = MCTS(
            self.game,
            UniformEvaluator(self.game.get_action_size()),
            args,
            np.random.default_rng(6),
        )

        first_policy = first.get_action_prob(root, add_root_noise=True)
        second_policy = second.get_action_prob(root, add_root_noise=True)

        np.testing.assert_allclose(first_policy, second_policy)
        root_key = self.game.to_string(root)
        np.testing.assert_allclose(
            first.ps[root_key],
            np.full(self.game.get_action_size(), 1.0 / self.game.get_action_size()),
        )

    def test_all_zero_policy_uses_visible_legal_fallback(self) -> None:
        evaluator = UniformEvaluator(self.game.get_action_size())
        evaluator.policy[:] = 0.0
        mcts = MCTS(self.game, evaluator, MCTSArgs(num_simulations=8), np.random.default_rng(7))
        root = self.game.init_board()

        policy = mcts.get_action_prob(root)
        telemetry = mcts.telemetry(root)

        self.assertGreater(telemetry.invalid_policy_fallbacks, 0)
        self.assertAlmostEqual(1.0, float(policy.sum()))

    def test_single_simulation_falls_back_to_expanded_prior(self) -> None:
        evaluator = UniformEvaluator(self.game.get_action_size())
        evaluator.policy[:] = 0.0
        evaluator.policy[3] = 1.0
        mcts = MCTS(self.game, evaluator, MCTSArgs(num_simulations=1), np.random.default_rng(8))
        root = self.game.init_board()

        policy = mcts.get_action_prob(root, temp=0)

        self.assertEqual(1.0, policy[3])
        self.assertEqual(1, mcts.telemetry(root).zero_visit_fallbacks)


if __name__ == "__main__":
    unittest.main()
