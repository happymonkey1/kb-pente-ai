import unittest
from typing import Sequence

import numpy as np

from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.batched import (
    BatchedSearchAccumulator,
    SearchSession,
    evaluate_search_wave,
    run_batched_search,
)
from src.mcts.mcts_v2 import MCTS, MCTSArgs


class RecordingEvaluator:
    def __init__(self, action_size: int) -> None:
        self.policy = np.full(action_size, 1.0 / action_size, dtype=np.float64)
        self.batch_sizes: list[int] = []

    def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        self.batch_sizes.append(1)
        return np.array(self.policy, copy=True), 0.0

    def evaluate_batch(self, positions: Sequence[PenteBoard]) -> tuple[np.ndarray, np.ndarray]:
        self.batch_sizes.append(len(positions))
        return (
            np.repeat(self.policy[np.newaxis, :], len(positions), axis=0),
            np.zeros(len(positions), dtype=np.float64),
        )


class BatchedSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        self.args = MCTSArgs(
            num_simulations=24,
            root_noise_epsilon=0.75,
            root_dirichlet_alpha=0.3,
        )

    def test_matches_independent_search_without_noise(self) -> None:
        root = self.game.init_board()
        single_evaluator = RecordingEvaluator(self.game.get_action_size())
        single = MCTS(self.game, single_evaluator, self.args, np.random.default_rng(3))
        expected = single.get_action_prob(root, add_root_noise=False)

        batch_evaluator = RecordingEvaluator(self.game.get_action_size())
        searches = [
            MCTS(self.game, batch_evaluator, self.args, np.random.default_rng(index))
            for index in range(4)
        ]
        result = run_batched_search(
            searches,
            [root] * len(searches),
            [1.0] * len(searches),
            add_root_noise=False,
        )

        for policy in result.policies:
            np.testing.assert_allclose(expected, policy)
        self.assertEqual(4, result.telemetry.root_count)
        self.assertGreater(result.telemetry.duplicate_leaf_rate, 0.0)
        self.assertLess(result.telemetry.unique_evaluations, result.telemetry.evaluation_requests)
        self.assertEqual(tuple(batch_evaluator.batch_sizes), result.telemetry.inference_batch_sizes)
        self.assertEqual(min(batch_evaluator.batch_sizes), result.telemetry.min_inference_batch_size)
        self.assertEqual(
            float(np.median(batch_evaluator.batch_sizes)),
            result.telemetry.median_inference_batch_size,
        )
        self.assertEqual(
            float(np.percentile(batch_evaluator.batch_sizes, 95)),
            result.telemetry.p95_inference_batch_size,
        )

    def test_root_noise_diversifies_batches_and_is_reproducible(self) -> None:
        root = self.game.init_board()

        def run() -> tuple[list[np.ndarray], list[int]]:
            evaluator = RecordingEvaluator(self.game.get_action_size())
            searches = [
                MCTS(self.game, evaluator, self.args, np.random.default_rng(index + 10))
                for index in range(8)
            ]
            result = run_batched_search(
                searches,
                [root] * len(searches),
                [1.0] * len(searches),
                add_root_noise=True,
            )
            self.assertGreater(result.telemetry.max_inference_batch_size, 1)
            self.assertEqual(8, result.telemetry.root_count)
            return result.policies, evaluator.batch_sizes

        first_policies, first_batches = run()
        second_policies, second_batches = run()

        for first, second in zip(first_policies, second_policies):
            np.testing.assert_allclose(first, second)
        self.assertEqual(first_batches, second_batches)

    def test_completed_session_can_be_replaced_while_peer_search_continues(self) -> None:
        evaluator = RecordingEvaluator(self.game.get_action_size())
        first_root = self.game.init_board()
        second_root, _ = self.game.apply_action(first_root, first_root.current_player, 0)
        replacement_root, _ = self.game.apply_action(first_root, first_root.current_player, 1)
        short = SearchSession(
            MCTS(self.game, evaluator, MCTSArgs(num_simulations=1)),
            first_root,
            1.0,
            False,
        )
        long = SearchSession(
            MCTS(self.game, evaluator, MCTSArgs(num_simulations=2)),
            second_root,
            1.0,
            False,
        )
        accumulator = BatchedSearchAccumulator(2)

        evaluate_search_wave([short, long], accumulator)

        self.assertTrue(short.is_complete)
        self.assertFalse(long.is_complete)
        replacement = SearchSession(
            MCTS(self.game, evaluator, MCTSArgs(num_simulations=1)),
            replacement_root,
            1.0,
            False,
        )
        evaluate_search_wave([replacement, long], accumulator)

        self.assertTrue(replacement.is_complete)
        self.assertTrue(long.is_complete)
        self.assertEqual([2, 2], evaluator.batch_sizes)
        self.assertEqual((2, 2), accumulator.telemetry().inference_batch_sizes)


if __name__ == "__main__":
    unittest.main()
