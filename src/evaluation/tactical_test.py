import unittest

import numpy as np

from src.evaluation.tactical import build_tactical_cases, evaluate_tactical_suite
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset


class ExpectedActionEvaluator:
    def __init__(self, expected_by_key: dict[bytes, int], action_size: int) -> None:
        self.expected_by_key = expected_by_key
        self.action_size = action_size

    def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        policy = np.zeros(self.action_size, dtype=np.float64)
        policy[self.expected_by_key[position.state_key()]] = 1.0
        return policy, 0.0


class TacticalSuiteTest(unittest.TestCase):
    def test_fixed_cases_are_legal_and_scored_by_category(self) -> None:
        game = PenteGame(9, ruleset=PenteRuleset.FREESTYLE)
        cases = build_tactical_cases(9)
        expected = {case.position.state_key(): case.expected_actions[0] for case in cases}
        evaluator = ExpectedActionEvaluator(expected, game.get_action_size())

        stats = evaluate_tactical_suite(evaluator, game, cases)

        self.assertEqual(6, stats.cases)
        self.assertEqual(6, stats.correct)
        self.assertEqual(1.0, stats.accuracy)
        self.assertEqual(1.0, stats.mean_expected_policy_mass)
        self.assertEqual(
            {"capture_win": 1.0, "line_block": 1.0, "line_win": 1.0},
            stats.category_accuracy,
        )
        for case in cases:
            legal = game.get_valid_moves(case.position, case.position.current_player)
            self.assertTrue(all(legal[action] for action in case.expected_actions))


if __name__ == "__main__":
    unittest.main()
