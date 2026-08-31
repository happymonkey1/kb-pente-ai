import unittest

import numpy as np

from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.train.random_player import RandomPlayer


class RandomPlayerTest(unittest.TestCase):
    def test_seeded_players_produce_the_same_legal_sequence(self) -> None:
        game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        first = RandomPlayer(np.random.default_rng(21))
        second = RandomPlayer(np.random.default_rng(21))
        first_position = game.init_board()
        second_position = game.init_board()
        first_actions = []
        second_actions = []

        for _ in range(8):
            first_action = first.play(
                game,
                first_position,
                first_position.current_player,
            )
            second_action = second.play(
                game,
                second_position,
                second_position.current_player,
            )
            first_actions.append(first_action)
            second_actions.append(second_action)
            first_position, _ = game.apply_action(
                first_position,
                first_position.current_player,
                first_action,
            )
            second_position, _ = game.apply_action(
                second_position,
                second_position.current_player,
                second_action,
            )

        self.assertEqual(first_actions, second_actions)
        self.assertEqual(first_position.state_key(), second_position.state_key())


if __name__ == "__main__":
    unittest.main()
