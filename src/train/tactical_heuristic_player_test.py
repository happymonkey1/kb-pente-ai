import unittest

import numpy as np

from src.game.game import Game
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.train.tactical_heuristic_player import TacticalHeuristicPlayer


class TacticalHeuristicPlayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        self.player = TacticalHeuristicPlayer()

    def test_takes_immediate_line_win(self) -> None:
        stones = np.zeros((5, 5), dtype=np.int8)
        stones[0, :4] = Game.PLAYER_ONE
        stones[2, (0, 2)] = Game.PLAYER_TWO
        stones[3, (1, 3)] = Game.PLAYER_TWO
        board = PenteBoard(stones, np.zeros(2), current_player=Game.PLAYER_ONE)

        action = self.player.play(self.game, board, Game.PLAYER_ONE)

        self.assertEqual(4, action)

    def test_blocks_opponent_line_win(self) -> None:
        stones = np.zeros((5, 5), dtype=np.int8)
        stones[0, :4] = Game.PLAYER_TWO
        stones[(2, 2, 3, 4), (0, 2, 4, 1)] = Game.PLAYER_ONE
        board = PenteBoard(stones, np.zeros(2), current_player=Game.PLAYER_ONE)

        action = self.player.play(self.game, board, Game.PLAYER_ONE)

        self.assertEqual(4, action)

    def test_prefers_capture_over_shape_fallback(self) -> None:
        stones = np.zeros((5, 5), dtype=np.int8)
        stones[0, 1:3] = Game.PLAYER_TWO
        stones[0, 3] = Game.PLAYER_ONE
        stones[4, 4] = Game.PLAYER_ONE
        board = PenteBoard(stones, np.zeros(2), current_player=Game.PLAYER_ONE)

        action = self.player.play(self.game, board, Game.PLAYER_ONE)

        self.assertEqual(0, action)

    def test_empty_freestyle_board_prefers_center(self) -> None:
        board = self.game.init_board()

        action = self.player.play(self.game, board, Game.PLAYER_ONE)

        self.assertEqual(12, action)


if __name__ == "__main__":
    unittest.main()
