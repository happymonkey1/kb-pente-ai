import unittest

from src.game.pente.pente_board import PenteBoard
import numpy as np

from src.game.pente.pente_game import PenteGame


class PenteBoardTest(unittest.TestCase):
    def test_apply_move_succeeds(self):
        board = PenteBoard.new_board()
        self.assertEqual(0, board.board[0,0])

        new_board = board.apply_move(PenteGame.PLAYER_ONE, PenteGame.PLAYER_TWO, (0,0))
        self.assertEqual(1, new_board.board[0,0])

        # Check no in place mutation
        self.assertEqual(0, board.board[0, 0])

    def test_apply_move_capture_succeeds(self):
        board = PenteBoard.new_board()
        board.board[0,0] = PenteGame.PLAYER_ONE
        board.board[0,1] = PenteGame.PLAYER_TWO
        board.board[0,2] = PenteGame.PLAYER_TWO

        new_board = board.apply_move(PenteGame.PLAYER_ONE, PenteGame.PLAYER_TWO, (0,3))
        self.assertEqual(1, new_board.captures[0])
        self.assertEqual(0, new_board.board[0,1])
        self.assertEqual(0, new_board.board[0,2])
        self.assertEqual(1, new_board.board[0,3])

        # Check no in place mutation
        self.assertEqual(0, board.captures[0])
        self.assertEqual(0, board.board[0,3])

    def test_legal_moves(self):
        board = PenteBoard.new_board()
        board.board = np.ones((19, 19))
        board.board[0,0] = 0

        legal_moves = board.get_legal_moves()
        self.assertEqual(PenteGame.PLAYER_ONE, len(legal_moves))
        self.assertEqual((0,0), legal_moves[0])

if __name__ == '__main__':
    unittest.main()
