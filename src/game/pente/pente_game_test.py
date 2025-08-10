import unittest

from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
import numpy as np


class PenteGameTest(unittest.TestCase):
    BOARD_SIZE = 6
    def test_is_terminal_five_in_a_row(self):
        game = PenteGame(board_size=PenteGameTest.BOARD_SIZE)
        board = PenteBoard.new_board(PenteGameTest.BOARD_SIZE)
        board.board[0,0] = 1
        board.board[0,1] = 1
        board.board[0,2] = 1
        board.board[0,3] = 1
        board.board[0,4] = 1

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_ONE)
        self.assertTrue(is_terminal)
        self.assertEqual(1, score)

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_TWO)
        self.assertTrue(is_terminal)
        self.assertEqual(1, score)

    def test_is_terminal_five_in_a_row_horizontal(self):
        game = PenteGame(board_size=PenteGameTest.BOARD_SIZE)
        board = PenteBoard.new_board(PenteGameTest.BOARD_SIZE)
        board.board[0,0] = 1
        board.board[1,0] = 1
        board.board[2,0] = 1
        board.board[3,0] = 1
        board.board[4,0] = 1

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_ONE)
        self.assertTrue(is_terminal)
        self.assertEqual(1, score)

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_TWO)
        self.assertTrue(is_terminal)
        self.assertEqual(1, score)

    def test_is_terminal_five_in_a_row_diag(self):
        game = PenteGame(board_size=PenteGameTest.BOARD_SIZE)
        board = PenteBoard.new_board(PenteGameTest.BOARD_SIZE)
        board.board[0,0] = 1
        board.board[1,1] = 1
        board.board[2,2] = 1
        board.board[3,3] = 1
        board.board[4,4] = 1

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_ONE)
        self.assertTrue(is_terminal)
        self.assertEqual(1, score)

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_TWO)
        self.assertTrue(is_terminal)
        self.assertEqual(1, score)

    def test_is_terminal_captures_player_one_in_two_player_game(self):
        game = PenteGame(board_size=PenteGameTest.BOARD_SIZE)
        board = PenteBoard.new_board(PenteGameTest.BOARD_SIZE)
        board.captures[0] = 5

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_ONE)
        self.assertTrue(is_terminal)
        self.assertEqual(score, 1)

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_TWO)
        self.assertTrue(is_terminal)
        self.assertEqual(score, 1)

    def test_is_terminal_captures_player_two_in_two_player_game(self):
        game = PenteGame(board_size=PenteGameTest.BOARD_SIZE)
        board = PenteBoard.new_board(PenteGameTest.BOARD_SIZE)
        board.captures[1] = 5

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_ONE)
        self.assertTrue(is_terminal)
        self.assertEqual(score, -1)

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_TWO)
        self.assertTrue(is_terminal)
        self.assertEqual(score, -1)

    def test_is_terminal_board_full(self):
        game = PenteGame(board_size=PenteGameTest.BOARD_SIZE)
        board = PenteBoard.new_board(PenteGameTest.BOARD_SIZE)
        board.board = np.ones((PenteGameTest.BOARD_SIZE, PenteGameTest.BOARD_SIZE), dtype=np.uint8)

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_ONE)
        self.assertTrue(is_terminal)
        self.assertEqual(score, 0)

        is_terminal, score = game.check_game_end(board, PenteGame.PLAYER_TWO)
        self.assertTrue(is_terminal)
        self.assertEqual(score, 0)

    def test_get_valid_moves_succeeds(self):
        board = PenteBoard.new_board(PenteGameTest.BOARD_SIZE)
        board.board = np.ones((PenteGameTest.BOARD_SIZE, PenteGameTest.BOARD_SIZE))
        board.board[0,0] = 0

        game = PenteGame(PenteGameTest.BOARD_SIZE)
        legal_moves = game.get_valid_moves(board, PenteGame.PLAYER_TWO)
        self.assertEqual(1, np.sum(legal_moves == 1))
        self.assertEqual(1, legal_moves[0])


if __name__ == '__main__':
    unittest.main()
