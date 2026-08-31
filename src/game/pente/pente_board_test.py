import unittest

import numpy as np

from src.game.game import Game
from src.game.pente.pente_board import PenteBoard


class PenteBoardTest(unittest.TestCase):
    BOARD_SIZE = 9

    def test_apply_move_is_immutable(self) -> None:
        board = PenteBoard.new_board(self.BOARD_SIZE)

        moved = board.apply_move(Game.PLAYER_ONE, (0, 0))

        self.assertEqual(Game.PLAYER_ONE, moved.board[0, 0])
        self.assertEqual(Game.PLAYER_TWO, moved.current_player)
        self.assertEqual(1, moved.ply)
        self.assertEqual(0, moved.last_action)
        self.assertEqual(0, board.board[0, 0])
        self.assertEqual(Game.PLAYER_ONE, board.current_player)
        self.assertFalse(board.board.flags.writeable)
        self.assertFalse(board.captures.flags.writeable)

    def test_player_one_capture_updates_player_one(self) -> None:
        board = self._play(
            PenteBoard.new_board(self.BOARD_SIZE),
            ((4, 0), (4, 1), (0, 0), (4, 2), (4, 3)),
        )

        self.assertEqual(1, board.get_capture_count(Game.PLAYER_ONE))
        self.assertEqual(0, board.get_capture_count(Game.PLAYER_TWO))
        self.assertEqual(0, board.board[4, 1])
        self.assertEqual(0, board.board[4, 2])

    def test_player_two_capture_updates_player_two(self) -> None:
        board = self._play(
            PenteBoard.new_board(self.BOARD_SIZE),
            ((0, 0), (4, 0), (4, 1), (0, 1), (4, 2), (4, 3)),
        )

        self.assertEqual(0, board.get_capture_count(Game.PLAYER_ONE))
        self.assertEqual(1, board.get_capture_count(Game.PLAYER_TWO))
        self.assertEqual(0, board.board[4, 1])
        self.assertEqual(0, board.board[4, 2])

    def test_captures_in_all_directions(self) -> None:
        center = (4, 4)
        directions = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1))

        for row_step, column_step in directions:
            with self.subTest(direction=(row_step, column_step)):
                stones = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
                stones[0, 0] = Game.PLAYER_ONE
                stones[center[0] + row_step, center[1] + column_step] = Game.PLAYER_TWO
                stones[center[0] + 2 * row_step, center[1] + 2 * column_step] = Game.PLAYER_TWO
                stones[center[0] + 3 * row_step, center[1] + 3 * column_step] = Game.PLAYER_ONE
                board = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=Game.PLAYER_ONE)

                moved = board.apply_move(Game.PLAYER_ONE, center)

                self.assertEqual(1, moved.get_capture_count(Game.PLAYER_ONE))
                self.assertEqual(0, moved.board[center[0] + row_step, center[1] + column_step])
                self.assertEqual(0, moved.board[center[0] + 2 * row_step, center[1] + 2 * column_step])

    def test_one_move_can_cross_capture_threshold_with_multiple_pairs(self) -> None:
        stones = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        stones[4, 1] = Game.PLAYER_ONE
        stones[4, 2:4] = Game.PLAYER_TWO
        stones[1, 4] = Game.PLAYER_ONE
        stones[2:4, 4] = Game.PLAYER_TWO
        board = PenteBoard(
            stones,
            np.array((4, 0), dtype=np.int16),
            current_player=Game.PLAYER_ONE,
        )

        moved = board.apply_move(Game.PLAYER_ONE, (4, 4))

        self.assertEqual(6, moved.get_capture_count(Game.PLAYER_ONE))
        self.assertEqual(0, moved.get_capture_count(Game.PLAYER_TWO))

    def test_state_key_includes_captures_player_and_ply(self) -> None:
        stones = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        base = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=Game.PLAYER_ONE)

        captured_stones = np.zeros_like(stones)
        captured_stones[0, 0] = Game.PLAYER_ONE
        captured_stones[0, 1] = Game.PLAYER_TWO
        with_capture = PenteBoard(
            captured_stones,
            np.array((1, 0), dtype=np.int16),
            current_player=Game.PLAYER_ONE,
        )
        without_capture = PenteBoard(
            captured_stones,
            np.zeros(2, dtype=np.int16),
            current_player=Game.PLAYER_ONE,
            ply=2,
        )

        self.assertNotEqual(base.state_key(), with_capture.state_key())
        self.assertNotEqual(with_capture.state_key(), without_capture.state_key())

    def test_feature_planes_are_relative_to_side_to_move(self) -> None:
        stones = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        stones[0, 0] = Game.PLAYER_ONE
        stones[0, 1] = Game.PLAYER_TWO
        stones[1, 0] = Game.PLAYER_ONE
        board = PenteBoard(
            stones,
            np.array((2, 4), dtype=np.int16),
            current_player=Game.PLAYER_TWO,
        )

        planes = board.feature_planes()

        self.assertEqual((4, self.BOARD_SIZE, self.BOARD_SIZE), planes.shape)
        self.assertEqual(1.0, planes[0, 0, 1])
        self.assertEqual(1.0, planes[1, 0, 0])
        np.testing.assert_allclose(planes[2], 0.8)
        np.testing.assert_allclose(planes[3], 0.4)

        caller_owned = np.empty_like(planes)
        board.write_feature_planes(caller_owned)
        np.testing.assert_array_equal(planes, caller_owned)

    def test_rejects_invalid_state_and_move_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid stone"):
            PenteBoard(np.full((5, 5), 2), np.zeros(2))
        with self.assertRaisesRegex(ValueError, "shape"):
            PenteBoard(np.zeros((5, 5)), np.zeros(3))
        with self.assertRaisesRegex(ValueError, "Invalid player"):
            PenteBoard.new_board().apply_move(0, (0, 0))
        with self.assertRaisesRegex(ValueError, "out of range"):
            PenteBoard.new_board().apply_move(Game.PLAYER_ONE, (-1, 0))

    @staticmethod
    def _play(board: PenteBoard, moves: tuple[tuple[int, int], ...]) -> PenteBoard:
        for move in moves:
            board = board.apply_move(board.current_player, move)
        return board


if __name__ == "__main__":
    unittest.main()
