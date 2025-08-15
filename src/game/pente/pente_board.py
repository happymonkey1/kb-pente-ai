from src.game.game import Game

from dataclasses import dataclass
from numba import njit
import numpy as np
import base64
import logging

logger = logging.getLogger(__name__)

@dataclass
class PenteBoard:
    board: np.ndarray
    captures: np.ndarray

    @staticmethod
    def new_board(board_size: int = 19, player_count: int = 2) -> 'PenteBoard':
        return PenteBoard(
            board=np.zeros((board_size, board_size), dtype=np.int8),
            captures=np.zeros(player_count, dtype=np.int8)
        )

    def apply_move(self, player: int, opponent: int, move: tuple[int, int]) -> 'PenteBoard':
        r, c = move
        new_board = np.copy(self.board)
        new_board[r,c] = player

        capture_count = _apply_captures_inplace(new_board, r, c, player, opponent)
        new_captures = np.copy(self.captures)

        new_captures_before_add = new_captures[player - 1]
        if new_captures_before_add + capture_count <= 5:
            new_captures[player - 1] += capture_count
        else:
            logger.warning(f"Preventing capture overflow with current={new_captures_before_add} and add={capture_count}.")
            new_captures[player - 1] = 5

        return PenteBoard(new_board, new_captures)

    def get_capture_count(self, player: int) -> int:
        assert self.captures.size == 2, "get_capture_count only supports two players"
        index = 0 if player == Game.PLAYER_ONE else 1
        return self.captures[index]

    def get_legal_moves(self) -> list[tuple[int, int]]:
        return _get_legal_moves(self.board)

    def get_canonical_form(self, player: int, player_count) -> 'PenteBoard':
        assert Game.PLAYER_ONE == 1, f"Optimization assumption failed, expected Player 1 == 1, found {Game.PLAYER_ONE}"
        board: np.ndarray
        if player_count == 2:
            board = _get_canonical_form_two_players_v2(self.board, player)
        else:
            board = _get_canonical_form_n_players(self.board, player, player_count)

        return PenteBoard(board, self.captures)

    def to_string(self) -> str:
        return base64.b64encode(self.board.tobytes()).decode('ascii')

@njit(cache=True)
def _get_legal_moves(board: np.ndarray) -> list[tuple[int, int]]:
    return list(zip(*np.where(board == 0)))

@njit(cache=True)
def _in_bounds(r: int, c: int, w: int, h: int) -> bool:
    return 0 <= r < w and 0 <= c < h

@njit(cache=True)
def _apply_captures_inplace(board: np.ndarray, r: int, c: int, player: int, opp: int) -> int:
    h, w = board.shape
    captured_count = 0
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))

    for dr, dc in dirs:
        for sign in (-1, 1):
            r1, c1 = r + sign * dr, c + sign * dc
            r2, c2 = r + sign * 2 * dr, c + sign * 2 * dc
            r3, c3 = r + sign * 3 * dr, c + sign * 3 * dc
            if (_in_bounds(r3, c3, h, w) and
                    board[r1, c1] == opp and
                    board[r2, c2] == opp and
                    board[r3, c3] == player):
                board[r1, c1], board[r2, c2] = np.int8(0), np.int8(0)
                captured_count += 1

    return captured_count

@njit
def _get_canonical_form_two_players_v2(board: np.ndarray, player: int) -> np.ndarray:
    return np.int8(player)*board

# @njit(cache=True)
def _get_canonical_form_two_players(board: np.ndarray, player: int) -> np.ndarray:
    if player == 1:
        # TODO: copy here?
        return board
    else:
        canonical_board = np.copy(board)
        non_zero_indices = canonical_board != 0
        canonical_board[non_zero_indices] = 3 - canonical_board[non_zero_indices]
        return canonical_board

@njit(cache=True)
def _get_canonical_form_n_players(board: np.ndarray, player: int, player_count: int) -> np.ndarray:
    if player == 1:
        # TODO: copy here?
        return board
    else:
        remapping = np.arange(player_count + 1)

        remapping[1] = player
        remapping[player] = 1

        return remapping[board]