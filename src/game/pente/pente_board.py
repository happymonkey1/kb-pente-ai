from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np

from src.artifacts import POSITION_SCHEMA_VERSION
from src.game.game import Game


CAPTURES_TO_WIN = 5


@dataclass(frozen=True, slots=True, eq=False)
class PenteBoard:
    board: np.ndarray
    captures: np.ndarray
    current_player: int = Game.PLAYER_ONE
    ply: int | None = None
    last_action: int | None = None

    def __post_init__(self) -> None:
        stones = np.asarray(self.board, dtype=np.int8)
        captures = np.asarray(self.captures, dtype=np.int16)

        if stones.ndim != 2 or stones.shape[0] != stones.shape[1]:
            raise ValueError(f"Pente board must be square, found shape {stones.shape}")
        if stones.shape[0] < 5:
            raise ValueError("Pente board must be at least 5 by 5")
        if not np.isin(stones, (Game.PLAYER_TWO, 0, Game.PLAYER_ONE)).all():
            raise ValueError("Pente board contains an invalid stone value")
        if captures.shape != (2,):
            raise ValueError(f"Pente captures must have shape (2,), found {captures.shape}")
        if np.any(captures < 0):
            raise ValueError("Pente capture counts cannot be negative")

        Game.validate_player(self.current_player)
        inferred_ply = int(np.count_nonzero(stones) + 2 * np.sum(captures))
        ply = inferred_ply if self.ply is None else self.ply
        if ply < 0:
            raise ValueError("Ply cannot be negative")
        if ply != inferred_ply:
            raise ValueError(f"Ply {ply} is inconsistent with board and captures; expected {inferred_ply}")

        expected_player = Game.PLAYER_ONE if ply % 2 == 0 else Game.PLAYER_TWO
        if self.current_player != expected_player:
            raise ValueError(
                f"Current player {self.current_player} is inconsistent with ply {ply}; expected {expected_player}"
            )

        if self.last_action is not None:
            if not 0 <= self.last_action < stones.size:
                raise ValueError(f"Last action is out of range: {self.last_action}")
            previous_player = Game.opponent(self.current_player)
            if stones.reshape(-1)[self.last_action] != previous_player:
                raise ValueError("Last action does not contain the stone placed by the previous player")

        stones = np.array(stones, dtype=np.int8, copy=True, order="C")
        captures = np.array(captures, dtype=np.int16, copy=True, order="C")
        stones.flags.writeable = False
        captures.flags.writeable = False

        object.__setattr__(self, "board", stones)
        object.__setattr__(self, "captures", captures)
        object.__setattr__(self, "ply", ply)

    @classmethod
    def new_board(cls, board_size: int = 19) -> PenteBoard:
        return cls(
            board=np.zeros((board_size, board_size), dtype=np.int8),
            captures=np.zeros(2, dtype=np.int16),
            current_player=Game.PLAYER_ONE,
            ply=0,
        )

    def apply_move(self, player: int, move: tuple[int, int]) -> PenteBoard:
        Game.validate_player(player)
        if player != self.current_player:
            raise ValueError(f"Expected Player {self.current_player} to move, received Player {player}")

        row, column = move
        board_size = self.board.shape[0]
        if not (0 <= row < board_size and 0 <= column < board_size):
            raise ValueError(f"Move is out of range: {move}")
        if self.board[row, column] != 0:
            raise ValueError(f"Move is occupied: {move}")

        stones = np.array(self.board, copy=True)
        stones[row, column] = player
        captured_pairs = _apply_captures_inplace(stones, row, column, player)

        captures = np.array(self.captures, copy=True)
        captures[Game.player_index(player)] += captured_pairs

        action = row * board_size + column
        assert self.ply is not None
        return PenteBoard(
            board=stones,
            captures=captures,
            current_player=Game.opponent(player),
            ply=self.ply + 1,
            last_action=action,
        )

    def get_capture_count(self, player: int) -> int:
        return int(self.captures[Game.player_index(player)])

    def state_key(self) -> bytes:
        board_size = self.board.shape[0]
        header = struct.pack(
            "<BHbI",
            POSITION_SCHEMA_VERSION,
            board_size,
            self.current_player,
            self.ply,
        )
        return header + self.captures.astype("<i2", copy=False).tobytes() + self.board.tobytes()

    def feature_planes(self) -> np.ndarray:
        current = self.current_player
        opponent = Game.opponent(current)
        current_captures = self.get_capture_count(current) / CAPTURES_TO_WIN
        opponent_captures = self.get_capture_count(opponent) / CAPTURES_TO_WIN
        shape = self.board.shape

        return np.stack(
            (
                self.board == current,
                self.board == opponent,
                np.full(shape, current_captures, dtype=np.float32),
                np.full(shape, opponent_captures, dtype=np.float32),
            ),
            axis=0,
        ).astype(np.float32, copy=False)


def _apply_captures_inplace(
    stones: np.ndarray,
    row: int,
    column: int,
    player: int,
) -> int:
    opponent = Game.opponent(player)
    board_size = stones.shape[0]
    captured_pairs = 0

    for row_step, column_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for direction in (-1, 1):
            row_one = row + direction * row_step
            column_one = column + direction * column_step
            row_two = row + direction * 2 * row_step
            column_two = column + direction * 2 * column_step
            row_three = row + direction * 3 * row_step
            column_three = column + direction * 3 * column_step

            if not (0 <= row_three < board_size and 0 <= column_three < board_size):
                continue
            if (
                stones[row_one, column_one] == opponent
                and stones[row_two, column_two] == opponent
                and stones[row_three, column_three] == player
            ):
                stones[row_one, column_one] = 0
                stones[row_two, column_two] = 0
                captured_pairs += 1

    return captured_pairs
