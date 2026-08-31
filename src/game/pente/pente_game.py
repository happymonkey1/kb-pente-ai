from __future__ import annotations

import numpy as np

from src.game.game import Game, TerminalResult
from src.game.pente.pente_board import CAPTURES_TO_WIN, PenteBoard
from src.game.pente.rules import PenteRuleset, is_legal_action, legal_action_mask


class PenteGame(Game):
    CAPTURES_TO_WIN = CAPTURES_TO_WIN

    def __init__(
        self,
        board_size: int = 19,
        player_count: int = 2,
        ruleset: PenteRuleset = PenteRuleset.STANDARD,
    ) -> None:
        if player_count != 2:
            raise ValueError("PenteGame supports exactly two players")
        if board_size < 5:
            raise ValueError("Pente board must be at least 5 by 5")
        if ruleset is not PenteRuleset.FREESTYLE and board_size % 2 == 0:
            raise ValueError("Standard and tournament Pente require an odd board size with one center")

        self.board_size = board_size
        self.player_count = player_count
        self.ruleset = ruleset

    def init_board(self) -> PenteBoard:
        return PenteBoard.new_board(self.board_size)

    def get_board_size(self) -> int:
        return self.board_size

    def get_player_count(self) -> int:
        return self.player_count

    def get_action_size(self) -> int:
        return self.board_size * self.board_size

    def apply_action(self, board: PenteBoard, player: int, action: int) -> tuple[PenteBoard, int]:
        self._validate_position(board)
        Game.validate_player(player)
        if player != board.current_player:
            raise ValueError(f"Expected Player {board.current_player} to move, received Player {player}")
        assert board.ply is not None
        if not is_legal_action(
            board.board,
            board.current_player,
            board.ply,
            self.ruleset,
            action,
        ):
            raise ValueError(f"Invalid action: {action}")

        row, column = divmod(action, self.board_size)
        next_board = board._apply_validated_move(player, row, column)
        return next_board, next_board.current_player

    def get_next_player(self, player: int) -> int:
        return Game.opponent(player)

    def get_valid_moves(self, board: PenteBoard, player: int) -> np.ndarray:
        self._validate_position(board)
        Game.validate_player(player)
        if player != board.current_player:
            raise ValueError(f"Expected Player {board.current_player}, received Player {player}")
        assert board.ply is not None
        return legal_action_mask(board.board, board.current_player, board.ply, self.ruleset).astype(np.int8)

    def is_valid_move(self, board: PenteBoard, player: int, action: int) -> bool:
        self._validate_position(board)
        Game.validate_player(player)
        if player != board.current_player:
            raise ValueError(f"Expected Player {board.current_player}, received Player {player}")
        assert board.ply is not None
        return is_legal_action(
            board.board,
            board.current_player,
            board.ply,
            self.ruleset,
            action,
        )

    def would_form_line(self, board: PenteBoard, player: int, action: int) -> bool:
        """Return whether an empty-point placement would form five for either player."""
        self._validate_position(board)
        Game.validate_player(player)
        if not 0 <= action < self.get_action_size():
            return False
        row, column = divmod(action, self.board_size)
        if board.board[row, column] != 0:
            return False
        stones = np.array(board.board, copy=True)
        stones[row, column] = player
        return _has_five_from_action(stones, player, action)

    def check_game_end(self, board: PenteBoard) -> TerminalResult:
        self._validate_position(board)
        player_one_captures = int(board.captures[Game.player_index(Game.PLAYER_ONE)])
        player_two_captures = int(board.captures[Game.player_index(Game.PLAYER_TWO)])
        if (
            player_one_captures >= self.CAPTURES_TO_WIN
            and player_two_captures >= self.CAPTURES_TO_WIN
        ):
            raise ValueError("Position has multiple capture winners")
        if player_one_captures >= self.CAPTURES_TO_WIN:
            return TerminalResult.win(Game.PLAYER_ONE, "capture")
        if player_two_captures >= self.CAPTURES_TO_WIN:
            return TerminalResult.win(Game.PLAYER_TWO, "capture")

        if board.last_action is not None:
            previous_player = Game.opponent(board.current_player)
            if _has_five_from_action(board.board, previous_player, board.last_action):
                return TerminalResult.win(previous_player, "line")
        else:
            line_winners = [
                player
                for player in (Game.PLAYER_ONE, Game.PLAYER_TWO)
                if _has_any_five(board.board, player)
            ]
            if len(line_winners) > 1:
                raise ValueError("Position has multiple line winners")
            if line_winners:
                return TerminalResult.win(line_winners[0], "line")

        assert board.ply is not None
        stones_on_board = board.ply - 2 * (
            player_one_captures + player_two_captures
        )
        if stones_on_board == board.board.size:
            return TerminalResult.draw()
        return TerminalResult.in_progress()

    def get_symmetries(
        self,
        board: PenteBoard,
        policy: np.ndarray,
    ) -> list[tuple[PenteBoard, np.ndarray]]:
        if policy.shape != (self.get_action_size(),):
            raise ValueError(f"Policy must have shape ({self.get_action_size()},), found {policy.shape}")
        return [self.get_symmetry(board, policy, index) for index in range(8)]

    def get_symmetry(
        self,
        board: PenteBoard,
        policy: np.ndarray,
        index: int,
    ) -> tuple[PenteBoard, np.ndarray]:
        if not 0 <= index < 8:
            raise ValueError(f"Symmetry index must be in [0, 8), found {index}")

        rotations = index % 4
        reflect = index >= 4
        stones = np.rot90(board.board, rotations)
        policy_board = np.rot90(policy.reshape(self.board_size, self.board_size), rotations)

        action_board = None
        if board.last_action is not None:
            action_board = np.zeros_like(board.board, dtype=np.int8)
            action_board.reshape(-1)[board.last_action] = 1
            action_board = np.rot90(action_board, rotations)

        if reflect:
            stones = np.fliplr(stones)
            policy_board = np.fliplr(policy_board)
            if action_board is not None:
                action_board = np.fliplr(action_board)

        last_action = None if action_board is None else int(np.argmax(action_board))
        transformed = PenteBoard(
            board=stones,
            captures=board.captures,
            current_player=board.current_player,
            ply=board.ply,
            last_action=last_action,
        )
        return transformed, np.array(policy_board.reshape(-1), copy=True)

    def to_string(self, board: PenteBoard) -> bytes:
        self._validate_position(board)
        return board.state_key()

    def _validate_position(self, board: PenteBoard) -> None:
        if board.board.shape != (self.board_size, self.board_size):
            raise ValueError(
                f"Position shape {board.board.shape} does not match game size {self.board_size}"
            )


def _has_five_from_action(stones: np.ndarray, player: int, action: int) -> bool:
    board_size = stones.shape[0]
    row, column = divmod(action, board_size)
    if stones[row, column] != player:
        return False

    for row_step, column_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for direction in (-1, 1):
            next_row = row + direction * row_step
            next_column = column + direction * column_step
            while (
                0 <= next_row < board_size
                and 0 <= next_column < board_size
                and stones[next_row, next_column] == player
            ):
                count += 1
                next_row += direction * row_step
                next_column += direction * column_step
        if count >= 5:
            return True
    return False


def _has_any_five(stones: np.ndarray, player: int) -> bool:
    actions = np.flatnonzero(stones.reshape(-1) == player)
    return any(_has_five_from_action(stones, player, int(action)) for action in actions)
