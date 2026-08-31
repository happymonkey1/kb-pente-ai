from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class GameStatus(Enum):
    IN_PROGRESS = "in_progress"
    DRAW = "draw"
    WIN = "win"


@dataclass(frozen=True, slots=True)
class TerminalResult:
    status: GameStatus
    winner: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is GameStatus.WIN and self.winner not in (Game.PLAYER_ONE, Game.PLAYER_TWO):
            raise ValueError("A win must identify Player 1 or Player 2")
        if self.status is not GameStatus.WIN and self.winner is not None:
            raise ValueError("Only a win can identify a winner")
        if self.status is not GameStatus.WIN and self.reason is not None:
            raise ValueError("Only a win can identify a reason")

    @property
    def is_terminal(self) -> bool:
        return self.status is not GameStatus.IN_PROGRESS

    def value_for(self, player: int) -> float:
        Game.validate_player(player)
        if self.status in (GameStatus.IN_PROGRESS, GameStatus.DRAW):
            return 0.0
        return 1.0 if self.winner == player else -1.0

    @classmethod
    def in_progress(cls) -> TerminalResult:
        return cls(GameStatus.IN_PROGRESS)

    @classmethod
    def draw(cls) -> TerminalResult:
        return cls(GameStatus.DRAW)

    @classmethod
    def win(cls, player: int, reason: str | None = None) -> TerminalResult:
        Game.validate_player(player)
        return cls(GameStatus.WIN, player, reason)


class Game(ABC):
    PLAYER_ONE = 1
    PLAYER_TWO = -1

    @staticmethod
    def validate_player(player: int) -> None:
        if player not in (Game.PLAYER_ONE, Game.PLAYER_TWO):
            raise ValueError(f"Invalid player: {player}")

    @staticmethod
    def player_index(player: int) -> int:
        Game.validate_player(player)
        return 0 if player == Game.PLAYER_ONE else 1

    @staticmethod
    def opponent(player: int) -> int:
        Game.validate_player(player)
        return Game.PLAYER_TWO if player == Game.PLAYER_ONE else Game.PLAYER_ONE

    @abstractmethod
    def get_player_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def init_board(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_board_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_action_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, board: Any, player: int, action: int) -> tuple[Any, int]:
        raise NotImplementedError

    @abstractmethod
    def get_next_player(self, player: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_valid_moves(self, board: Any, player: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_valid_move(self, board: Any, player: int, action: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def check_game_end(self, board: Any) -> TerminalResult:
        raise NotImplementedError

    @abstractmethod
    def get_symmetries(self, board: Any, policy: np.ndarray) -> list[tuple[Any, np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def to_string(self, board: Any) -> bytes:
        raise NotImplementedError
