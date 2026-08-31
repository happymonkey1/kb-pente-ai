from __future__ import annotations

from abc import ABC, abstractmethod

from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame


class Player(ABC):
    @abstractmethod
    def play(
        self,
        game: PenteGame,
        board: PenteBoard,
        player: int,
        debug: bool = False,
    ) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        return
