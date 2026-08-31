import numpy as np

from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.train.player import Player
import logging

logger = logging.getLogger(__name__)

class RandomPlayer(Player):
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def play(
        self,
        game: PenteGame,
        board: PenteBoard,
        player: int,
        debug: bool = False,
    ) -> int:
        valid_moves = game.get_valid_moves(board, player)
        if valid_moves.sum() == 0:
            logger.error("No valid moves for random player")
            logger.info(f"Board dump:\n {board}")
            raise ValueError("No valid moves for random player")

        return int(self.rng.choice(np.flatnonzero(valid_moves)))
