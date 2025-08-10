import numpy as np

from src.game.game import Game
from src.train.player import Player
import logging

logger = logging.getLogger(__name__)

class RandomPlayer(Player):

    def __init__(self):
        pass

    def play(self, game: 'Game', board, player: int):
        valid_moves = game.get_valid_moves(board, player)
        if valid_moves.sum() == 0:
            logger.error("No valid moves for random player")
            logger.info(f"Board dump:\n {board}")
            raise ValueError("No valid moves for random player")

        return np.random.choice(np.argwhere(valid_moves == 1).flatten())
