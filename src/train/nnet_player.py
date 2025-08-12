from src.train.player import Player
from src.model.model_v1 import PenteNet
from src.game.game import Game
from src.mcts.mcts_v2 import MCTS

import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)

class NNetPlayer(Player):
    def __init__(self, net: 'PenteNet', mcts: Union['MCTS', None], name: str = "NNetPlayer"):
        self.net = net
        self.mcts = mcts
        self.name = name

    def reset(self):
        if self.mcts:
            self.mcts.reset()

    def play(self, game: 'Game', board, player: int, debug: bool = False):
        if not self.mcts:
            canonical_board = game.get_canonical_form(board, player)
            legal_moves = game.get_valid_moves(board, player)
            p, v = self.net.predict(canonical_board)
            if debug:
                logger.info(f"{self.name} prediction (no mcts): {v * player}")
            masked = p.cpu().numpy() * legal_moves
            return np.argmax(masked)
        else:
            canonical_board = game.get_canonical_form(board, player)
            action_probs = self.mcts.get_action_prob(canonical_board, temp=0)
            if debug:
                _, v = self.net.predict(canonical_board)
                logger.info(f"{self.name} prediction (mcts): {v * player}")

            return np.argmax(action_probs)