from src.train.player import Player
from src.model.model_v1 import PenteNet
from src.game.game import Game
from src.mcts.mcts_v2 import MCTS

import numpy as np
from typing import Union

class NNetPlayer(Player):
    def __init__(self, net: 'PenteNet', mcts: Union['MCTS', None]):
        self.net = net
        self.mcts = mcts

    def play(self, game: 'Game', board, player: int):
        if not self.mcts:
            legal_moves = game.get_valid_moves(board, player)
            p, v = self.net.predict(board)
            masked = p.cpu().numpy() * legal_moves
            return np.argmax(masked)
        else:
            canonical_board = game.get_canonical_form(board, player)
            action_probs = self.mcts.get_action_prob(canonical_board, temp=0)
            return np.argmax(action_probs)