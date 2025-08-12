import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.game.game import Game
from dataclasses import dataclass

from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.model.model_v1 import PenteNet

from numba import njit

EPS = 1e-8

logger = logging.getLogger(__name__)

@dataclass
class MCTSArgs:
    num_simulations: int = 400
    c_puct: float = 1.0

# Reference: https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: 'Game', network: 'PenteNet', args: 'MCTSArgs'):
        self.game = game
        self.net = network
        self.args = args
        self.qsa = {}
        self.nsa = {}
        self.ns = {}
        self.ps = {}

        self.es = {}
        self.vs = {}

    def reset(self):
        self.qsa = {}
        self.nsa = {}
        self.ns = {}
        self.ps = {}

        self.es = {}
        self.vs = {}

    def get_action_prob(self, canonical_board: 'PenteBoard', temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.num_simulations):
            self.search(canonical_board)

        s = self.game.to_string(canonical_board)
        counts = [self.nsa[(s, a)] if (s, a) in self.nsa else 0 for a in range(self.game.get_action_size())]

        if temp == 0:
            return self.__random_exploration(canonical_board, counts)

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if counts_sum == 0:
            logger.error("Counts sum is zero")
            return self.__random_exploration(canonical_board, counts)

        probs = [x / counts_sum for x in counts]
        return probs

    def __random_exploration(self, canonical_board, counts):
        best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
        valid_moves = self.game.get_valid_moves(canonical_board, Game.PLAYER_ONE)
        masked = valid_moves[best_as].astype(bool)
        best_a = np.random.choice(best_as[masked])

        if not self.game.is_valid_move(canonical_board, PenteGame.PLAYER_ONE, best_a):
            raise ValueError(f"Failed to compute random exploration best move ")

        ps = [0] * len(counts)
        ps[best_a] = 1
        return ps

    def search(self, canonical_board: 'PenteBoard'):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.to_string(canonical_board)

        if s not in self.es:
            self.es[s] = self.game.check_game_end(canonical_board, Game.PLAYER_ONE)[1]

        if self.es[s] != 0:
            # terminal node
            return -self.es[s]

        if s not in self.ps:
            # leaf node
            self.ps[s], v = self.net.predict(canonical_board)
            valid_moves = self.game.get_valid_moves(canonical_board, Game.PLAYER_ONE)
            self.ps[s] = self.ps[s].cpu().numpy().reshape(-1) * valid_moves  # masking invalid moves
            sum_ps_s = np.sum(self.ps[s])
            if sum_ps_s > 0:
                self.ps[s] /= sum_ps_s
            else:
                logger.error("All valid moves were masked, doing a workaround.")
                self.ps[s] = self.ps[s] + valid_moves
                self.ps[s] /= np.sum(self.ps[s])

            self.vs[s] = valid_moves
            self.ns[s] = 0
            return -v

        valid_moves = self.vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.get_action_size()):
            if valid_moves[a]:
                if (s, a) in self.qsa:
                    u = self.qsa[(s, a)] + self.args.c_puct * self.ps[s][a] * math.sqrt(self.ns[s]) / (
                            1 + self.nsa[(s, a)])
                else:
                    u = self.args.c_puct * self.ps[s][a] * math.sqrt(self.ns[s] + EPS)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.apply_action(canonical_board, Game.PLAYER_ONE, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.qsa:
            self.qsa[(s, a)] = (self.nsa[(s, a)] * self.qsa[(s, a)] + v) / (self.nsa[(s, a)] + 1)
            self.nsa[(s, a)] += 1

        else:
            self.qsa[(s, a)] = v
            self.nsa[(s, a)] = 1

        self.ns[s] += 1
        return -v