from __future__ import annotations

import numpy as np

from src.game.game import Game
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTS
from src.model.model_v1 import PenteNet
from src.train.player import Player


class NNetPlayer(Player):
    def __init__(self, net: PenteNet, mcts: MCTS | None, name: str = "NNetPlayer") -> None:
        self.net = net
        self.mcts = mcts
        self.name = name

    def reset(self) -> None:
        if self.mcts is not None:
            self.mcts.reset()

    def play(
        self,
        game: PenteGame,
        board: PenteBoard,
        player: int,
        debug: bool = False,
    ) -> int:
        if player != board.current_player:
            raise ValueError(f"Expected Player {board.current_player}, received Player {player}")
        self.net.eval()
        if self.mcts is not None:
            policy = self.mcts.get_action_prob(board, temp=0, add_root_noise=False)
            return int(np.argmax(policy))

        policy, _ = self.net.evaluate(board)
        legal = game.get_valid_moves(board, player).astype(bool)
        masked = np.where(legal, policy, -np.inf)
        return int(np.argmax(masked))
