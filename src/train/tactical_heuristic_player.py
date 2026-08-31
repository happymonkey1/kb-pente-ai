from __future__ import annotations

import numpy as np

from src.game.game import Game
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.train.player import Player


class TacticalHeuristicPlayer(Player):
    """Deterministic non-neural baseline for immediate tactics and local shape."""

    def play(
        self,
        game: PenteGame,
        board: PenteBoard,
        player: int,
        debug: bool = False,
    ) -> int:
        if player != board.current_player:
            raise ValueError(f"Expected Player {board.current_player}, received Player {player}")
        legal_actions = [
            int(action)
            for action in np.flatnonzero(game.get_valid_moves(board, player))
        ]
        if not legal_actions:
            raise ValueError("No legal moves for tactical heuristic player")

        transitions = {
            action: game.apply_action(board, player, action)[0]
            for action in legal_actions
        }
        for action in legal_actions:
            result = game.check_game_end(transitions[action])
            if result.winner == player:
                return action

        opponent = Game.opponent(player)
        for action in legal_actions:
            if game.would_form_line(board, opponent, action):
                return action

        capture_gain = {
            action: (
                transitions[action].get_capture_count(player)
                - board.get_capture_count(player)
            )
            for action in legal_actions
        }
        maximum_capture_gain = max(capture_gain.values())
        if maximum_capture_gain > 0:
            return min(
                action
                for action, gain in capture_gain.items()
                if gain == maximum_capture_gain
            )

        center = (game.get_board_size() - 1) / 2.0
        return max(
            legal_actions,
            key=lambda action: (
                _adjacent_stones(board.board, action),
                -_center_distance(action, game.get_board_size(), center),
                -action,
            ),
        )


def _adjacent_stones(stones: np.ndarray, action: int) -> int:
    board_size = stones.shape[0]
    row, column = divmod(action, board_size)
    return sum(
        stones[other_row, other_column] != 0
        for other_row in range(max(0, row - 1), min(board_size, row + 2))
        for other_column in range(max(0, column - 1), min(board_size, column + 2))
    )


def _center_distance(action: int, board_size: int, center: float) -> float:
    row, column = divmod(action, board_size)
    return abs(row - center) + abs(column - center)
