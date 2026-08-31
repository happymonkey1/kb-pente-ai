from __future__ import annotations

from enum import Enum

import numpy as np

from src.game.game import Game


class PenteRuleset(Enum):
    STANDARD = "standard"
    TOURNAMENT = "tournament"
    FREESTYLE = "freestyle"

    @classmethod
    def parse(cls, value: str) -> PenteRuleset:
        try:
            return cls(value.lower())
        except ValueError as error:
            choices = ", ".join(ruleset.value for ruleset in cls)
            raise ValueError(f"Unknown Pente ruleset {value!r}; expected one of: {choices}") from error


def legal_action_mask(
    stones: np.ndarray,
    current_player: int,
    ply: int,
    ruleset: PenteRuleset,
) -> np.ndarray:
    Game.validate_player(current_player)
    board_size = stones.shape[0]
    legal = stones.reshape(-1) == 0

    if ruleset is PenteRuleset.FREESTYLE:
        return legal

    center = board_size // 2
    if ply == 0:
        legal[:] = False
        legal[center * board_size + center] = True
        return legal

    if ruleset is PenteRuleset.TOURNAMENT and ply == 2 and current_player == Game.PLAYER_ONE:
        rows, columns = np.indices(stones.shape)
        outside_center = np.maximum(np.abs(rows - center), np.abs(columns - center)) >= 3
        legal &= outside_center.reshape(-1)

    return legal


def is_legal_action(
    stones: np.ndarray,
    current_player: int,
    ply: int,
    ruleset: PenteRuleset,
    action: int,
) -> bool:
    Game.validate_player(current_player)
    board_size = stones.shape[0]
    if not 0 <= action < stones.size:
        return False
    if stones.reshape(-1)[action] != 0:
        return False
    if ruleset is PenteRuleset.FREESTYLE:
        return True

    center = board_size // 2
    if ply == 0:
        return action == center * board_size + center
    if ruleset is PenteRuleset.TOURNAMENT and ply == 2 and current_player == Game.PLAYER_ONE:
        row, column = divmod(action, board_size)
        return max(abs(row - center), abs(column - center)) >= 3
    return True
