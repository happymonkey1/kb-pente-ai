from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.game.game import Game
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset


class TacticalEvaluator(Protocol):
    def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        """Return policy and side-to-move value for one position."""


@dataclass(frozen=True, slots=True)
class TacticalCase:
    name: str
    category: str
    position: PenteBoard
    expected_actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TacticalSuiteStats:
    cases: int
    correct: int
    accuracy: float
    mean_expected_policy_mass: float
    category_accuracy: dict[str, float]


def build_tactical_cases(board_size: int = 9) -> tuple[TacticalCase, ...]:
    if board_size < 9:
        raise ValueError("The fixed tactical suite requires a board of at least 9 by 9")
    row = board_size // 2
    cases = [
        _line_case(board_size, row, Game.PLAYER_ONE, "complete_line_p1"),
        _line_case(board_size, row, Game.PLAYER_TWO, "complete_line_p2"),
        _block_case(board_size, row, Game.PLAYER_ONE, "block_line_p1"),
        _block_case(board_size, row, Game.PLAYER_TWO, "block_line_p2"),
        _capture_case(board_size, row, Game.PLAYER_ONE, "capture_win_p1"),
        _capture_case(board_size, row, Game.PLAYER_TWO, "capture_win_p2"),
    ]
    return tuple(cases)


def evaluate_tactical_suite(
    evaluator: TacticalEvaluator,
    game: PenteGame,
    cases: tuple[TacticalCase, ...] | None = None,
) -> TacticalSuiteStats:
    selected_cases = build_tactical_cases(game.get_board_size()) if cases is None else cases
    correct = 0
    masses: list[float] = []
    category_results: dict[str, list[bool]] = {}

    for case in selected_cases:
        policy, _ = evaluator.evaluate(case.position)
        policy = np.asarray(policy, dtype=np.float64).reshape(-1)
        legal = game.get_valid_moves(case.position, case.position.current_player).astype(bool)
        masked = np.where(legal, policy, -np.inf)
        selected_action = int(np.argmax(masked))
        is_correct = selected_action in case.expected_actions
        correct += is_correct
        masses.append(float(policy[list(case.expected_actions)].sum()))
        category_results.setdefault(case.category, []).append(is_correct)

    category_accuracy = {
        category: sum(results) / len(results)
        for category, results in category_results.items()
    }
    return TacticalSuiteStats(
        cases=len(selected_cases),
        correct=correct,
        accuracy=correct / len(selected_cases),
        mean_expected_policy_mass=float(np.mean(masses)),
        category_accuracy=category_accuracy,
    )


def _line_case(board_size: int, row: int, player: int, name: str) -> TacticalCase:
    stones = np.zeros((board_size, board_size), dtype=np.int8)
    stones[row, 2:6] = player
    filler_count = 4 if player == Game.PLAYER_ONE else 3
    fillers = ((0, 0), (0, 2), (1, 5), (2, 8))
    for coordinate in fillers[:filler_count]:
        stones[coordinate] = Game.opponent(player)
    position = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=player)
    return TacticalCase(
        name=name,
        category="line_win",
        position=position,
        expected_actions=(row * board_size + 1, row * board_size + 6),
    )


def _block_case(board_size: int, row: int, player: int, name: str) -> TacticalCase:
    stones = np.zeros((board_size, board_size), dtype=np.int8)
    stones[row, 2:6] = Game.opponent(player)
    filler_count = 4 if player == Game.PLAYER_ONE else 3
    fillers = ((0, 0), (0, 2), (1, 5), (2, 8))
    for coordinate in fillers[:filler_count]:
        stones[coordinate] = player
    position = PenteBoard(stones, np.zeros(2, dtype=np.int16), current_player=player)
    return TacticalCase(
        name=name,
        category="line_block",
        position=position,
        expected_actions=(row * board_size + 1, row * board_size + 6),
    )


def _capture_case(board_size: int, row: int, player: int, name: str) -> TacticalCase:
    stones = np.zeros((board_size, board_size), dtype=np.int8)
    if player == Game.PLAYER_ONE:
        stones[row, 1] = player
        stones[row, 2:4] = Game.PLAYER_TWO
        stones[0, 0] = Game.PLAYER_ONE
        action = row * board_size + 4
        captures = np.array((4, 0), dtype=np.int16)
    else:
        stones[row, 7] = player
        stones[row, 5:7] = Game.PLAYER_ONE
        action = row * board_size + 4
        captures = np.array((0, 4), dtype=np.int16)
    position = PenteBoard(stones, captures, current_player=player)
    return TacticalCase(
        name=name,
        category="capture_win",
        position=position,
        expected_actions=(action,),
    )
