"""Deterministic reachable-position corpus for native Pente tests."""

from __future__ import annotations

from collections.abc import Iterable
import random

import numpy as np

from src.game.game import GameStatus
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset

from script.native_game_differential_protocol import (
    CorpusEntry,
    _python_terminal,
)


_TARGET_POSITIONS = 400
# Every five-board tournament position is exhausted after the opening and
# P2's 24 legal placements because no point is outside the radius-2 exclusion.
_TOURNAMENT_FIVE_TARGET = 26
# Keep these seeds fixed so a mismatch can be reproduced from its diagnostics.
_BASE_SEED = 20260830
_DRAW_SEED = _BASE_SEED + 35


def _legal_actions(game: PenteGame, board: PenteBoard) -> np.ndarray:
    return np.flatnonzero(game.get_valid_moves(board, board.current_player))


def _apply(game: PenteGame, board: PenteBoard, action: int) -> PenteBoard:
    next_board, _ = game.apply_action(board, board.current_player, action)
    return next_board


def _safe_filler(
    game: PenteGame,
    board: PenteBoard,
    excluded: set[int],
) -> tuple[PenteBoard, int] | None:
    for raw_action in _legal_actions(game, board):
        action = int(raw_action)
        if action in excluded:
            continue
        candidate = _apply(game, board, action)
        if _python_terminal(game, candidate).status is GameStatus.IN_PROGRESS:
            return candidate, action
    return None


def _guided_line(game: PenteGame, player: int) -> list[PenteBoard]:
    size = game.board_size
    center = size // 2
    board = game.init_board()
    states = [board]
    if size < 9:
        return states

    if player == 1:
        row = center
        targets = [row * size + center]
        targets.extend(row * size + column for column in range(center - 4, center))
    else:
        row = center - 1
        start = center - 4
        targets = [row * size + column for column in range(start, center + 1)]

    target_index = 0
    excluded = set(targets)
    while target_index < len(targets):
        if board.current_player == player:
            action = targets[target_index]
            if not game.is_valid_move(board, board.current_player, action):
                return states
            board = _apply(game, board, action)
            states.append(board)
            target_index += 1
            if _python_terminal(game, board).is_terminal:
                return states
        else:
            filler = _safe_filler(game, board, excluded)
            if filler is None:
                return states
            board, _ = filler
            states.append(board)
            if _python_terminal(game, board).is_terminal:
                return states
    return states


def _capture_segments(size: int) -> list[tuple[int, int, int, int, int]]:
    center = size // 2
    if size == 5:
        return [(0, 0, 1, 2, 3)]
    return [
        (center, center - 3, center - 2, center - 1, center),
        (0, size - 4, size - 3, size - 2, size - 1),
        (size - 1, 0, 1, 2, 3),
        (2, size - 4, size - 3, size - 2, size - 1),
        (size - 3, 0, 1, 2, 3),
    ]


def _guided_capture(game: PenteGame, player: int) -> list[PenteBoard]:
    size = game.board_size
    board = game.init_board()
    states = [board]
    segments = _capture_segments(size)
    if player != 1:
        segments = segments[1:] + segments[:1]

    for row, final_column, pair_one_column, pair_two_column, anchor_column in segments:
        anchor_action = row * size + anchor_column
        pair_one_action = row * size + pair_one_column
        pair_two_action = row * size + pair_two_column
        final_action = row * size + final_column
        target_actions = {
            anchor_action,
            pair_one_action,
            pair_two_action,
            final_action,
        }
        if board.current_player != player:
            filler = _safe_filler(game, board, target_actions)
            if filler is None:
                return states
            board, _ = filler
            states.append(board)

        if not game.is_valid_move(board, board.current_player, anchor_action):
            return states
        board = _apply(game, board, anchor_action)
        states.append(board)

        if board.current_player == player:
            return states
        if not game.is_valid_move(board, board.current_player, pair_one_action):
            return states
        board = _apply(game, board, pair_one_action)
        states.append(board)

        filler = _safe_filler(
            game,
            board,
            {anchor_action, pair_one_action, pair_two_action, final_action},
        )
        if filler is None:
            return states
        board, _ = filler
        states.append(board)

        if not game.is_valid_move(board, board.current_player, pair_two_action):
            return states
        board = _apply(game, board, pair_two_action)
        states.append(board)

        if not game.is_valid_move(board, board.current_player, final_action):
            return states
        board = _apply(game, board, final_action)
        states.append(board)
        if _python_terminal(game, board).is_terminal:
            return states

    return states


def _guided_draw(game: PenteGame) -> list[PenteBoard]:
    if game.board_size != 5 or game.ruleset is not PenteRuleset.FREESTYLE:
        return [game.init_board()]
    trajectory = _random_trajectory(game, _DRAW_SEED, game.get_action_size())
    if _python_terminal(game, trajectory[-1]).status is not GameStatus.DRAW:
        raise RuntimeError("deterministic 5 by 5 freestyle draw fixture is invalid")
    return trajectory


def _random_trajectory(
    game: PenteGame,
    seed: int,
    max_plies: int,
) -> list[PenteBoard]:
    rng = random.Random(seed)
    board = game.init_board()
    states = [board]
    for _ in range(max_plies):
        legal = _legal_actions(game, board)
        if legal.size == 0:
            break
        action = int(rng.choice([int(value) for value in legal]))
        board = _apply(game, board, action)
        states.append(board)
        if _python_terminal(game, board).is_terminal:
            break
    return states


def _entry_key(board: PenteBoard) -> bytes:
    return board.state_key()


def _build_config_corpus(
    board_size: int,
    ruleset: PenteRuleset,
    target: int,
    config_index: int,
) -> list[CorpusEntry]:
    game = PenteGame(board_size, ruleset=ruleset)
    entries: list[CorpusEntry] = []
    seen: set[bytes] = set()

    def add_trajectory(
        trajectory: Iterable[PenteBoard],
        seed: int,
        game_index: int,
    ) -> None:
        for step, board in enumerate(trajectory):
            key = _entry_key(board)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                CorpusEntry(
                    board=board,
                    board_size=board_size,
                    ruleset=ruleset,
                    seed=seed,
                    game_index=game_index,
                    step=step,
                )
            )

    add_trajectory(_guided_line(game, 1), _BASE_SEED + config_index, -1)
    add_trajectory(_guided_line(game, -1), _BASE_SEED + config_index, -2)
    add_trajectory(_guided_capture(game, 1), _BASE_SEED + config_index, -3)
    add_trajectory(_guided_capture(game, -1), _BASE_SEED + config_index, -4)
    add_trajectory(_guided_draw(game), _BASE_SEED + config_index, -5)

    if board_size == 5 and ruleset is PenteRuleset.TOURNAMENT:
        center = board_size // 2
        opening = _apply(game, game.init_board(), center * board_size + center)
        for second_action in range(board_size * board_size):
            if second_action == center * board_size + center:
                continue
            add_trajectory(
                [game.init_board(), opening, _apply(game, opening, second_action)],
                _BASE_SEED + config_index,
                second_action,
            )

    max_plies = {5: 80, 9: 100, 19: 140}[board_size]
    game_index = 0
    while len(entries) < target:
        seed = _BASE_SEED + config_index * 100_000 + game_index
        add_trajectory(
            _random_trajectory(game, seed, max_plies),
            seed,
            game_index,
        )
        game_index += 1
        if game_index > 20_000:
            raise RuntimeError(
                f"unable to build {target} unique positions for "
                f"board_size={board_size}, ruleset={ruleset.value}; "
                f"got {len(entries)}"
            )
    return entries[:target]


def build_corpus() -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    configs = [
        (board_size, ruleset)
        for board_size in (5, 9, 19)
        for ruleset in PenteRuleset
    ]
    for config_index, (board_size, ruleset) in enumerate(configs):
        target = (
            _TOURNAMENT_FIVE_TARGET
            if board_size == 5 and ruleset is PenteRuleset.TOURNAMENT
            else _TARGET_POSITIONS
        )
        entries.extend(
            _build_config_corpus(board_size, ruleset, target, config_index)
        )
    return entries
