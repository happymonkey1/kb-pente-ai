"""Deterministic reachable roots and evaluator cases for native MCTS parity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from src.game.game import GameStatus
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from script.native_game_differential_corpus import build_corpus
from script.native_game_differential_protocol import (
    CorpusEntry,
    _python_terminal,
)
from script.native_mcts_differential_protocol import EvaluatorMode, MctsCase


def _usable(entry: CorpusEntry, game: PenteGame) -> bool:
    if _python_terminal(game, entry.board).status is not GameStatus.IN_PROGRESS:
        return False
    return bool(np.any(game.get_valid_moves(entry.board, entry.board.current_player)))


def _choose_position_entries(
    entries: Iterable[CorpusEntry],
) -> list[tuple[CorpusEntry, str]]:
    """Choose fixed tactical and random roots without changing game semantics."""

    grouped: dict[tuple[int, PenteRuleset], list[CorpusEntry]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.board_size, entry.ruleset)].append(entry)

    selected: list[tuple[CorpusEntry, str]] = []
    for (board_size, ruleset), config_entries in grouped.items():
        game = PenteGame(board_size, ruleset=ruleset)
        usable = [entry for entry in config_entries if _usable(entry, game)]
        if not usable:
            raise RuntimeError(f"no usable MCTS roots for {board_size} {ruleset.value}")

        used_keys: set[bytes] = set()

        def add(label: str, candidates: Iterable[CorpusEntry]) -> None:
            for candidate in candidates:
                key = candidate.board.state_key()
                if key in used_keys or not _usable(candidate, game):
                    continue
                selected.append((candidate, label))
                used_keys.add(key)
                return

        add("initial", (entry for entry in usable if entry.step == 0))
        add(
            "after-opening",
            (entry for entry in usable if entry.step == 1),
        )
        add(
            "tournament-second",
            (
                entry
                for entry in usable
                if ruleset is PenteRuleset.TOURNAMENT and entry.board.ply == 2
            ),
        )

        capture_entries = sorted(
            usable,
            key=lambda entry: (
                int(np.sum(entry.board.captures)),
                int(entry.board.ply or 0),
            ),
            reverse=True,
        )
        add(
            "capture",
            (entry for entry in capture_entries if np.any(entry.board.captures > 0)),
        )

        add(
            "near-line",
            sorted(
                (
                    entry
                    for entry in usable
                    if entry.game_index in (-1, -2)
                ),
                key=lambda entry: int(entry.board.ply or 0),
                reverse=True,
            ),
        )
        add(
            "near-draw",
            sorted(
                (
                    entry
                    for entry in usable
                    if entry.game_index == -5
                ),
                key=lambda entry: int(entry.board.ply or 0),
                reverse=True,
            ),
        )

        random_entries = [entry for entry in usable if entry.game_index >= 0]
        if random_entries:
            random_entries = sorted(
                random_entries,
                key=lambda entry: (entry.seed, entry.game_index, entry.step),
            )
            add("random-early", (random_entries[len(random_entries) // 3],))
            add("random-late", (random_entries[-1],))

        # Small tournament boards can be exhausted before all categories exist.
        # Keep one deterministic fallback root so each configuration remains in
        # the bulk request while retaining the special opening roots above.
        add("fallback", usable)

    return selected


def _case_specs() -> tuple[tuple[EvaluatorMode, int, float, bool], ...]:
    return (
        (EvaluatorMode.UNIFORM_ZERO, 16, 0.0, True),
        (EvaluatorMode.UNIFORM_ZERO, 16, 1.0, True),
        (EvaluatorMode.FIXED_NONUNIFORM, 8, 0.0, True),
        (EvaluatorMode.FIXED_NONUNIFORM, 8, 1.0, True),
        (EvaluatorMode.CONSTANT_POSITIVE, 8, 1.0, True),
        (EvaluatorMode.CONSTANT_NEGATIVE, 8, 0.0, True),
        (EvaluatorMode.ALL_ZERO, 1, 0.0, True),
        (EvaluatorMode.ALL_ZERO, 1, 1.0, True),
        (EvaluatorMode.POSITION_DEPENDENT, 12, 0.0, False),
        (EvaluatorMode.POSITION_DEPENDENT, 12, 1.0, False),
    )


def build_mcts_corpus() -> list[MctsCase]:
    """Build the complete bounded MCTS case corpus from reachable game roots."""

    roots = _choose_position_entries(build_corpus())
    cases: list[MctsCase] = []
    for entry, label in roots:
        for mode, budget, temperature, exact in _case_specs():
            if (
                entry.board_size == 5
                and entry.ruleset is PenteRuleset.TOURNAMENT
            ):
                # The documented 5 by 5 tournament opening has no legal
                # third-ply point. Keep its forced-opening roots at one
                # simulation, before search reaches that exhausted state.
                budget = 1
            mode_label = mode.name.lower().replace("_", "-")
            cases.append(
                MctsCase(
                    board=entry.board,
                    board_size=entry.board_size,
                    ruleset=entry.ruleset,
                    seed=entry.seed,
                    game_index=entry.game_index,
                    step=entry.step,
                    label=f"{label}:{mode_label}:temp={temperature:g}",
                    mode=mode,
                    simulation_budget=budget,
                    temperature=temperature,
                    exact=exact,
                )
            )
    return cases
