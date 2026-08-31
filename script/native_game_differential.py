"""Differentially exercise the native game layer against the Python oracle.

The native executable is a test-only binary protocol endpoint. One process
receives the complete corpus and returns one complete state snapshot plus all
legal successors per request. The core library itself has no Python or Torch
dependency.
"""

from __future__ import annotations

import argparse
from collections import Counter
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.game.game import GameStatus
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from script.native_game_differential_corpus import build_corpus
from script.native_game_differential_protocol import (
    BridgeError,
    CorpusEntry,
    DifferentialMismatch,
    NativeBridge,
    NativeResponse,
    _HEADER_FORMAT,
    _RESPONSE_MAGIC,
    _compare_features,
    _compare_mask,
    _compare_position,
    _compare_terminal,
    _decode_header,
    _python_terminal,
)


def _run_protocol_unit_tests(runner: Path) -> None:
    try:
        _decode_header(b"bad", 0, _RESPONSE_MAGIC)
    except BridgeError:
        pass
    else:
        raise AssertionError("truncated response header was accepted")

    malformed = struct.pack(_HEADER_FORMAT, b"WRONG!!!", 0)
    try:
        _decode_header(malformed, 0, _RESPONSE_MAGIC)
    except BridgeError:
        pass
    else:
        raise AssertionError("unexpected response magic was accepted")

    completed = subprocess.run(
        [str(runner)],
        input=b"bad",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode == 0 or b"protocol error" not in completed.stderr:
        raise AssertionError("native runner did not reject malformed input")


def _compare_entry(entry: CorpusEntry, response: NativeResponse) -> int:
    game = PenteGame(entry.board_size, ruleset=entry.ruleset)
    expected_terminal = _python_terminal(game, entry.board)
    _compare_position(entry, entry.board, response.position)
    _compare_terminal(entry, expected_terminal, response.terminal)

    expected_mask = game.get_valid_moves(entry.board, entry.board.current_player)
    _compare_mask(entry, expected_mask, response.legal_words)
    _compare_features(
        entry,
        entry.board.feature_planes().reshape(4, -1),
        response.features,
    )

    expected_actions = [int(action) for action in np.flatnonzero(expected_mask)]
    expected_successors = (
        [] if expected_terminal.status is not GameStatus.IN_PROGRESS else expected_actions
    )
    actual_actions = [successor.action for successor in response.successors]
    if expected_successors != actual_actions:
        raise DifferentialMismatch(
            f"{entry.ruleset.value} board_size={entry.board_size} "
            f"seed={entry.seed} game={entry.game_index} step={entry.step} "
            f"ply={entry.board.ply} field=successor.actions: "
            f"python={expected_successors!r}, native={actual_actions!r}"
        )

    for successor in response.successors:
        child, _ = game.apply_action(
            entry.board, entry.board.current_player, successor.action
        )
        child_terminal = _python_terminal(game, child)
        _compare_position(entry, child, successor.position, successor.action)
        _compare_terminal(entry, child_terminal, successor.terminal, successor.action)
    return len(response.successors)


def run_differential(runner: Path) -> None:
    _run_protocol_unit_tests(runner)
    started = time.perf_counter()
    corpus = build_corpus()
    entries_by_config: dict[tuple[int, PenteRuleset], int] = {}
    terminal_counts: Counter[tuple[GameStatus, str | None]] = Counter()
    capture_positions = 0
    for entry in corpus:
        key = (entry.board_size, entry.ruleset)
        entries_by_config[key] = entries_by_config.get(key, 0) + 1
        terminal = _python_terminal(
            PenteGame(entry.board_size, ruleset=entry.ruleset), entry.board
        )
        terminal_counts[(terminal.status, terminal.reason)] += 1
        if any(int(value) > 0 for value in entry.board.captures):
            capture_positions += 1

    if len(corpus) < 3_000:
        raise RuntimeError(f"corpus has only {len(corpus)} positions")
    if capture_positions == 0:
        raise RuntimeError("corpus contains no positions with captures")
    if not any(status is GameStatus.WIN for status, _ in terminal_counts):
        raise RuntimeError("corpus contains no terminal wins")
    if not any(status is GameStatus.DRAW for status, _ in terminal_counts):
        raise RuntimeError("corpus contains no terminal draws")

    successors = 0
    for entry, response in zip(corpus, NativeBridge(runner).run(corpus)):
        successors += _compare_entry(entry, response)
    elapsed = time.perf_counter() - started
    print("native differential protocol self-tests: PASS")
    for (board_size, ruleset), count in entries_by_config.items():
        print(f"corpus board_size={board_size} ruleset={ruleset.value} positions={count}")
    print(f"corpus positions_with_captures={capture_positions}")
    for (status, reason), count in sorted(
        terminal_counts.items(), key=lambda item: (item[0][0].value, str(item[0][1]))
    ):
        print(f"corpus terminal status={status.value} reason={reason} positions={count}")
    print(f"corpus positions={len(corpus)} legal_successors={successors}")
    print(f"differential runtime_seconds={elapsed:.3f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner",
        required=True,
        type=Path,
        help="path to the built kb_pente_native_diff_runner executable",
    )
    args = parser.parse_args(argv)

    try:
        run_differential(args.runner)
    except (BridgeError, DifferentialMismatch, RuntimeError, AssertionError) as error:
        print(f"native differential FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
