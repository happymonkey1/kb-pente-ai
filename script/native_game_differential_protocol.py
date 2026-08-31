"""Protocol and value comparisons for native Pente differential tests."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import subprocess
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from src.game.game import GameStatus
from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset


_REQUEST_MAGIC = b"KBPDIFF3"
_RESPONSE_MAGIC = b"KBPRES3\0"
_HEADER_FORMAT = "<8sI"
_TERMINAL_FORMAT = "<BbB"
_INVALID_ACTION = 0xFFFF
_MAX_ACTIONS = 19 * 19
_WIRE_POSITION_FORMAT = "<361bBBHHBb"
_WIRE_POSITION_SIZE = struct.calcsize(_WIRE_POSITION_FORMAT)
_TERMINAL_SIZE = struct.calcsize(_TERMINAL_FORMAT)

_RULESET_CODES = {
    PenteRuleset.STANDARD: 0,
    PenteRuleset.TOURNAMENT: 1,
    PenteRuleset.FREESTYLE: 2,
}
_REASON_CODES = {None: 0, "line": 1, "capture": 2}
_CODE_REASONS = {value: key for key, value in _REASON_CODES.items()}


class BridgeError(RuntimeError):
    """Raised when the test-only native protocol is malformed or unavailable."""


class DifferentialMismatch(AssertionError):
    """Raised with enough corpus context to reproduce a parity mismatch."""


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    board: PenteBoard
    board_size: int
    ruleset: PenteRuleset
    seed: int
    game_index: int
    step: int


@dataclass(frozen=True, slots=True)
class NativePosition:
    stones: bytes
    captures: tuple[int, int]
    ply: int
    last_action: int | None
    board_size: int
    current_player: int


@dataclass(frozen=True, slots=True)
class NativeTerminal:
    status: GameStatus
    winner: int | None
    reason: str | None

    @property
    def is_terminal(self) -> bool:
        return self.status is not GameStatus.IN_PROGRESS


@dataclass(frozen=True, slots=True)
class NativeSuccessor:
    action: int
    position: NativePosition
    terminal: NativeTerminal


@dataclass(frozen=True, slots=True)
class NativeResponse:
    position: NativePosition
    terminal: NativeTerminal
    legal_words: tuple[int, ...]
    features: np.ndarray
    successors: tuple[NativeSuccessor, ...]


def _read_exact(data: bytes, offset: int, size: int) -> tuple[bytes, int]:
    end = offset + size
    if end > len(data):
        raise BridgeError(
            f"truncated response at byte {offset}: needed {size}, "
            f"only {len(data) - offset} available"
        )
    return data[offset:end], end


def _decode_header(
    data: bytes,
    offset: int,
    expected_magic: bytes,
) -> tuple[int, int]:
    raw, next_offset = _read_exact(
        data, offset, struct.calcsize(_HEADER_FORMAT)
    )
    magic, count = struct.unpack(_HEADER_FORMAT, raw)
    if magic != expected_magic:
        raise BridgeError(f"unexpected protocol magic {magic!r}")
    return count, next_offset


def _encode_position(board: PenteBoard) -> bytes:
    flat = board.board.reshape(-1)
    padding = np.zeros(_MAX_ACTIONS - flat.size, dtype=np.int8)
    stones = np.concatenate((flat, padding))
    captures = (int(board.captures[0]), int(board.captures[1]))
    last_action = _INVALID_ACTION if board.last_action is None else board.last_action
    if board.ply is None:
        raise BridgeError("oracle position has no ply")
    return struct.pack(
        _WIRE_POSITION_FORMAT,
        *[int(stone) for stone in stones],
        *captures,
        int(board.ply),
        int(last_action),
        board.board.shape[0],
        int(board.current_player),
    )


def _decode_position(data: bytes, offset: int) -> tuple[NativePosition, int]:
    raw, next_offset = _read_exact(data, offset, _WIRE_POSITION_SIZE)
    values = struct.unpack(_WIRE_POSITION_FORMAT, raw)
    stones = bytes(np.asarray(values[:_MAX_ACTIONS], dtype=np.int8))
    captures = (int(values[_MAX_ACTIONS]), int(values[_MAX_ACTIONS + 1]))
    ply = int(values[_MAX_ACTIONS + 2])
    raw_last_action = int(values[_MAX_ACTIONS + 3])
    last_action = None if raw_last_action == _INVALID_ACTION else raw_last_action
    board_size = int(values[_MAX_ACTIONS + 4])
    current_player = int(values[_MAX_ACTIONS + 5])
    return (
        NativePosition(
            stones=stones,
            captures=captures,
            ply=ply,
            last_action=last_action,
            board_size=board_size,
            current_player=current_player,
        ),
        next_offset,
    )


def _decode_terminal(data: bytes, offset: int) -> tuple[NativeTerminal, int]:
    raw, next_offset = _read_exact(data, offset, _TERMINAL_SIZE)
    status_code, winner_code, reason_code = struct.unpack(_TERMINAL_FORMAT, raw)
    try:
        status = (GameStatus.IN_PROGRESS, GameStatus.DRAW, GameStatus.WIN)[status_code]
    except IndexError as error:
        raise BridgeError(f"unknown terminal status code {status_code}") from error
    winner = None if winner_code == 0 else int(winner_code)
    if reason_code not in _CODE_REASONS:
        raise BridgeError(f"unknown terminal reason code {reason_code}")
    return (
        NativeTerminal(status=status, winner=winner, reason=_CODE_REASONS[reason_code]),
        next_offset,
    )


def _python_terminal(game: PenteGame, board: PenteBoard) -> NativeTerminal:
    result = game.check_game_end(board)
    return NativeTerminal(
        status=result.status,
        winner=result.winner,
        reason=result.reason,
    )


def _python_stones(board: PenteBoard) -> bytes:
    return bytes(np.asarray(board.board.reshape(-1), dtype=np.int8))


def _context(entry: CorpusEntry, action: int | None = None) -> str:
    action_text = "" if action is None else f", action={action}"
    return (
        f"seed={entry.seed}, ruleset={entry.ruleset.value}, "
        f"board_size={entry.board_size}, game={entry.game_index}, "
        f"step={entry.step}, ply={entry.board.ply}{action_text}"
    )


def _mismatch(
    entry: CorpusEntry,
    field: str,
    python_value: object,
    native_value: object,
    action: int | None = None,
) -> DifferentialMismatch:
    return DifferentialMismatch(
        f"{_context(entry, action)} field={field}: "
        f"python={python_value!r}, native={native_value!r}"
    )


def _compare_terminal(
    entry: CorpusEntry,
    expected: NativeTerminal,
    actual: NativeTerminal,
    action: int | None = None,
) -> None:
    if expected.status != actual.status:
        raise _mismatch(entry, "terminal.status", expected.status, actual.status, action)
    if expected.winner != actual.winner:
        raise _mismatch(entry, "terminal.winner", expected.winner, actual.winner, action)
    if expected.reason != actual.reason:
        raise _mismatch(entry, "terminal.reason", expected.reason, actual.reason, action)


def _compare_position(
    entry: CorpusEntry,
    expected: PenteBoard,
    actual: NativePosition,
    action: int | None = None,
) -> None:
    expected_stones = _python_stones(expected)
    if expected.board.shape[0] != actual.board_size:
        raise _mismatch(entry, "board_size", expected.board.shape[0], actual.board_size, action)
    if expected_stones != actual.stones[: len(expected_stones)]:
        first_difference = next(
            index
            for index, (python_stone, native_stone) in enumerate(
                zip(expected_stones, actual.stones)
            )
            if python_stone != native_stone
        )
        raise _mismatch(
            entry,
            f"stones[{first_difference}]",
            expected_stones[first_difference],
            actual.stones[first_difference],
            action,
        )
    expected_padding = b"\0" * (_MAX_ACTIONS - len(expected_stones))
    if actual.stones[len(expected_stones) :] != expected_padding:
        first_difference = next(
            index
            for index, native_stone in enumerate(actual.stones[len(expected_stones) :])
            if native_stone != 0
        )
        absolute_index = len(expected_stones) + first_difference
        raise _mismatch(
            entry,
            f"stones[{absolute_index}]",
            0,
            actual.stones[absolute_index],
            action,
        )
    expected_captures = (
        int(expected.captures[0]),
        int(expected.captures[1]),
    )
    if expected_captures != actual.captures:
        raise _mismatch(entry, "captures", expected_captures, actual.captures, action)
    if expected.ply != actual.ply:
        raise _mismatch(entry, "ply", expected.ply, actual.ply, action)
    expected_last_action = expected.last_action
    if expected_last_action != actual.last_action:
        raise _mismatch(
            entry,
            "last_action",
            expected_last_action,
            actual.last_action,
            action,
        )
    if expected.current_player != actual.current_player:
        raise _mismatch(
            entry,
            "current_player",
            expected.current_player,
            actual.current_player,
            action,
        )


def _mask_words(mask: np.ndarray) -> tuple[int, ...]:
    words = [0] * 6
    for action in np.flatnonzero(mask):
        words[int(action) // 64] |= 1 << (int(action) % 64)
    return tuple(words)


def _compare_mask(
    entry: CorpusEntry,
    expected_mask: np.ndarray,
    actual_words: tuple[int, ...],
) -> None:
    expected_words = _mask_words(expected_mask)
    if expected_words == actual_words:
        return
    for action in range(entry.board_size * entry.board_size):
        expected = bool(expected_mask[action])
        actual = bool(actual_words[action // 64] & (1 << (action % 64)))
        if expected != actual:
            raise _mismatch(entry, f"legal_mask[{action}]", expected, actual)
    raise _mismatch(entry, "legal_mask.words", expected_words, actual_words)


def _compare_features(
    entry: CorpusEntry,
    expected: np.ndarray,
    actual: np.ndarray,
) -> None:
    if expected.shape != actual.shape:
        raise _mismatch(entry, "features.shape", expected.shape, actual.shape)
    if np.allclose(expected, actual, rtol=1e-6, atol=1e-6):
        return
    difference = np.abs(expected - actual)
    index = tuple(
        int(value) for value in np.unravel_index(np.argmax(difference), difference.shape)
    )
    raise _mismatch(
        entry,
        f"features{index}",
        float(expected[index]),
        float(actual[index]),
    )


def _decode_response(
    data: bytes,
    offset: int,
    entry: CorpusEntry,
) -> tuple[NativeResponse, int]:
    position, offset = _decode_position(data, offset)
    terminal, offset = _decode_terminal(data, offset)
    words: list[int] = []
    for _ in range(6):
        raw, offset = _read_exact(data, offset, 8)
        words.append(struct.unpack("<Q", raw)[0])
    area = entry.board_size * entry.board_size
    feature_bytes, offset = _read_exact(data, offset, 4 * area * 4)
    features = np.frombuffer(feature_bytes, dtype="<f4").copy().reshape(4, area)
    raw_count, offset = _read_exact(data, offset, 2)
    successor_count = struct.unpack("<H", raw_count)[0]
    successors: list[NativeSuccessor] = []
    for _ in range(successor_count):
        raw_action, offset = _read_exact(data, offset, 2)
        action = struct.unpack("<H", raw_action)[0]
        successor_position, offset = _decode_position(data, offset)
        successor_terminal, offset = _decode_terminal(data, offset)
        successors.append(
            NativeSuccessor(
                action=action,
                position=successor_position,
                terminal=successor_terminal,
            )
        )
    return (
        NativeResponse(
            position=position,
            terminal=terminal,
            legal_words=tuple(words),
            features=features,
            successors=tuple(successors),
        ),
        offset,
    )


class NativeBridge:
    """Run one persistent native protocol process for a complete corpus."""

    def __init__(self, runner: Path) -> None:
        self.runner = runner

    def run(self, entries: Sequence[CorpusEntry]) -> Iterator[NativeResponse]:
        payload = bytearray(struct.pack(_HEADER_FORMAT, _REQUEST_MAGIC, len(entries)))
        for entry in entries:
            payload.extend(struct.pack("<B", _RULESET_CODES[entry.ruleset]))
            payload.extend(_encode_position(entry.board))

        try:
            completed = subprocess.run(
                [str(self.runner)],
                input=bytes(payload),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as error:
            raise BridgeError(
                f"unable to execute native runner {self.runner}: {error}"
            ) from error
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise BridgeError(
                f"native runner exited {completed.returncode}: {stderr or '<no stderr>'}"
            )

        offset = 0
        count, offset = _decode_header(completed.stdout, offset, _RESPONSE_MAGIC)
        if count != len(entries):
            raise BridgeError(
                f"native response count {count} does not match request count {len(entries)}"
            )
        for entry in entries:
            response, offset = _decode_response(completed.stdout, offset, entry)
            yield response
        if offset != len(completed.stdout):
            raise BridgeError(
                f"native response has {len(completed.stdout) - offset} trailing bytes"
            )
