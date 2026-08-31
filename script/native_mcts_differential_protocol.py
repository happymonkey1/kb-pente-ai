"""Versioned bulk protocol for deterministic Python/native MCTS parity tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
import subprocess
from pathlib import Path
from typing import Iterator, Sequence

from src.game.pente.pente_board import PenteBoard
from src.game.pente.rules import PenteRuleset

from script.native_game_differential_protocol import (
    _MAX_ACTIONS,
    _encode_position,
)


_REQUEST_MAGIC = b"KPMCTS6B"
_RESPONSE_MAGIC = b"KPMRES6B"
_PROTOCOL_VERSION = 1
_HEADER_FORMAT = "<8sHI"
_RECORD_FORMAT = "<BBIff"
_MAX_UINT32 = (1 << 32) - 1


class EvaluatorMode(IntEnum):
    """Deterministic evaluator formula shared by the Python and C++ tests."""

    UNIFORM_ZERO = 0
    FIXED_NONUNIFORM = 1
    CONSTANT_POSITIVE = 2
    CONSTANT_NEGATIVE = 3
    ALL_ZERO = 4
    POSITION_DEPENDENT = 5


@dataclass(frozen=True, slots=True)
class MctsCase:
    """One root/configuration request in the bulk MCTS differential corpus."""

    board: PenteBoard
    board_size: int
    ruleset: PenteRuleset
    seed: int
    game_index: int
    step: int
    label: str
    mode: EvaluatorMode
    simulation_budget: int
    temperature: float
    c_puct: float = 1.5
    exact: bool = True


@dataclass(frozen=True, slots=True)
class NativeMctsTelemetry:
    completed_simulations: int
    evaluator_completions: int
    terminal_simulations: int
    selected_leaves: int
    max_selected_path_depth: int
    root_legal_actions: int
    root_edge_visits: int
    root_children_visited: int
    root_visit_entropy: float
    root_max_visit_share: float
    root_collapse_eligible: bool
    root_search_collapsed: bool
    invalid_policy_fallbacks: int
    zero_visit_fallbacks: int


@dataclass(frozen=True, slots=True)
class NativeMctsResponse:
    visits: tuple[int, ...]
    value_sums: tuple[float, ...]
    policy: tuple[float, ...]
    telemetry: NativeMctsTelemetry


class BridgeError(RuntimeError):
    """Raised when the test-only native MCTS protocol is malformed or unavailable."""


class DifferentialMismatch(AssertionError):
    """Raised with enough corpus context to reproduce a parity mismatch."""


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
    magic, version, count = struct.unpack(_HEADER_FORMAT, raw)
    if magic != expected_magic:
        raise BridgeError(f"unexpected protocol magic {magic!r}")
    if version != _PROTOCOL_VERSION:
        raise BridgeError(f"unsupported protocol version {version}")
    return count, next_offset


def encode_request(cases: Sequence[MctsCase]) -> bytes:
    """Encode all cases into one explicit little-endian request payload."""

    if len(cases) > _MAX_UINT32:
        raise BridgeError("too many MCTS differential cases")
    ruleset_codes = {
        PenteRuleset.STANDARD: 0,
        PenteRuleset.TOURNAMENT: 1,
        PenteRuleset.FREESTYLE: 2,
    }
    payload = bytearray(
        struct.pack(
            _HEADER_FORMAT,
            _REQUEST_MAGIC,
            _PROTOCOL_VERSION,
            len(cases),
        )
    )
    for case in cases:
        try:
            ruleset_code = ruleset_codes[case.ruleset]
        except KeyError as error:
            raise BridgeError(f"unknown ruleset {case.ruleset!r}") from error
        if case.simulation_budget < 1 or case.simulation_budget > _MAX_UINT32:
            raise BridgeError("simulation budget must fit a positive uint32")
        if not 0 <= int(case.mode) <= int(max(EvaluatorMode)):
            raise BridgeError(f"unknown evaluator mode {case.mode!r}")
        payload.extend(
            struct.pack(
                _RECORD_FORMAT,
                ruleset_code,
                int(case.mode),
                case.simulation_budget,
                case.c_puct,
                case.temperature,
            )
        )
        payload.extend(_encode_position(case.board))
    return bytes(payload)


def _decode_u64(data: bytes, offset: int) -> tuple[int, int]:
    raw, next_offset = _read_exact(data, offset, 8)
    return struct.unpack("<Q", raw)[0], next_offset


def _decode_u32(data: bytes, offset: int) -> tuple[int, int]:
    raw, next_offset = _read_exact(data, offset, 4)
    return struct.unpack("<I", raw)[0], next_offset


def _decode_float(data: bytes, offset: int) -> tuple[float, int]:
    raw, next_offset = _read_exact(data, offset, 4)
    return struct.unpack("<f", raw)[0], next_offset


def _decode_response(
    data: bytes,
    offset: int,
    case: MctsCase,
) -> tuple[NativeMctsResponse, int]:
    raw_count, offset = _read_exact(data, offset, 2)
    active_actions = struct.unpack("<H", raw_count)[0]
    expected_actions = case.board_size * case.board_size
    if active_actions != expected_actions:
        raise BridgeError(
            f"response action count {active_actions} does not match "
            f"board area {expected_actions}"
        )
    if active_actions > _MAX_ACTIONS:
        raise BridgeError("response action count exceeds native capacity")

    visits: list[int] = []
    for _ in range(active_actions):
        visit, offset = _decode_u32(data, offset)
        visits.append(visit)
    value_sums: list[float] = []
    for _ in range(active_actions):
        value_sum, offset = _decode_float(data, offset)
        value_sums.append(value_sum)
    policy: list[float] = []
    for _ in range(active_actions):
        policy_value, offset = _decode_float(data, offset)
        policy.append(policy_value)

    telemetry_values: list[int] = []
    for _ in range(8):
        value, offset = _decode_u64(data, offset)
        telemetry_values.append(value)
    entropy, offset = _decode_float(data, offset)
    maximum_share, offset = _decode_float(data, offset)
    invalid_fallbacks, offset = _decode_u64(data, offset)
    zero_visit_fallbacks, offset = _decode_u64(data, offset)
    raw_flags, offset = _read_exact(data, offset, 2)
    collapse_eligible, search_collapsed = struct.unpack("<BB", raw_flags)
    if collapse_eligible > 1 or search_collapsed > 1:
        raise BridgeError("invalid boolean telemetry flag")

    telemetry = NativeMctsTelemetry(
        completed_simulations=telemetry_values[0],
        evaluator_completions=telemetry_values[1],
        terminal_simulations=telemetry_values[2],
        selected_leaves=telemetry_values[3],
        max_selected_path_depth=telemetry_values[4],
        root_legal_actions=telemetry_values[5],
        root_edge_visits=telemetry_values[6],
        root_children_visited=telemetry_values[7],
        root_visit_entropy=entropy,
        root_max_visit_share=maximum_share,
        root_collapse_eligible=bool(collapse_eligible),
        root_search_collapsed=bool(search_collapsed),
        invalid_policy_fallbacks=invalid_fallbacks,
        zero_visit_fallbacks=zero_visit_fallbacks,
    )
    return (
        NativeMctsResponse(
            visits=tuple(visits),
            value_sums=tuple(value_sums),
            policy=tuple(policy),
            telemetry=telemetry,
        ),
        offset,
    )


class NativeMctsBridge:
    """Run one native MCTS process for a complete deterministic case corpus."""

    def __init__(self, runner: Path, timeout_seconds: float = 600.0) -> None:
        self.runner = runner
        self.timeout_seconds = timeout_seconds

    def run(self, cases: Sequence[MctsCase]) -> Iterator[NativeMctsResponse]:
        payload = encode_request(cases)
        try:
            completed = subprocess.run(
                [str(self.runner)],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise BridgeError(
                f"native MCTS runner exceeded {self.timeout_seconds:.1f}s"
            ) from error
        except OSError as error:
            raise BridgeError(
                f"unable to execute native MCTS runner {self.runner}: {error}"
            ) from error
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise BridgeError(
                f"native MCTS runner exited {completed.returncode}: "
                f"{stderr or '<no stderr>'}"
            )

        offset = 0
        count, offset = _decode_header(
            completed.stdout, offset, _RESPONSE_MAGIC
        )
        if count != len(cases):
            raise BridgeError(
                f"native response count {count} does not match request count "
                f"{len(cases)}"
            )
        for case in cases:
            response, offset = _decode_response(completed.stdout, offset, case)
            yield response
        if offset != len(completed.stdout):
            raise BridgeError(
                f"native response has {len(completed.stdout) - offset} trailing bytes"
            )
