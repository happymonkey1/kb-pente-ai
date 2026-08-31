"""Differentially exercise native MCTS against the corrected Python oracle."""

from __future__ import annotations

import argparse
from collections import Counter
import math
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import LeafSelection, MCTS, MCTSArgs
from script.native_mcts_differential_corpus import build_mcts_corpus
from script.native_mcts_differential_protocol import (
    _HEADER_FORMAT,
    _PROTOCOL_VERSION,
    _REQUEST_MAGIC,
    _RESPONSE_MAGIC,
    DifferentialMismatch,
    EvaluatorMode,
    MctsCase,
    NativeMctsBridge,
    NativeMctsResponse,
    NativeMctsTelemetry,
    BridgeError,
    encode_request,
    _decode_header,
)


# Native selection stores float32 values while the Python oracle accumulates
# most arithmetic in float64. This tolerance is intentionally tight; only the
# position-dependent cases permit a one-visit difference near a tied score.
_FLOAT_TOLERANCE = 2.0e-5
_APPROXIMATE_VISIT_TOLERANCE = 1


def _position_checksum(position: PenteBoard) -> int:
    checksum = int(position.ply or 0)
    checksum += 3 * int(position.captures[0])
    checksum += 5 * int(position.captures[1])
    checksum += 11 if position.current_player == 1 else 17
    for index, stone in enumerate(position.board.reshape(-1)):
        checksum += (index + 1) * (int(stone) + 1)
    return checksum


class DeterministicEvaluator:
    """Match the six allocation-free evaluator formulas in the native runner."""

    def __init__(self, action_size: int, mode: EvaluatorMode) -> None:
        self.action_size = action_size
        self.mode = mode
        self.calls = 0

    def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        self.calls += 1
        policy = np.zeros(self.action_size, dtype=np.float64)
        checksum = _position_checksum(position)
        for index in range(self.action_size):
            if self.mode in (
                EvaluatorMode.UNIFORM_ZERO,
                EvaluatorMode.CONSTANT_POSITIVE,
                EvaluatorMode.CONSTANT_NEGATIVE,
            ):
                probability = 1.0
            elif self.mode is EvaluatorMode.FIXED_NONUNIFORM:
                probability = (1.0, 0.5, 0.25, 0.125)[index % 4]
            elif self.mode is EvaluatorMode.ALL_ZERO:
                probability = 0.0
            else:
                probability = (0.0, 0.25, 0.5, 1.0)[
                    (checksum + 7 * index) % 4
                ]
            policy[index] = probability

        if self.mode is EvaluatorMode.CONSTANT_POSITIVE:
            value = 0.25
        elif self.mode is EvaluatorMode.CONSTANT_NEGATIVE:
            value = -0.25
        elif self.mode is EvaluatorMode.POSITION_DEPENDENT:
            value = (-0.25, 0.0, 0.25)[checksum % 3]
        else:
            value = 0.0
        return policy, value


class InstrumentedMCTS(MCTS):
    """Add only the selection counters absent from the Python telemetry type."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.selected_leaves = 0
        self.terminal_simulations = 0
        self.maximum_selected_path_depth = 0

    def select_leaf(
        self,
        root: PenteBoard,
        root_key: bytes | None = None,
        root_priors: np.ndarray | None = None,
    ) -> LeafSelection:
        selection = super().select_leaf(root, root_key, root_priors)
        self.selected_leaves += 1
        self.maximum_selected_path_depth = max(
            self.maximum_selected_path_depth, len(selection.path)
        )
        if selection.terminal_result.is_terminal:
            self.terminal_simulations += 1
        return selection


@dataclass(frozen=True, slots=True)
class PythonMctsResult:
    visits: tuple[int, ...]
    value_sums: tuple[float, ...]
    policy: tuple[float, ...]
    telemetry: NativeMctsTelemetry


def _run_python_case(case: MctsCase) -> PythonMctsResult:
    game = PenteGame(case.board_size, ruleset=case.ruleset)
    evaluator = DeterministicEvaluator(case.board_size * case.board_size, case.mode)
    search = InstrumentedMCTS(
        game,
        evaluator,
        MCTSArgs(
            num_simulations=case.simulation_budget,
            c_puct=case.c_puct,
            root_noise_epsilon=0.0,
        ),
        np.random.default_rng(0),
    )
    policy = search.get_action_prob(
        case.board,
        temp=case.temperature,
        add_root_noise=False,
    )
    root_key = game.to_string(case.board)
    visits: list[int] = []
    value_sums: list[float] = []
    for action in range(case.board_size * case.board_size):
        edge_key = (root_key, action)
        visit_count = int(search.nsa.get(edge_key, 0))
        visits.append(visit_count)
        value_sums.append(
            float(search.qsa.get(edge_key, 0.0) * visit_count)
        )

    python_telemetry = search.telemetry(case.board)
    telemetry = NativeMctsTelemetry(
        completed_simulations=python_telemetry.simulations,
        evaluator_completions=python_telemetry.evaluator_calls,
        terminal_simulations=search.terminal_simulations,
        selected_leaves=search.selected_leaves,
        max_selected_path_depth=search.maximum_selected_path_depth,
        root_legal_actions=python_telemetry.root_legal_actions,
        root_edge_visits=python_telemetry.root_edge_visits,
        root_children_visited=python_telemetry.root_children_visited,
        root_visit_entropy=python_telemetry.root_visit_entropy,
        root_max_visit_share=python_telemetry.root_max_visit_share,
        root_collapse_eligible=python_telemetry.root_collapse_eligible,
        root_search_collapsed=python_telemetry.root_search_collapsed,
        invalid_policy_fallbacks=python_telemetry.invalid_policy_fallbacks,
        zero_visit_fallbacks=python_telemetry.zero_visit_fallbacks,
    )
    return PythonMctsResult(
        visits=tuple(visits),
        value_sums=tuple(value_sums),
        policy=tuple(float(value) for value in policy),
        telemetry=telemetry,
    )


def _context(case: MctsCase, action: int | None = None) -> str:
    action_text = "" if action is None else f", action={action}"
    return (
        f"seed={case.seed}, ruleset={case.ruleset.value}, "
        f"board_size={case.board_size}, game={case.game_index}, "
        f"step={case.step}, ply={case.board.ply}, label={case.label}, "
        f"mode={case.mode.name.lower()}, budget={case.simulation_budget}, "
        f"temperature={case.temperature:g}{action_text}"
    )


def _mismatch(
    case: MctsCase,
    field: str,
    python_value: object,
    native_value: object,
    action: int | None = None,
) -> DifferentialMismatch:
    return DifferentialMismatch(
        f"{_context(case, action)} field={field}: "
        f"python={python_value!r}, native={native_value!r}"
    )


def _compare_float(
    case: MctsCase,
    field: str,
    expected: float,
    actual: float,
    tolerance: float = _FLOAT_TOLERANCE,
    action: int | None = None,
) -> None:
    if not math.isfinite(actual) or not math.isclose(
        expected, actual, rel_tol=tolerance, abs_tol=tolerance
    ):
        raise _mismatch(case, field, expected, actual, action)


def _compare_telemetry(
    case: MctsCase,
    expected: NativeMctsTelemetry,
    actual: NativeMctsTelemetry,
) -> None:
    integer_fields = (
        "completed_simulations",
        "evaluator_completions",
        "terminal_simulations",
        "selected_leaves",
        "max_selected_path_depth",
        "root_legal_actions",
        "root_edge_visits",
        "invalid_policy_fallbacks",
        "zero_visit_fallbacks",
    )
    for field in integer_fields:
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if expected_value != actual_value:
            raise _mismatch(case, f"telemetry.{field}", expected_value, actual_value)

    children_difference = abs(
        expected.root_children_visited - actual.root_children_visited
    )
    if children_difference > (
        _APPROXIMATE_VISIT_TOLERANCE if not case.exact else 0
    ):
        raise _mismatch(
            case,
            "telemetry.root_children_visited",
            expected.root_children_visited,
            actual.root_children_visited,
        )
    _compare_float(
        case,
        "telemetry.root_visit_entropy",
        expected.root_visit_entropy,
        actual.root_visit_entropy,
    )
    _compare_float(
        case,
        "telemetry.root_max_visit_share",
        expected.root_max_visit_share,
        actual.root_max_visit_share,
    )
    for field in ("root_collapse_eligible", "root_search_collapsed"):
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if expected_value != actual_value:
            raise _mismatch(case, f"telemetry.{field}", expected_value, actual_value)


def _compare_case(
    case: MctsCase,
    expected: PythonMctsResult,
    actual: NativeMctsResponse,
) -> None:
    visit_tolerance = (
        0 if case.exact else _APPROXIMATE_VISIT_TOLERANCE
    )
    for action, (python_visits, native_visits) in enumerate(
        zip(expected.visits, actual.visits)
    ):
        if abs(python_visits - native_visits) > visit_tolerance:
            raise _mismatch(
                case,
                "root_edge_visits",
                python_visits,
                native_visits,
                action,
            )
    if len(expected.visits) != len(actual.visits):
        raise _mismatch(
            case,
            "root_edge_visits.length",
            len(expected.visits),
            len(actual.visits),
        )

    for action, (python_value, native_value) in enumerate(
        zip(expected.value_sums, actual.value_sums)
    ):
        _compare_float(
            case,
            "root_value_sums",
            python_value,
            native_value,
            action=action,
        )
    for action, (python_value, native_value) in enumerate(
        zip(expected.policy, actual.policy)
    ):
        _compare_float(
            case,
            "root_policy",
            python_value,
            native_value,
            action=action,
        )
    _compare_telemetry(case, expected.telemetry, actual.telemetry)


def _assert_runner_rejects(
    runner: Path,
    payload: bytes,
    description: str,
) -> None:
    completed = subprocess.run(
        [str(runner)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode == 0:
        raise AssertionError(f"native runner accepted {description}")
    if b"protocol error" not in completed.stderr:
        raise AssertionError(
            f"native runner rejected {description} without protocol diagnostics"
        )


def _run_protocol_unit_tests(runner: Path) -> None:
    try:
        _decode_header(b"short", 0, _RESPONSE_MAGIC)
    except BridgeError:
        pass
    else:
        raise AssertionError("truncated response header was accepted")

    malformed = struct.pack(
        _HEADER_FORMAT, b"WRONG!!!", _PROTOCOL_VERSION, 0
    )
    try:
        _decode_header(malformed, 0, _RESPONSE_MAGIC)
    except BridgeError:
        pass
    else:
        raise AssertionError("unexpected response magic was accepted")

    _assert_runner_rejects(runner, b"bad", "a malformed header")
    truncated = struct.pack(
        _HEADER_FORMAT, _REQUEST_MAGIC, _PROTOCOL_VERSION, 1
    )
    _assert_runner_rejects(runner, truncated, "a truncated record")

    game = PenteGame(5)
    base_case = MctsCase(
        board=game.init_board(),
        board_size=5,
        ruleset=game.ruleset,
        seed=1,
        game_index=-1,
        step=0,
        label="protocol",
        mode=EvaluatorMode.UNIFORM_ZERO,
        simulation_budget=1,
        temperature=0.0,
    )
    record_offset = struct.calcsize(_HEADER_FORMAT)

    invalid_mode = bytearray(encode_request([base_case]))
    invalid_mode[record_offset + 1] = 255
    _assert_runner_rejects(runner, bytes(invalid_mode), "an unknown evaluator mode")

    invalid_budget = bytearray(encode_request([base_case]))
    struct.pack_into("<I", invalid_budget, record_offset + 2, 0)
    _assert_runner_rejects(runner, bytes(invalid_budget), "a zero simulation budget")

    invalid_temperature = bytearray(encode_request([base_case]))
    struct.pack_into(
        "<f",
        invalid_temperature,
        record_offset + 10,
        float("nan"),
    )
    _assert_runner_rejects(
        runner, bytes(invalid_temperature), "a non-finite temperature"
    )


def run_differential(runner: Path) -> None:
    _run_protocol_unit_tests(runner)
    started = time.perf_counter()
    cases = build_mcts_corpus()
    if len(cases) < 300:
        raise RuntimeError(f"MCTS differential corpus has only {len(cases)} cases")

    expected_configs = {
        (board_size, ruleset.value)
        for board_size in (5, 9, 19)
        for ruleset in PenteRuleset
    }
    root_counts: Counter[tuple[int, str]] = Counter(
        (case.board_size, case.ruleset.value) for case in cases
    )
    if set(root_counts) != expected_configs:
        raise RuntimeError(
            f"MCTS corpus configuration coverage is incomplete: {sorted(root_counts)}"
        )
    mode_counts: Counter[str] = Counter(case.mode.name.lower() for case in cases)
    if set(mode_counts) != {mode.name.lower() for mode in EvaluatorMode}:
        raise RuntimeError("MCTS corpus evaluator-mode coverage is incomplete")
    if {case.temperature for case in cases} != {0.0, 1.0}:
        raise RuntimeError("MCTS corpus temperature coverage is incomplete")
    labels = Counter(case.label.split(":", 1)[0] for case in cases)
    for required_label in ("capture", "near-line", "near-draw", "random-early"):
        if labels[required_label] == 0:
            raise RuntimeError(
                f"MCTS corpus lacks required {required_label} roots"
            )
    roots_with_captures = {
        (case.board_size, case.ruleset.value, case.board.state_key())
        for case in cases
        if np.any(case.board.captures > 0)
    }
    exact_count = sum(case.exact for case in cases)
    python_results = [_run_python_case(case) for case in cases]
    native_results = list(NativeMctsBridge(runner).run(cases))
    if len(native_results) != len(cases):
        raise BridgeError(
            f"native response count {len(native_results)} does not match "
            f"case count {len(cases)}"
        )
    for case, expected, actual in zip(cases, python_results, native_results):
        _compare_case(case, expected, actual)

    print("native MCTS differential protocol self-tests: PASS")
    print(
        f"MCTS roots={len(root_counts)} cases={len(cases)} "
        f"exact_cases={exact_count} approximate_cases={len(cases) - exact_count}"
    )
    for (board_size, ruleset), count in sorted(root_counts.items()):
        print(
            f"MCTS corpus board_size={board_size} ruleset={ruleset} "
            f"cases={count}"
        )
    for mode, count in sorted(mode_counts.items()):
        print(f"MCTS evaluator mode={mode} cases={count}")
    print(f"MCTS roots_with_captures={len(roots_with_captures)}")
    for label, count in sorted(labels.items()):
        print(f"MCTS root category={label} cases={count}")
    print(f"MCTS differential runtime_seconds={time.perf_counter() - started:.3f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner",
        required=True,
        type=Path,
        help="path to the test-only native MCTS differential runner",
    )
    args = parser.parse_args(argv)
    try:
        run_differential(args.runner)
    except (BridgeError, DifferentialMismatch, RuntimeError, AssertionError) as error:
        print(f"native MCTS differential FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
