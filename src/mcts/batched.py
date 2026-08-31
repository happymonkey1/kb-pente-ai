from __future__ import annotations

from dataclasses import dataclass, field
import operator
import time
from typing import Any, Mapping, Sequence, SupportsIndex, SupportsFloat, cast

import numpy as np

from src.game.pente.pente_board import PenteBoard
from src.mcts.mcts_v2 import LeafSelection, MCTS


@dataclass(frozen=True, slots=True)
class BatchedSearchTelemetry:
    root_count: int
    simulation_waves: int
    selected_leaves: int
    evaluation_requests: int
    unique_evaluations: int
    evaluator_calls: int
    inference_batch_sizes: tuple[int, ...]
    min_inference_batch_size: int
    mean_inference_batch_size: float
    median_inference_batch_size: float
    p95_inference_batch_size: float
    max_inference_batch_size: int
    duplicate_leaf_rate: float
    native_select_seconds: float = 0.0
    native_deduplication_seconds: float = 0.0
    native_feature_encode_seconds: float = 0.0
    native_backup_seconds: float = 0.0
    model_inference_seconds: float = 0.0
    host_to_device_seconds: float = 0.0
    device_to_host_seconds: float = 0.0
    inference_wait_seconds: float = 0.0
    native_worker_threads: int = 0
    native_worker_busy_seconds: float = 0.0
    native_worker_capacity_seconds: float = 0.0
    native_worker_busy_percent: float = 0.0


@dataclass(frozen=True, slots=True)
class BatchedSearchResult:
    policies: list[np.ndarray]
    telemetry: BatchedSearchTelemetry


@dataclass(slots=True)
class SearchSession:
    search: MCTS
    root: PenteBoard
    temperature: float
    add_root_noise: bool
    root_key: bytes = b""
    root_priors: np.ndarray | None = None
    completed_simulations: int = 0

    def __post_init__(self) -> None:
        if self.search.game.check_game_end(self.root).is_terminal:
            raise ValueError("Cannot search a terminal root")
        if self.temperature < 0:
            raise ValueError("Temperature cannot be negative")
        self.root_key = self.search.game.to_string(self.root)
        self.root_priors = self.search.root_priors(
            self.root,
            self.add_root_noise,
        )

    @property
    def is_complete(self) -> bool:
        return self.completed_simulations >= self.search.args.num_simulations

    def select_evaluation_leaf(self) -> tuple[LeafSelection | None, int]:
        selected_leaves = 0
        while not self.is_complete:
            selection = self.search.select_leaf(
                self.root,
                self.root_key,
                self.root_priors,
            )
            selected_leaves += 1
            if not selection.terminal_result.is_terminal:
                return selection, selected_leaves

            self.search.expand_and_backup(selection)
            self.search.record_simulation()
            self.completed_simulations += 1
        return None, selected_leaves

    def accept_evaluation(
        self,
        selection: LeafSelection,
        evaluation: tuple[np.ndarray, float],
        batch_size: int,
        elapsed_seconds: float,
    ) -> None:
        if self.is_complete:
            raise RuntimeError("Cannot evaluate a completed search")
        self.search.expand_and_backup(selection, evaluation)
        self.search.record_batch_evaluation(batch_size, elapsed_seconds)
        self.search.record_simulation()
        self.completed_simulations += 1
        if self.root_priors is None:
            self.root_priors = self.search.root_priors(
                self.root,
                self.add_root_noise,
            )

    def policy(self) -> np.ndarray:
        if not self.is_complete:
            raise RuntimeError("Cannot obtain a policy from an incomplete search")
        return self.search.action_prob_from_counts(self.root, self.temperature)


@dataclass(slots=True)
class BatchedSearchAccumulator:
    root_count: int
    simulation_waves: int = 0
    selected_leaves: int = 0
    evaluation_requests: int = 0
    unique_evaluations: int = 0
    evaluator_calls: int = 0
    inference_batch_sizes: list[int] = field(default_factory=list)
    native_select_seconds: float = 0.0
    native_deduplication_seconds: float = 0.0
    native_feature_encode_seconds: float = 0.0
    native_backup_seconds: float = 0.0
    model_inference_seconds: float = 0.0
    host_to_device_seconds: float = 0.0
    device_to_host_seconds: float = 0.0
    inference_wait_seconds: float = 0.0
    native_worker_threads: int = 0
    native_worker_busy_seconds: float = 0.0
    native_worker_capacity_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.root_count < 1:
            raise ValueError("At least one root is required")

    def telemetry(self) -> BatchedSearchTelemetry:
        batch_sizes = self.inference_batch_sizes
        duplicate_rate = (
            1.0 - self.unique_evaluations / self.evaluation_requests
            if self.evaluation_requests
            else 0.0
        )
        return BatchedSearchTelemetry(
            root_count=self.root_count,
            simulation_waves=self.simulation_waves,
            selected_leaves=self.selected_leaves,
            evaluation_requests=self.evaluation_requests,
            unique_evaluations=self.unique_evaluations,
            evaluator_calls=self.evaluator_calls,
            inference_batch_sizes=tuple(batch_sizes),
            min_inference_batch_size=min(batch_sizes, default=0),
            mean_inference_batch_size=(
                float(np.mean(batch_sizes)) if batch_sizes else 0.0
            ),
            median_inference_batch_size=(
                float(np.median(batch_sizes)) if batch_sizes else 0.0
            ),
            p95_inference_batch_size=(
                float(np.percentile(batch_sizes, 95)) if batch_sizes else 0.0
            ),
            max_inference_batch_size=max(batch_sizes, default=0),
            duplicate_leaf_rate=duplicate_rate,
            native_select_seconds=self.native_select_seconds,
            native_deduplication_seconds=self.native_deduplication_seconds,
            native_feature_encode_seconds=self.native_feature_encode_seconds,
            native_backup_seconds=self.native_backup_seconds,
            model_inference_seconds=self.model_inference_seconds,
            host_to_device_seconds=self.host_to_device_seconds,
            device_to_host_seconds=self.device_to_host_seconds,
            inference_wait_seconds=self.inference_wait_seconds,
            native_worker_threads=self.native_worker_threads,
            native_worker_busy_seconds=self.native_worker_busy_seconds,
            native_worker_capacity_seconds=self.native_worker_capacity_seconds,
            native_worker_busy_percent=(
                100.0 * self.native_worker_busy_seconds
                / self.native_worker_capacity_seconds
                if self.native_worker_capacity_seconds > 0.0
                else 0.0
            ),
        )

    def record_native_wave(
        self,
        wave: Any,
        timing: Mapping[str, Any] | None,
        worker_threads: int,
        selected_leaves: int,
    ) -> None:
        """Record one native selection/inference/backup generation.

        ``NativeWave.size`` is the unique evaluator row count and
        ``NativeWave.raw_size`` is the raw evaluator request count. The
        selected-leaf count is supplied separately because native selection
        can resolve terminal leaves without producing evaluator requests.
        Stage timing is read from the native binding's latest-generation
        report so retries do not accidentally include an older generation.
        """

        raw_size = _nonnegative_int(wave.raw_size)
        unique_size = _nonnegative_int(wave.size)
        selected_count = _nonnegative_int(selected_leaves)
        parsed_worker_threads = _nonnegative_int(worker_threads)
        if unique_size > raw_size:
            raise ValueError("Native unique requests cannot exceed raw requests")

        self.simulation_waves += 1
        self.selected_leaves += selected_count
        self.evaluation_requests += raw_size
        self.unique_evaluations += unique_size
        if unique_size:
            self.evaluator_calls += 1
            self.inference_batch_sizes.append(unique_size)

        self.host_to_device_seconds += _finite_seconds(
            wave.host_to_device_seconds
        )
        self.model_inference_seconds += _finite_seconds(
            wave.model_inference_seconds
        )
        self.device_to_host_seconds += _finite_seconds(
            wave.device_to_host_seconds
        )
        self.inference_wait_seconds += _finite_seconds(
            wave.inference_wait_seconds
        )

        latest_value = (
            timing.get("latest_generation", {})
            if isinstance(timing, Mapping)
            else {}
        )
        latest = latest_value if isinstance(latest_value, Mapping) else {}
        self.native_select_seconds += _stage_seconds(latest, "select")
        self.native_deduplication_seconds += _stage_seconds(latest, "dedup")
        self.native_feature_encode_seconds += _stage_seconds(latest, "features")
        self.native_backup_seconds += _stage_seconds(latest, "backup")

        for stage_name in ("select", "dedup", "features", "backup"):
            stage_worker = _stage_worker(latest, stage_name)
            workers = _nonnegative_int(
                stage_worker.get("workers", parsed_worker_threads)
            )
            if workers:
                self.native_worker_threads = max(
                    self.native_worker_threads,
                    workers,
                )
            wall_seconds = _finite_seconds(stage_worker.get("wall_seconds", 0.0))
            busy_seconds = _finite_seconds(
                stage_worker.get("callback_busy_seconds", 0.0)
            )
            self.native_worker_busy_seconds += busy_seconds
            self.native_worker_capacity_seconds += wall_seconds * workers


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("Native telemetry counts must be integers")
    try:
        result = operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise TypeError("Native telemetry counts must be integers") from error
    if result < 0:
        raise ValueError("Native telemetry counts cannot be negative")
    return result


def _finite_seconds(value: object) -> float:
    try:
        result = float(cast(SupportsFloat, value))
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) and result >= 0.0 else 0.0


def _stage_seconds(timing: Mapping[str, Any], name: str) -> float:
    stage = timing.get(name, {})
    if not isinstance(stage, Mapping):
        return 0.0
    return _finite_seconds(stage.get("wall_seconds", 0.0))


def _stage_worker(timing: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    stage = timing.get(name, {})
    if not isinstance(stage, Mapping):
        return {}
    worker = stage.get("worker", {})
    return worker if isinstance(worker, Mapping) else {}


def evaluate_search_wave(
    sessions: Sequence[SearchSession],
    accumulator: BatchedSearchAccumulator,
    deduplicate_evaluations: bool = True,
) -> None:
    if not sessions:
        raise ValueError("At least one search session is required")
    if len(sessions) > accumulator.root_count:
        raise ValueError("Search wave exceeds its configured root count")
    evaluator = sessions[0].search.evaluator
    if any(session.search.evaluator is not evaluator for session in sessions):
        raise ValueError("Batched searches must share one evaluator")

    accumulator.simulation_waves += 1
    selections: dict[int, LeafSelection] = {}
    evaluation_groups: list[list[int]] = []
    group_by_state: dict[bytes, list[int]] = {}
    for index, session in enumerate(sessions):
        selection, selected_leaves = session.select_evaluation_leaf()
        accumulator.selected_leaves += selected_leaves
        if selection is None:
            continue
        selections[index] = selection
        if deduplicate_evaluations:
            group = group_by_state.get(selection.state_key)
            if group is None:
                group = []
                group_by_state[selection.state_key] = group
                evaluation_groups.append(group)
            group.append(index)
        else:
            evaluation_groups.append([index])
        accumulator.evaluation_requests += 1

    if not evaluation_groups:
        return

    representative_indices = [indices[0] for indices in evaluation_groups]
    positions = [selections[index].position for index in representative_indices]
    started = time.perf_counter()
    policies, values = evaluator.evaluate_batch(positions)
    elapsed = time.perf_counter() - started
    policies = np.asarray(policies)
    values = np.asarray(values).reshape(-1)
    action_size = sessions[0].search.game.get_action_size()
    if policies.shape != (len(positions), action_size):
        raise ValueError(f"Unexpected batched policy shape: {policies.shape}")
    if values.shape != (len(positions),):
        raise ValueError(f"Unexpected batched value shape: {values.shape}")

    batch_size = len(positions)
    batch_sizes = accumulator.inference_batch_sizes
    batch_sizes.append(batch_size)
    accumulator.evaluator_calls += 1
    accumulator.unique_evaluations += len(
        {selection.state_key for selection in selections.values()}
    )
    for group_offset, indices in enumerate(evaluation_groups):
        evaluation = (policies[group_offset], float(values[group_offset]))
        for index in indices:
            sessions[index].accept_evaluation(
                selections[index],
                evaluation,
                batch_size,
                elapsed,
            )


def run_batched_search(
    searches: Sequence[MCTS],
    roots: Sequence[PenteBoard],
    temperatures: Sequence[float],
    add_root_noise: bool,
    deduplicate_evaluations: bool = True,
) -> BatchedSearchResult:
    if not searches:
        raise ValueError("At least one search is required")
    if len(searches) != len(roots) or len(roots) != len(temperatures):
        raise ValueError("Searches, roots, and temperatures must have equal lengths")
    sessions = [
        SearchSession(search, root, temperature, add_root_noise)
        for search, root, temperature in zip(searches, roots, temperatures)
    ]
    accumulator = BatchedSearchAccumulator(len(sessions))
    while not all(session.is_complete for session in sessions):
        evaluate_search_wave(
            [session for session in sessions if not session.is_complete],
            accumulator,
            deduplicate_evaluations,
        )
    return BatchedSearchResult(
        [session.policy() for session in sessions],
        accumulator.telemetry(),
    )
