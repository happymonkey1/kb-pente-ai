from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Sequence

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
        )


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
