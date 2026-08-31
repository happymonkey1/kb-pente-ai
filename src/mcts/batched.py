from __future__ import annotations

from dataclasses import dataclass
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


def run_batched_search(
    searches: Sequence[MCTS],
    roots: Sequence[PenteBoard],
    temperatures: Sequence[float],
    add_root_noise: bool,
) -> BatchedSearchResult:
    if not searches:
        raise ValueError("At least one search is required")
    if len(searches) != len(roots) or len(roots) != len(temperatures):
        raise ValueError("Searches, roots, and temperatures must have equal lengths")
    evaluator = searches[0].evaluator
    if any(search.evaluator is not evaluator for search in searches):
        raise ValueError("Batched searches must share one evaluator")
    for search, root in zip(searches, roots):
        if search.game.check_game_end(root).is_terminal:
            raise ValueError("Cannot search a terminal root")

    root_keys = [search.game.to_string(root) for search, root in zip(searches, roots)]
    root_priors = [search.root_priors(root, add_root_noise) for search, root in zip(searches, roots)]
    completed_simulations = [0] * len(searches)
    selected_leaves = 0
    evaluation_requests = 0
    unique_evaluations = 0
    batch_sizes: list[int] = []
    evaluator_calls = 0

    simulation_waves = 0
    while any(
        completed < search.args.num_simulations
        for completed, search in zip(completed_simulations, searches)
    ):
        simulation_waves += 1
        active_indices = [
            index
            for index, search in enumerate(searches)
            if completed_simulations[index] < search.args.num_simulations
        ]
        selections: dict[int, LeafSelection] = {}
        evaluation_groups: dict[bytes, list[int]] = {}

        for index in active_indices:
            while completed_simulations[index] < searches[index].args.num_simulations:
                selection = searches[index].select_leaf(
                    roots[index],
                    root_keys[index],
                    root_priors[index],
                )
                selected_leaves += 1
                if selection.terminal_result.is_terminal:
                    searches[index].expand_and_backup(selection)
                    searches[index].record_simulation()
                    completed_simulations[index] += 1
                    continue

                selections[index] = selection
                state_key = searches[index].game.to_string(selection.position)
                evaluation_groups.setdefault(state_key, []).append(index)
                evaluation_requests += 1
                break

        evaluations: dict[int, tuple[np.ndarray, float]] = {}
        if evaluation_groups:
            representative_indices = [indices[0] for indices in evaluation_groups.values()]
            positions = [selections[index].position for index in representative_indices]
            started = time.perf_counter()
            policies, values = evaluator.evaluate_batch(positions)
            elapsed = time.perf_counter() - started
            policies = np.asarray(policies)
            values = np.asarray(values).reshape(-1)
            if policies.shape != (len(positions), searches[0].game.get_action_size()):
                raise ValueError(f"Unexpected batched policy shape: {policies.shape}")
            if values.shape != (len(positions),):
                raise ValueError(f"Unexpected batched value shape: {values.shape}")

            batch_size = len(positions)
            batch_sizes.append(batch_size)
            evaluator_calls += 1
            unique_evaluations += batch_size
            for group_offset, indices in enumerate(evaluation_groups.values()):
                evaluation = (policies[group_offset], float(values[group_offset]))
                for index in indices:
                    evaluations[index] = evaluation
                    searches[index].record_batch_evaluation(batch_size, elapsed)

        for index, selection in selections.items():
            searches[index].expand_and_backup(selection, evaluations[index])
            searches[index].record_simulation()
            completed_simulations[index] += 1
            if root_priors[index] is None:
                root_priors[index] = searches[index].root_priors(roots[index], add_root_noise)

    duplicate_rate = (
        1.0 - unique_evaluations / evaluation_requests
        if evaluation_requests
        else 0.0
    )
    telemetry = BatchedSearchTelemetry(
        root_count=len(roots),
        simulation_waves=simulation_waves,
        selected_leaves=selected_leaves,
        evaluation_requests=evaluation_requests,
        unique_evaluations=unique_evaluations,
        evaluator_calls=evaluator_calls,
        inference_batch_sizes=tuple(batch_sizes),
        min_inference_batch_size=min(batch_sizes, default=0),
        mean_inference_batch_size=float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        median_inference_batch_size=float(np.median(batch_sizes)) if batch_sizes else 0.0,
        p95_inference_batch_size=float(np.percentile(batch_sizes, 95)) if batch_sizes else 0.0,
        max_inference_batch_size=max(batch_sizes, default=0),
        duplicate_leaf_rate=duplicate_rate,
    )
    output_policies = [
        search.action_prob_from_counts(root, temperature)
        for search, root, temperature in zip(searches, roots, temperatures)
    ]
    return BatchedSearchResult(output_policies, telemetry)
