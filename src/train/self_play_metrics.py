from __future__ import annotations

import numpy as np

from src.game.game import Game
from src.mcts.batched import BatchedSearchTelemetry
from src.train.self_play_generation import PlayedGame


def collect_self_play_metrics(
    games: list[PlayedGame],
    batches: list[BatchedSearchTelemetry],
    elapsed_seconds: float,
) -> dict[str, int | float]:
    trajectories = {game.actions for game in games}
    positions = sum(len(game.examples) for game in games)
    p1_wins = sum(game.winner == Game.PLAYER_ONE for game in games)
    p2_wins = sum(game.winner == Game.PLAYER_TWO for game in games)
    draws = sum(game.winner is None for game in games)
    p1_capture_wins = sum(
        game.winner == Game.PLAYER_ONE and game.win_reason == "capture"
        for game in games
    )
    p2_capture_wins = sum(
        game.winner == Game.PLAYER_TWO and game.win_reason == "capture"
        for game in games
    )
    roots = [telemetry for game in games for telemetry in game.root_telemetry]
    final_searches = [game.root_telemetry[-1] for game in games]
    unique_evaluations = sum(batch.unique_evaluations for batch in batches)
    evaluation_requests = sum(batch.evaluation_requests for batch in batches)
    inference_batch_sizes = [
        batch_size
        for batch in batches
        for batch_size in batch.inference_batch_sizes
    ]
    active_game_target = max((batch.root_count for batch in batches), default=0)
    steady_state_batch_sizes = [
        batch_size
        for batch in batches
        if batch.root_count == active_game_target
        for batch_size in batch.inference_batch_sizes
    ]
    native_select_seconds = sum(
        batch.native_select_seconds for batch in batches
    )
    native_deduplication_seconds = sum(
        batch.native_deduplication_seconds for batch in batches
    )
    native_feature_encode_seconds = sum(
        batch.native_feature_encode_seconds for batch in batches
    )
    native_backup_seconds = sum(
        batch.native_backup_seconds for batch in batches
    )
    model_inference_seconds = sum(
        batch.model_inference_seconds for batch in batches
    )
    host_to_device_seconds = sum(
        batch.host_to_device_seconds for batch in batches
    )
    device_to_host_seconds = sum(
        batch.device_to_host_seconds for batch in batches
    )
    inference_wait_seconds = sum(
        batch.inference_wait_seconds for batch in batches
    )
    native_worker_busy_seconds = sum(
        batch.native_worker_busy_seconds for batch in batches
    )
    native_worker_capacity_seconds = sum(
        batch.native_worker_capacity_seconds for batch in batches
    )
    native_worker_busy_percent = (
        100.0
        * native_worker_busy_seconds
        / native_worker_capacity_seconds
        if native_worker_capacity_seconds > 0.0
        else 0.0
    )
    native_worker_threads = max(
        (batch.native_worker_threads for batch in batches),
        default=0,
    )
    collapse_eligible_roots = sum(root.root_collapse_eligible for root in roots)
    collapsed_roots = sum(root.root_search_collapsed for root in roots)
    return {
        "games": len(games),
        "positions_per_second": positions / elapsed_seconds if elapsed_seconds else 0.0,
        "games_per_second": len(games) / elapsed_seconds if elapsed_seconds else 0.0,
        "unique_trajectories": len(trajectories),
        "unique_trajectory_rate": len(trajectories) / len(games),
        "player_one_wins": p1_wins,
        "player_two_wins": p2_wins,
        "draws": draws,
        "player_one_capture_wins": p1_capture_wins,
        "player_two_capture_wins": p2_capture_wins,
        "mean_root_children_visited": float(
            np.mean([root.root_children_visited for root in roots])
        ),
        "mean_root_visit_entropy": float(
            np.mean([root.root_visit_entropy for root in roots])
        ),
        "mean_root_max_visit_share": float(
            np.mean([root.root_max_visit_share for root in roots])
        ),
        "search_collapse_eligible_roots": collapse_eligible_roots,
        "search_collapsed_roots": collapsed_roots,
        "search_collapse_rate": (
            collapsed_roots / collapse_eligible_roots
            if collapse_eligible_roots
            else 0.0
        ),
        "search_collapse_detected": int(collapsed_roots > 0),
        "mcts_invalid_policy_fallbacks": sum(
            search.invalid_policy_fallbacks for search in final_searches
        ),
        "mcts_zero_visit_fallbacks": sum(
            search.zero_visit_fallbacks for search in final_searches
        ),
        "mcts_max_depth": max(search.max_depth for search in final_searches),
        "leaf_evaluations_per_second": (
            unique_evaluations / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "min_inference_batch_size": min(inference_batch_sizes, default=0),
        "mean_inference_batch_size": (
            float(np.mean(inference_batch_sizes)) if inference_batch_sizes else 0.0
        ),
        "median_inference_batch_size": (
            float(np.median(inference_batch_sizes)) if inference_batch_sizes else 0.0
        ),
        "p95_inference_batch_size": (
            float(np.percentile(inference_batch_sizes, 95))
            if inference_batch_sizes
            else 0.0
        ),
        "max_inference_batch_size": max(inference_batch_sizes, default=0),
        "active_game_target": active_game_target,
        "steady_state_inference_batches": len(steady_state_batch_sizes),
        "steady_state_mean_inference_batch_size": (
            float(np.mean(steady_state_batch_sizes))
            if steady_state_batch_sizes
            else 0.0
        ),
        "steady_state_mean_batch_occupancy": (
            float(np.mean(steady_state_batch_sizes)) / active_game_target
            if steady_state_batch_sizes and active_game_target
            else 0.0
        ),
        "duplicate_leaf_rate": (
            1.0 - unique_evaluations / evaluation_requests
            if evaluation_requests
            else 0.0
        ),
        "mcts_select_seconds": native_select_seconds,
        "mcts_dedup_seconds": native_deduplication_seconds,
        "mcts_feature_encode_seconds": native_feature_encode_seconds,
        "mcts_backup_seconds": native_backup_seconds,
        "model_inference_seconds": model_inference_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "device_to_host_seconds": device_to_host_seconds,
        "inference_wait_seconds": inference_wait_seconds,
        "native_worker_threads": native_worker_threads,
        "native_worker_busy_percent": native_worker_busy_percent,
        "native_worker_busy_seconds": native_worker_busy_seconds,
        "native_worker_capacity_seconds": native_worker_capacity_seconds,
    }
