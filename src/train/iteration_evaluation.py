from __future__ import annotations

import numpy as np

from src.evaluation.statistics import elo_difference, wilson_interval
from src.evaluation.tactical import evaluate_tactical_suite
from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTSArgs
from src.model.model_v1 import PenteNet
from src.telemetry import MetricSink
from src.train.arena import Arena
from src.train.player_builder import build_player
from src.train.random_player import RandomPlayer
from src.train.self_play_args import SearchBackend
from src.train.tactical_heuristic_player import TacticalHeuristicPlayer


def evaluate_training_iteration(
    game: PenteGame,
    previous_net: PenteNet,
    current_net: PenteNet,
    mcts_args: MCTSArgs,
    metric_sink: MetricSink,
    iteration: int,
    num_games: int,
    opening_plies: int,
    debug: bool,
    seed: int,
    search_backend: SearchBackend = "python",
    native_worker_threads: int = 1,
) -> None:
    previous_net.eval()
    current_net.eval()
    arena = Arena(
        player1=build_player(
            previous_net,
            None,
            "previous",
            search_backend=search_backend,
            game=game,
            mcts_args=mcts_args,
            seed=seed,
            native_worker_threads=native_worker_threads,
        ),
        player2=build_player(
            current_net,
            None,
            "current",
            search_backend=search_backend,
            game=game,
            mcts_args=mcts_args,
            seed=seed + 1,
            native_worker_threads=native_worker_threads,
        ),
        game=game,
        debug=debug,
        opening_plies=opening_plies,
        rng=np.random.default_rng(seed + iteration + 10),
    )
    stats = arena.play_games(num_games)
    decisive = stats.p1_wins + stats.p2_wins
    current_win_rate = stats.p2_wins / decisive if decisive else 0.0
    current_lower, current_upper = wilson_interval(stats.p2_wins, decisive)
    current_pair_wins = stats.p1_pair_losses
    current_pair_losses = stats.p1_pair_wins
    current_decisive_pairs = current_pair_wins + current_pair_losses
    current_pair_lower, current_pair_upper = wilson_interval(
        current_pair_wins,
        current_decisive_pairs,
    )
    current_score = (stats.p2_wins + 0.5 * stats.draws) / num_games

    random_player = build_player(
        current_net,
        None,
        "current",
        search_backend=search_backend,
        game=game,
        mcts_args=mcts_args,
        seed=seed + 2,
        native_worker_threads=native_worker_threads,
    )
    random_arena = Arena(
        player1=random_player,
        player2=RandomPlayer(np.random.default_rng(seed + iteration + 3)),
        game=game,
        debug=debug,
        opening_plies=opening_plies,
        rng=np.random.default_rng(seed + iteration + 11),
    )
    random_stats = random_arena.play_games(num_games)
    random_decisive = random_stats.p1_wins + random_stats.p2_wins
    random_win_rate = random_stats.p1_wins / random_decisive if random_decisive else 0.0
    random_lower, random_upper = wilson_interval(random_stats.p1_wins, random_decisive)
    random_decisive_pairs = random_stats.p1_pair_wins + random_stats.p1_pair_losses
    random_pair_lower, random_pair_upper = wilson_interval(
        random_stats.p1_pair_wins,
        random_decisive_pairs,
    )
    random_score = (random_stats.p1_wins + 0.5 * random_stats.draws) / num_games

    heuristic_player = build_player(
        current_net,
        None,
        "current",
        search_backend=search_backend,
        game=game,
        mcts_args=mcts_args,
        seed=seed + 4,
        native_worker_threads=native_worker_threads,
    )
    heuristic_arena = Arena(
        player1=heuristic_player,
        player2=TacticalHeuristicPlayer(),
        game=game,
        debug=debug,
        opening_plies=opening_plies,
        rng=np.random.default_rng(seed + iteration + 12),
    )
    heuristic_stats = heuristic_arena.play_games(num_games)
    heuristic_decisive = heuristic_stats.p1_wins + heuristic_stats.p2_wins
    heuristic_win_rate = (
        heuristic_stats.p1_wins / heuristic_decisive if heuristic_decisive else 0.0
    )
    heuristic_lower, heuristic_upper = wilson_interval(
        heuristic_stats.p1_wins,
        heuristic_decisive,
    )
    heuristic_decisive_pairs = (
        heuristic_stats.p1_pair_wins + heuristic_stats.p1_pair_losses
    )
    heuristic_pair_lower, heuristic_pair_upper = wilson_interval(
        heuristic_stats.p1_pair_wins,
        heuristic_decisive_pairs,
    )

    tactical_metrics: dict[str, float] = {}
    if game.get_board_size() >= 9:
        tactical = evaluate_tactical_suite(current_net, game)
        tactical_metrics = {
            "tactical_accuracy": tactical.accuracy,
            "tactical_expected_policy_mass": tactical.mean_expected_policy_mass,
        }

    metric_sink.emit(
        "arena",
        iteration,
        {
            "previous_wins": stats.p1_wins,
            "current_wins": stats.p2_wins,
            "draws": stats.draws,
            "current_decisive_win_rate": current_win_rate,
            "current_win_rate_95pct_lower": current_lower,
            "current_win_rate_95pct_upper": current_upper,
            "current_pair_wins": current_pair_wins,
            "current_pair_losses": current_pair_losses,
            "current_pair_ties": stats.pair_ties,
            "current_paired_win_rate_95pct_lower": current_pair_lower,
            "current_paired_win_rate_95pct_upper": current_pair_upper,
            "current_vs_previous_elo": elo_difference(current_score),
            "average_moves": stats.avg_moves,
            "arena_player_one_color_wins": stats.player_one_color_wins,
            "arena_player_two_color_wins": stats.player_two_color_wins,
            "arena_opening_plies": stats.opening_plies,
            "arena_unique_openings": stats.unique_openings,
            "current_as_player_one_wins": stats.p2_as_player_one_wins,
            "current_as_player_two_wins": stats.p2_as_player_two_wins,
            "random_wins": random_stats.p2_wins,
            "current_vs_random_wins": random_stats.p1_wins,
            "current_vs_random_draws": random_stats.draws,
            "current_vs_random_decisive_win_rate": random_win_rate,
            "current_vs_random_win_rate_95pct_lower": random_lower,
            "current_vs_random_win_rate_95pct_upper": random_upper,
            "current_vs_random_pair_wins": random_stats.p1_pair_wins,
            "current_vs_random_pair_losses": random_stats.p1_pair_losses,
            "current_vs_random_pair_ties": random_stats.pair_ties,
            "current_vs_random_paired_win_rate_95pct_lower": random_pair_lower,
            "current_vs_random_paired_win_rate_95pct_upper": random_pair_upper,
            "current_vs_random_elo": elo_difference(random_score),
            "random_arena_player_one_color_wins": random_stats.player_one_color_wins,
            "random_arena_player_two_color_wins": random_stats.player_two_color_wins,
            "random_arena_unique_openings": random_stats.unique_openings,
            "current_vs_random_as_player_one_wins": random_stats.p1_as_player_one_wins,
            "current_vs_random_as_player_two_wins": random_stats.p1_as_player_two_wins,
            "heuristic_wins": heuristic_stats.p2_wins,
            "current_vs_heuristic_wins": heuristic_stats.p1_wins,
            "current_vs_heuristic_draws": heuristic_stats.draws,
            "current_vs_heuristic_decisive_win_rate": heuristic_win_rate,
            "current_vs_heuristic_win_rate_95pct_lower": heuristic_lower,
            "current_vs_heuristic_win_rate_95pct_upper": heuristic_upper,
            "current_vs_heuristic_pair_wins": heuristic_stats.p1_pair_wins,
            "current_vs_heuristic_pair_losses": heuristic_stats.p1_pair_losses,
            "current_vs_heuristic_pair_ties": heuristic_stats.pair_ties,
            "current_vs_heuristic_paired_win_rate_95pct_lower": heuristic_pair_lower,
            "current_vs_heuristic_paired_win_rate_95pct_upper": heuristic_pair_upper,
            "heuristic_arena_unique_openings": heuristic_stats.unique_openings,
            "current_vs_heuristic_as_player_one_wins": (
                heuristic_stats.p1_as_player_one_wins
            ),
            "current_vs_heuristic_as_player_two_wins": (
                heuristic_stats.p1_as_player_two_wins
            ),
            **tactical_metrics,
        },
    )
