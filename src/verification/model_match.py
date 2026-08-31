from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from src.evaluation.statistics import elo_difference, wilson_interval
from src.evaluation.tactical import TacticalSuiteStats, evaluate_tactical_suite
from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTS, MCTSArgs
from src.model.model_v1 import PenteNet
from src.train.arena import Arena, ArenaStats
from src.train.nnet_player import NNetPlayer


@dataclass(frozen=True, slots=True)
class ModelMatchCriteria:
    minimum_games: int = 100
    minimum_paired_win_rate_95pct_lower: float = 0.5
    require_wins_as_both_colors: bool = True
    require_tactical_non_regression: bool = True

    def __post_init__(self) -> None:
        if self.minimum_games < 2:
            raise ValueError("Model match requires at least two games")
        if not 0 <= self.minimum_paired_win_rate_95pct_lower <= 1:
            raise ValueError("Minimum confidence bound must be between zero and one")


@dataclass(frozen=True, slots=True)
class ModelMatchReport:
    games: int
    candidate_wins: int
    baseline_wins: int
    draws: int
    decisive_games: int
    candidate_decisive_win_rate: float
    candidate_decisive_win_rate_95pct_lower: float
    candidate_decisive_win_rate_95pct_upper: float
    candidate_score: float
    candidate_elo: float
    candidate_as_player_one_wins: int
    candidate_as_player_two_wins: int
    player_one_color_wins: int
    player_two_color_wins: int
    opening_plies: int
    unique_openings: int
    paired_openings: int
    candidate_pair_wins: int
    candidate_pair_losses: int
    pair_ties: int
    candidate_paired_win_rate: float
    candidate_paired_win_rate_95pct_lower: float
    candidate_paired_win_rate_95pct_upper: float
    average_moves: float
    elapsed_seconds: float
    moves_per_second: float
    candidate_tactical_accuracy: float
    baseline_tactical_accuracy: float
    candidate_tactical_policy_mass: float
    baseline_tactical_policy_mass: float
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_model_match(
    candidate: PenteNet,
    baseline: PenteNet,
    game: PenteGame,
    games: int,
    seed: int,
    simulations: int,
    opening_plies: int = 4,
    criteria: ModelMatchCriteria | None = None,
) -> ModelMatchReport:
    if simulations < 0:
        raise ValueError("Simulation count cannot be negative")
    candidate.eval()
    baseline.eval()
    mcts_args = MCTSArgs(num_simulations=simulations) if simulations else None
    candidate_mcts = (
        MCTS(game, candidate, mcts_args, np.random.default_rng(seed + 1))
        if mcts_args is not None
        else None
    )
    baseline_mcts = (
        MCTS(game, baseline, mcts_args, np.random.default_rng(seed + 2))
        if mcts_args is not None
        else None
    )
    arena = Arena(
        NNetPlayer(candidate, candidate_mcts, "candidate"),
        NNetPlayer(baseline, baseline_mcts, "baseline"),
        game,
        opening_plies=opening_plies,
        rng=np.random.default_rng(seed),
    )
    started = time.perf_counter()
    stats = arena.play_games(games)
    elapsed = time.perf_counter() - started
    candidate_tactical = evaluate_tactical_suite(candidate, game)
    baseline_tactical = evaluate_tactical_suite(baseline, game)
    return summarize_model_match(
        stats,
        elapsed,
        candidate_tactical,
        baseline_tactical,
        criteria,
    )


def summarize_model_match(
    stats: ArenaStats,
    elapsed_seconds: float,
    candidate_tactical: TacticalSuiteStats,
    baseline_tactical: TacticalSuiteStats,
    criteria: ModelMatchCriteria | None = None,
) -> ModelMatchReport:
    if elapsed_seconds < 0:
        raise ValueError("Elapsed time cannot be negative")
    selected = criteria or ModelMatchCriteria()
    games = stats.p1_wins + stats.p2_wins + stats.draws
    decisive_games = stats.p1_wins + stats.p2_wins
    decisive_win_rate = stats.p1_wins / decisive_games if decisive_games else 0.0
    lower, upper = wilson_interval(stats.p1_wins, decisive_games)
    decisive_pairs = stats.p1_pair_wins + stats.p1_pair_losses
    paired_win_rate = (
        stats.p1_pair_wins / decisive_pairs
        if decisive_pairs
        else 0.0
    )
    paired_lower, paired_upper = wilson_interval(
        stats.p1_pair_wins,
        decisive_pairs,
    )
    score = (stats.p1_wins + 0.5 * stats.draws) / games if games else 0.0
    failures = []
    if games < selected.minimum_games:
        failures.append("evaluation contains fewer games than required")
    if paired_lower <= selected.minimum_paired_win_rate_95pct_lower:
        failures.append("candidate paired confidence bound does not exceed threshold")
    if selected.require_wins_as_both_colors:
        if stats.p1_as_player_one_wins == 0:
            failures.append("candidate did not win as Player 1")
        if stats.p1_as_player_two_wins == 0:
            failures.append("candidate did not win as Player 2")
    if selected.require_tactical_non_regression:
        if candidate_tactical.accuracy < baseline_tactical.accuracy:
            failures.append("candidate tactical accuracy regressed")
        if (
            candidate_tactical.mean_expected_policy_mass
            < baseline_tactical.mean_expected_policy_mass
        ):
            failures.append("candidate tactical policy mass regressed")

    return ModelMatchReport(
        games=games,
        candidate_wins=stats.p1_wins,
        baseline_wins=stats.p2_wins,
        draws=stats.draws,
        decisive_games=decisive_games,
        candidate_decisive_win_rate=decisive_win_rate,
        candidate_decisive_win_rate_95pct_lower=lower,
        candidate_decisive_win_rate_95pct_upper=upper,
        candidate_score=score,
        candidate_elo=elo_difference(score),
        candidate_as_player_one_wins=stats.p1_as_player_one_wins,
        candidate_as_player_two_wins=stats.p1_as_player_two_wins,
        player_one_color_wins=stats.player_one_color_wins,
        player_two_color_wins=stats.player_two_color_wins,
        opening_plies=stats.opening_plies,
        unique_openings=stats.unique_openings,
        paired_openings=stats.paired_openings,
        candidate_pair_wins=stats.p1_pair_wins,
        candidate_pair_losses=stats.p1_pair_losses,
        pair_ties=stats.pair_ties,
        candidate_paired_win_rate=paired_win_rate,
        candidate_paired_win_rate_95pct_lower=paired_lower,
        candidate_paired_win_rate_95pct_upper=paired_upper,
        average_moves=stats.avg_moves,
        elapsed_seconds=elapsed_seconds,
        moves_per_second=(
            games * stats.avg_moves / elapsed_seconds
            if elapsed_seconds
            else 0.0
        ),
        candidate_tactical_accuracy=candidate_tactical.accuracy,
        baseline_tactical_accuracy=baseline_tactical.accuracy,
        candidate_tactical_policy_mass=candidate_tactical.mean_expected_policy_mass,
        baseline_tactical_policy_mass=baseline_tactical.mean_expected_policy_mass,
        passed=not failures,
        failures=tuple(failures),
    )
