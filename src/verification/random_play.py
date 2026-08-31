from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from src.evaluation.statistics import elo_difference, wilson_interval
from src.game.pente.pente_game import PenteGame
from src.model.model_v1 import PenteNet
from src.train.arena import Arena, ArenaStats
from src.train.nnet_player import NNetPlayer
from src.train.random_player import RandomPlayer


@dataclass(frozen=True, slots=True)
class RandomPlayCriteria:
    minimum_games: int = 100
    minimum_decisive_win_rate_95pct_lower: float = 0.5
    require_wins_as_both_colors: bool = True

    def __post_init__(self) -> None:
        if self.minimum_games < 2:
            raise ValueError("Random-play verification requires at least two games")
        if not 0 <= self.minimum_decisive_win_rate_95pct_lower <= 1:
            raise ValueError("Minimum confidence bound must be between zero and one")


@dataclass(frozen=True, slots=True)
class RandomPlayReport:
    games: int
    model_wins: int
    random_wins: int
    draws: int
    decisive_games: int
    model_decisive_win_rate: float
    model_decisive_win_rate_95pct_lower: float
    model_decisive_win_rate_95pct_upper: float
    model_score: float
    model_elo: float
    model_as_player_one_wins: int
    model_as_player_two_wins: int
    player_one_color_wins: int
    player_two_color_wins: int
    opening_plies: int
    unique_openings: int
    paired_openings: int
    model_pair_wins: int
    model_pair_losses: int
    pair_ties: int
    average_moves: float
    elapsed_seconds: float
    moves_per_second: float
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_network_against_random(
    model: PenteNet,
    game: PenteGame,
    games: int,
    seed: int,
    opening_plies: int = 4,
    criteria: RandomPlayCriteria | None = None,
) -> RandomPlayReport:
    model.eval()
    arena = Arena(
        NNetPlayer(model, None, "network"),
        RandomPlayer(np.random.default_rng(seed)),
        game,
        opening_plies=opening_plies,
        rng=np.random.default_rng(seed + 1),
    )
    started = time.perf_counter()
    stats = arena.play_games(games)
    elapsed = time.perf_counter() - started
    return summarize_random_play(stats, elapsed, criteria)


def summarize_random_play(
    stats: ArenaStats,
    elapsed_seconds: float,
    criteria: RandomPlayCriteria | None = None,
) -> RandomPlayReport:
    if elapsed_seconds < 0:
        raise ValueError("Elapsed time cannot be negative")
    selected = criteria or RandomPlayCriteria()
    games = stats.p1_wins + stats.p2_wins + stats.draws
    decisive_games = stats.p1_wins + stats.p2_wins
    decisive_win_rate = stats.p1_wins / decisive_games if decisive_games else 0.0
    lower, upper = wilson_interval(stats.p1_wins, decisive_games)
    score = (stats.p1_wins + 0.5 * stats.draws) / games if games else 0.0
    failures = []
    if games < selected.minimum_games:
        failures.append("evaluation contains fewer games than required")
    if lower <= selected.minimum_decisive_win_rate_95pct_lower:
        failures.append("decisive win-rate confidence bound does not exceed threshold")
    if selected.require_wins_as_both_colors:
        if stats.p1_as_player_one_wins == 0:
            failures.append("model did not win as Player 1")
        if stats.p1_as_player_two_wins == 0:
            failures.append("model did not win as Player 2")

    return RandomPlayReport(
        games=games,
        model_wins=stats.p1_wins,
        random_wins=stats.p2_wins,
        draws=stats.draws,
        decisive_games=decisive_games,
        model_decisive_win_rate=decisive_win_rate,
        model_decisive_win_rate_95pct_lower=lower,
        model_decisive_win_rate_95pct_upper=upper,
        model_score=score,
        model_elo=elo_difference(score),
        model_as_player_one_wins=stats.p1_as_player_one_wins,
        model_as_player_two_wins=stats.p1_as_player_two_wins,
        player_one_color_wins=stats.player_one_color_wins,
        player_two_color_wins=stats.player_two_color_wins,
        opening_plies=stats.opening_plies,
        unique_openings=stats.unique_openings,
        paired_openings=stats.paired_openings,
        model_pair_wins=stats.p1_pair_wins,
        model_pair_losses=stats.p1_pair_losses,
        pair_ties=stats.pair_ties,
        average_moves=stats.avg_moves,
        elapsed_seconds=elapsed_seconds,
        moves_per_second=(
            games * stats.avg_moves / elapsed_seconds
            if elapsed_seconds
            else 0.0
        ),
        passed=not failures,
        failures=tuple(failures),
    )
