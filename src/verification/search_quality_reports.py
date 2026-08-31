"""Immutable report types for model-backed search verification."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Literal, SupportsIndex, cast

from src.evaluation.tactical import TacticalSuiteStats
from src.mcts.mcts_v2 import MCTSArgs
from src.train.self_play_args import SearchBackend


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    """A 95 percent Wilson confidence interval for a binomial proportion."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError("Wilson interval bounds must be finite")
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise ValueError("Wilson interval bounds must be between zero and one")


@dataclass(frozen=True, slots=True)
class SearchQualityConfig:
    """Immutable configuration captured by every backend verification report."""

    games: int
    opening_plies: int
    seed: int
    mcts_args: MCTSArgs
    native_worker_threads: int
    board_size: int
    ruleset: str

    def __post_init__(self) -> None:
        _require_positive_int(self.games, "games")
        if self.games < 2:
            raise ValueError("games must be at least two")
        if self.games % 2:
            raise ValueError("games must be even for paired openings")
        _require_nonnegative_int(self.opening_plies, "opening_plies")
        _require_nonnegative_int(self.seed, "seed")
        if not isinstance(self.mcts_args, MCTSArgs):
            raise TypeError("mcts_args must be an MCTSArgs instance")
        if self.mcts_args.num_simulations < 1:
            raise ValueError("mcts_args.num_simulations must be positive")
        _require_positive_int(self.native_worker_threads, "native_worker_threads")
        _require_positive_int(self.board_size, "board_size")
        if not isinstance(self.ruleset, str) or not self.ruleset:
            raise ValueError("ruleset must be a non-empty string")


@dataclass(frozen=True, slots=True)
class GameResults:
    """Identity-level game outcomes and their Wilson interval."""

    games: int
    current_wins: int
    opponent_wins: int
    draws: int
    decisive_games: int
    current_decisive_win_rate: float
    current_decisive_win_rate_95pct: WilsonInterval
    current_score: float
    current_elo: float


@dataclass(frozen=True, slots=True)
class ColorResults:
    """Results split by the Pente color assigned to the current model."""

    current_as_player_one_wins: int
    current_as_player_two_wins: int
    player_one_color_wins: int
    player_two_color_wins: int


@dataclass(frozen=True, slots=True)
class PairResults:
    """Color-swapped opening-pair outcomes and their Wilson interval."""

    opening_plies: int
    unique_openings: int
    paired_openings: int
    current_pair_wins: int
    current_pair_losses: int
    pair_ties: int
    current_paired_win_rate: float
    current_paired_win_rate_95pct: WilsonInterval


@dataclass(frozen=True, slots=True)
class MatchReport:
    """Stable report for one current-model-versus-opponent arena."""

    opponent: Literal["previous", "random", "heuristic"]
    game: GameResults
    color: ColorResults
    pair: PairResults
    average_moves: float
    elapsed_seconds: float
    moves_per_second: float


@dataclass(frozen=True, slots=True)
class DirectTacticalMetrics:
    """Tactical-suite metrics from direct network inference."""

    cases: int
    correct: int
    accuracy: float
    mean_expected_policy_mass: float
    category_accuracy: tuple[tuple[str, float], ...]

    @staticmethod
    def from_stats(stats: TacticalSuiteStats) -> DirectTacticalMetrics:
        return DirectTacticalMetrics(
            cases=stats.cases,
            correct=stats.correct,
            accuracy=stats.accuracy,
            mean_expected_policy_mass=stats.mean_expected_policy_mass,
            category_accuracy=tuple(sorted(stats.category_accuracy.items())),
        )


@dataclass(frozen=True, slots=True)
class SearchTacticalCaseResult:
    """One learned tactical position evaluated by a search player."""

    name: str
    category: str
    selected_action: int
    expected_actions: tuple[int, ...]
    correct: bool


@dataclass(frozen=True, slots=True)
class SearchTacticalReport:
    """Search-player tactical actions and aggregate correctness."""

    cases: tuple[SearchTacticalCaseResult, ...]
    correct: int
    accuracy: float


@dataclass(frozen=True, slots=True)
class BackendSearchQualityReport:
    """Complete bounded verification result for one search backend."""

    backend: SearchBackend
    config: SearchQualityConfig
    current_vs_previous: MatchReport
    current_vs_random: MatchReport
    current_vs_heuristic: MatchReport
    current_direct_tactical: DirectTacticalMetrics
    previous_direct_tactical: DirectTacticalMetrics
    search_tactical: SearchTacticalReport
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class StatisticalParityCriteria:
    """Tolerance policy for comparing independent backend measurements.

    Parity compares rates, Wilson intervals, and search tactical accuracy. It
    does not compare trajectories, action sequences, raw game counts, or
    direct-network metrics that are identical inputs for both backends.
    """

    maximum_decisive_win_rate_difference: float = 0.10
    maximum_paired_win_rate_difference: float = 0.10
    maximum_tactical_accuracy_difference: float = 1.0 / 6.0
    require_wilson_overlap: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_decisive_win_rate_difference",
            "maximum_paired_win_rate_difference",
            "maximum_tactical_accuracy_difference",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class MatchParity:
    """Rate-level parity result for one opponent comparison."""

    opponent: Literal["previous", "random", "heuristic"]
    decisive_win_rate_difference: float
    paired_win_rate_difference: float
    decisive_interval_overlaps: bool
    paired_interval_overlaps: bool


@dataclass(frozen=True, slots=True)
class StatisticalParityReport:
    """Explicit statistical comparison between two backend reports."""

    left_backend: SearchBackend
    right_backend: SearchBackend
    matches: tuple[MatchParity, ...]
    tactical_accuracy_difference: float
    passed: bool
    failures: tuple[str, ...]


def _require_positive_int(value: object, name: str) -> int:
    parsed = _require_nonnegative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = int(operator.index(cast(SupportsIndex, value)))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < 0:
        raise ValueError(f"{name} cannot be negative")
    return parsed


__all__ = [
    "BackendSearchQualityReport",
    "ColorResults",
    "DirectTacticalMetrics",
    "GameResults",
    "MatchParity",
    "MatchReport",
    "PairResults",
    "SearchQualityConfig",
    "SearchTacticalCaseResult",
    "SearchTacticalReport",
    "StatisticalParityCriteria",
    "StatisticalParityReport",
    "WilsonInterval",
]
