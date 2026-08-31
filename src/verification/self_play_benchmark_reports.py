"""Immutable report types for the full self-play benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

from src.game.pente.rules import PenteRuleset
from src.monitoring.cpu_metrics import CpuMetrics
from src.monitoring.cuda_metrics import CudaMetrics
from src.train.self_play_args import SearchBackend


MetricValue: TypeAlias = int | float
MetricValues: TypeAlias = tuple[tuple[str, MetricValue], ...]
MedianMetricValues: TypeAlias = tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class SelfPlayBenchmarkConfig:
    """Configuration shared by every measured Python and native run."""

    board_size: int = 19
    ruleset: PenteRuleset = PenteRuleset.STANDARD
    games: int = 16
    max_active_games: int = 16
    simulations: int = 16
    temp_threshold: int = 3
    repeats: int = 3
    warmup_batches: int = 2
    seed: int = 37
    native_worker_threads: int = 1
    torch_threads: int | None = None
    minimum_steady_state_batch_occupancy: float = 0.8
    minimum_native_games_per_second_ratio: float = 2.0
    maximum_invalid_policy_fallbacks: int = 0
    maximum_zero_visit_fallbacks: int = 0
    maximum_cpu_sampling_errors: int = 0
    maximum_cuda_sampling_errors: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.board_size, "board_size")
        if self.board_size < 5:
            raise ValueError("board_size must be at least five")
        if not isinstance(self.ruleset, PenteRuleset):
            raise TypeError("ruleset must be a PenteRuleset")
        _positive_int(self.games, "games")
        _positive_int(self.max_active_games, "max_active_games")
        _positive_int(self.simulations, "simulations")
        _nonnegative_int(self.temp_threshold, "temp_threshold")
        _positive_int(self.repeats, "repeats")
        _nonnegative_int(self.warmup_batches, "warmup_batches")
        _nonnegative_int(self.seed, "seed")
        _positive_int(self.native_worker_threads, "native_worker_threads")
        if self.torch_threads is not None:
            _positive_int(self.torch_threads, "torch_threads")
        _unit_interval(
            self.minimum_steady_state_batch_occupancy,
            "minimum_steady_state_batch_occupancy",
        )
        _finite_nonnegative(
            self.minimum_native_games_per_second_ratio,
            "minimum_native_games_per_second_ratio",
        )
        _nonnegative_int(
            self.maximum_invalid_policy_fallbacks,
            "maximum_invalid_policy_fallbacks",
        )
        _nonnegative_int(
            self.maximum_zero_visit_fallbacks,
            "maximum_zero_visit_fallbacks",
        )
        _nonnegative_int(
            self.maximum_cpu_sampling_errors,
            "maximum_cpu_sampling_errors",
        )
        _nonnegative_int(
            self.maximum_cuda_sampling_errors,
            "maximum_cuda_sampling_errors",
        )

    def criteria(self) -> SelfPlayBenchmarkCriteria:
        """Return the immutable acceptance criteria selected by this config."""

        return SelfPlayBenchmarkCriteria(
            minimum_steady_state_batch_occupancy=(
                self.minimum_steady_state_batch_occupancy
            ),
            minimum_native_games_per_second_ratio=(
                self.minimum_native_games_per_second_ratio
            ),
            maximum_invalid_policy_fallbacks=self.maximum_invalid_policy_fallbacks,
            maximum_zero_visit_fallbacks=self.maximum_zero_visit_fallbacks,
            maximum_cpu_sampling_errors=self.maximum_cpu_sampling_errors,
            maximum_cuda_sampling_errors=self.maximum_cuda_sampling_errors,
        )


@dataclass(frozen=True, slots=True)
class SelfPlayBenchmarkCriteria:
    """Acceptance thresholds applied to every raw run and median ratios."""

    minimum_steady_state_batch_occupancy: float
    minimum_native_games_per_second_ratio: float
    maximum_invalid_policy_fallbacks: int
    maximum_zero_visit_fallbacks: int
    maximum_cpu_sampling_errors: int
    maximum_cuda_sampling_errors: int


@dataclass(frozen=True, slots=True)
class SelfPlayBenchmarkRun:
    """One complete measured run, including all collected telemetry."""

    backend: SearchBackend
    repeat: int
    order: int
    seed: int
    elapsed_seconds: float
    metrics: MetricValues
    cpu_metrics: CpuMetrics
    cuda_metrics: CudaMetrics | None

    def metric(self, name: str) -> MetricValue:
        """Return one collected metric by its stable name."""

        for metric_name, value in self.metrics:
            if metric_name == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class SelfPlayBenchmarkSummary:
    """Per-backend medians over raw benchmark runs."""

    backend: SearchBackend
    repeats: int
    median_metrics: MedianMetricValues

    def metric(self, name: str) -> float:
        """Return one median metric by its stable name."""

        for metric_name, value in self.median_metrics:
            if metric_name == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class SelfPlayBenchmarkRatios:
    """Native-over-Python median throughput ratios."""

    native_games_per_second: float
    native_positions_per_second: float
    native_leaf_evaluations_per_second: float


@dataclass(frozen=True, slots=True)
class SelfPlayBenchmarkReport:
    """Complete immutable benchmark evidence and acceptance result."""

    config: SelfPlayBenchmarkConfig
    criteria: SelfPlayBenchmarkCriteria
    raw_runs: tuple[SelfPlayBenchmarkRun, ...]
    python: SelfPlayBenchmarkSummary
    cpp: SelfPlayBenchmarkSummary
    ratios: SelfPlayBenchmarkRatios
    elapsed_seconds: float
    passed: bool
    failures: tuple[str, ...]


def _positive_int(value: object, name: str) -> int:
    parsed = _nonnegative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _unit_interval(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return parsed


def _finite_nonnegative(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


__all__ = [
    "CudaMetrics",
    "CpuMetrics",
    "MedianMetricValues",
    "MetricValue",
    "MetricValues",
    "SelfPlayBenchmarkConfig",
    "SelfPlayBenchmarkCriteria",
    "SelfPlayBenchmarkRatios",
    "SelfPlayBenchmarkReport",
    "SelfPlayBenchmarkRun",
    "SelfPlayBenchmarkSummary",
]
