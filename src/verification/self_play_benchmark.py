"""Reusable loaded-model full self-play benchmarking."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real
import statistics
import time
from typing import Protocol, Sequence

import numpy as np
import torch

from src.game.pente.pente_game import PenteGame
from src.game.pente.pente_board import PenteBoard
from src.mcts.mcts_v2 import MCTSArgs, PolicyValueEvaluator
from src.monitoring.cpu_metrics import CpuMetrics, measure_cpu_operation
from src.monitoring.cuda_metrics import CudaMetrics, measure_cuda_operation
from src.train.self_play_generation import SelfPlayGenerator
from src.train.self_play_metrics import collect_self_play_metrics
from src.train.self_play_args import SearchBackend
from src.verification.self_play_benchmark_reports import (
    MetricValue,
    MetricValues,
    SelfPlayBenchmarkConfig,
    SelfPlayBenchmarkReport,
    SelfPlayBenchmarkRun,
    SelfPlayBenchmarkSummary,
    SelfPlayBenchmarkRatios,
)


class _BenchmarkModel(PolicyValueEvaluator, Protocol):
    """Loaded model boundary needed by warmup and self-play evaluation."""

    training: bool

    def eval(self) -> object:
        """Switch the model to evaluation mode."""

    def train(self, mode: bool = True) -> object:
        """Restore the model's prior training mode."""

    def evaluate_batch(
        self,
        positions: Sequence[PenteBoard],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warm the model with a batch of board positions."""


def run_self_play_benchmark(
    model: _BenchmarkModel,
    game: PenteGame,
    config: SelfPlayBenchmarkConfig | None = None,
    device: torch.device | None = None,
) -> SelfPlayBenchmarkReport:
    """Run paired Python/native self-play repeats on one loaded model.

    The model and game are supplied by the caller so checkpoint and runtime
    concerns stay outside this reusable measurement boundary. Each repeat
    gives both backends the same seed and complete configuration while the
    backend order alternates to reduce order bias.
    """

    selected = config or SelfPlayBenchmarkConfig()
    selected_device = _selected_device(model, device)
    _validate_inputs(model, game, selected_device, selected)

    previous_torch_threads = (
        torch.get_num_threads() if selected.torch_threads is not None else None
    )
    previous_training = getattr(model, "training", None)
    if selected.torch_threads is not None:
        torch.set_num_threads(selected.torch_threads)

    try:
        model.eval()
        _warm_model(model, game, selected_device, selected)
        started = time.perf_counter()
        raw_runs: list[SelfPlayBenchmarkRun] = []
        for repeat in range(selected.repeats):
            backends: tuple[SearchBackend, SearchBackend] = (
                ("python", "cpp")
                if repeat % 2 == 0
                else ("cpp", "python")
            )
            repeat_seed = selected.seed + repeat
            for order, backend in enumerate(backends):
                raw_runs.append(
                    _run_once(
                        model,
                        game,
                        selected_device,
                        selected,
                        backend,
                        repeat,
                        order,
                        repeat_seed,
                    )
                )
        elapsed_seconds = time.perf_counter() - started
    finally:
        if previous_torch_threads is not None:
            torch.set_num_threads(previous_torch_threads)
        _restore_model_mode(model, previous_training)

    frozen_runs = tuple(raw_runs)
    python_summary = _summarize("python", frozen_runs)
    cpp_summary = _summarize("cpp", frozen_runs)
    ratios = _ratios(python_summary, cpp_summary)
    criteria = selected.criteria()
    failures = _failures(selected, frozen_runs, ratios)
    return SelfPlayBenchmarkReport(
        config=selected,
        criteria=criteria,
        raw_runs=frozen_runs,
        python=python_summary,
        cpp=cpp_summary,
        ratios=ratios,
        elapsed_seconds=elapsed_seconds,
        passed=not failures,
        failures=tuple(failures),
    )


def _selected_device(model: object, device: torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    model_device = getattr(model, "device", torch.device("cpu"))
    return torch.device(model_device)


def _validate_inputs(
    model: object,
    game: PenteGame,
    device: torch.device,
    config: SelfPlayBenchmarkConfig,
) -> None:
    if not isinstance(game, PenteGame):
        raise TypeError("game must be a PenteGame instance")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Self-play benchmark supports only CPU and CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA benchmark requested without an available CUDA device")
    if config.board_size != game.get_board_size():
        raise ValueError("Benchmark board size does not match game")
    if config.ruleset is not game.ruleset:
        raise ValueError("Benchmark ruleset does not match game")
    for name in ("eval", "train", "evaluate", "evaluate_batch"):
        if not callable(getattr(model, name, None)):
            raise TypeError(f"model must provide {name}()")

    model_device = getattr(model, "device", None)
    if model_device is not None and torch.device(model_device) != device:
        raise ValueError("Model device does not match benchmark device")
    model_config = getattr(model, "config", None)
    if model_config is None:
        return
    if (
        getattr(model_config, "board_size", config.board_size) != config.board_size
        or getattr(model_config, "action_size", game.get_action_size())
        != game.get_action_size()
    ):
        raise ValueError("Model dimensions do not match benchmark game")


def _warm_model(
    model: _BenchmarkModel,
    game: PenteGame,
    device: torch.device,
    config: SelfPlayBenchmarkConfig,
) -> None:
    if config.warmup_batches == 0:
        return
    positions = tuple(
        game.init_board() for _ in range(config.max_active_games)
    )
    _synchronize_cuda(device)
    for _ in range(config.warmup_batches):
        model.evaluate_batch(positions)
    _synchronize_cuda(device)


def _synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _restore_model_mode(model: object, previous_training: object) -> None:
    if not isinstance(previous_training, bool):
        return
    restore = getattr(model, "train" if previous_training else "eval")
    restore()


def _run_once(
    model: _BenchmarkModel,
    game: PenteGame,
    device: torch.device,
    config: SelfPlayBenchmarkConfig,
    backend: SearchBackend,
    repeat: int,
    order: int,
    seed: int,
) -> SelfPlayBenchmarkRun:
    generator = SelfPlayGenerator(
        game,
        model,
        MCTSArgs(num_simulations=config.simulations),
        config.temp_threshold,
        np.random.default_rng(seed),
        deduplicate_evaluations=device.type != "cuda",
        search_backend=backend,
        native_worker_threads=config.native_worker_threads,
    )

    def operation() -> tuple[dict[str, MetricValue], float]:
        _synchronize_cuda(device)
        started = time.perf_counter()
        games, batches = generator.play_games(
            config.games,
            config.max_active_games,
        )
        _synchronize_cuda(device)
        elapsed_seconds = time.perf_counter() - started
        metrics = collect_self_play_metrics(games, batches, elapsed_seconds)
        return metrics, elapsed_seconds

    measured, cpu_metrics = measure_cpu_operation(
        lambda: measure_cuda_operation(device, operation),
    )
    (metrics, elapsed_seconds), cuda_metrics = measured
    complete_metrics: dict[str, MetricValue] = dict(metrics)
    complete_metrics.update(cpu_metrics.to_metrics("self_play"))
    if cuda_metrics is not None:
        complete_metrics.update(cuda_metrics.to_metrics("self_play_gpu"))
    return SelfPlayBenchmarkRun(
        backend=backend,
        repeat=repeat,
        order=order,
        seed=seed,
        elapsed_seconds=elapsed_seconds,
        metrics=_freeze_metrics(complete_metrics),
        cpu_metrics=cpu_metrics,
        cuda_metrics=cuda_metrics,
    )


def _freeze_metrics(metrics: Mapping[str, object]) -> MetricValues:
    frozen: list[tuple[str, MetricValue]] = []
    for name, value in sorted(metrics.items()):
        if not isinstance(name, str):
            raise TypeError("Self-play metric names must be strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Self-play metric {name} must be numeric")
        parsed: MetricValue
        if isinstance(value, Integral):
            parsed = int(value)
        else:
            parsed = float(value)
        if not math.isfinite(float(parsed)):
            raise ValueError(f"Self-play metric {name} must be finite")
        frozen.append((name, parsed))
    return tuple(frozen)


def _summarize(
    backend: SearchBackend,
    runs: tuple[SelfPlayBenchmarkRun, ...],
) -> SelfPlayBenchmarkSummary:
    selected = tuple(run for run in runs if run.backend == backend)
    if not selected:
        raise ValueError(f"No benchmark runs for {backend}")
    by_name: dict[str, list[float]] = {}
    for run in selected:
        for name, value in run.metrics:
            by_name.setdefault(name, []).append(float(value))
    medians = tuple(
        (name, float(statistics.median(values)))
        for name, values in sorted(by_name.items())
    )
    return SelfPlayBenchmarkSummary(backend, len(selected), medians)


def _ratios(
    python: SelfPlayBenchmarkSummary,
    cpp: SelfPlayBenchmarkSummary,
) -> SelfPlayBenchmarkRatios:
    return SelfPlayBenchmarkRatios(
        native_games_per_second=_ratio(
            cpp.metric("games_per_second"),
            python.metric("games_per_second"),
        ),
        native_positions_per_second=_ratio(
            cpp.metric("positions_per_second"),
            python.metric("positions_per_second"),
        ),
        native_leaf_evaluations_per_second=_ratio(
            cpp.metric("leaf_evaluations_per_second"),
            python.metric("leaf_evaluations_per_second"),
        ),
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _failures(
    config: SelfPlayBenchmarkConfig,
    runs: tuple[SelfPlayBenchmarkRun, ...],
    ratios: SelfPlayBenchmarkRatios,
) -> list[str]:
    failures: list[str] = []
    for run in runs:
        metrics = dict(run.metrics)
        prefix = f"{run.backend} repeat {run.repeat}"
        if metrics.get("games", 0) != config.games:
            failures.append(f"{prefix} did not complete the requested games")
        if float(metrics.get("steady_state_mean_batch_occupancy", 0.0)) < (
            config.minimum_steady_state_batch_occupancy
        ):
            failures.append(f"{prefix} steady-state occupancy is below threshold")
        if int(metrics.get("mcts_invalid_policy_fallbacks", 0)) > (
            config.maximum_invalid_policy_fallbacks
        ):
            failures.append(f"{prefix} has too many invalid-policy fallbacks")
        if int(metrics.get("mcts_zero_visit_fallbacks", 0)) > (
            config.maximum_zero_visit_fallbacks
        ):
            failures.append(f"{prefix} has too many zero-visit fallbacks")
        if run.cpu_metrics.sampling_errors > config.maximum_cpu_sampling_errors:
            failures.append(f"{prefix} has too many CPU metric sampling errors")
        if (
            run.cuda_metrics is not None
            and run.cuda_metrics.sampling_errors > config.maximum_cuda_sampling_errors
        ):
            failures.append(f"{prefix} has too many CUDA metric sampling errors")
    if (
        ratios.native_games_per_second
        < config.minimum_native_games_per_second_ratio
    ):
        failures.append("native games/sec ratio is below threshold")
    return failures


__all__ = ["run_self_play_benchmark"]
