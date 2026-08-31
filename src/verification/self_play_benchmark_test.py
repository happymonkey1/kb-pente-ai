from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

import numpy as np
import torch

from src.game.pente.pente_game import PenteGame
from src.game.pente.pente_board import PenteBoard
from src.game.pente.rules import PenteRuleset
from src.model.model_v1 import PenteNetConfig
from src.monitoring.cpu_metrics import CpuMetrics
from src.monitoring.cuda_metrics import CudaMetrics
from src.verification import self_play_benchmark as benchmark
from src.verification.self_play_benchmark_reports import SelfPlayBenchmarkConfig


class _FakeModel:
    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.config = PenteNetConfig(board_size=5, action_size=25)
        self.training = True
        self.mode_calls: list[str] = []
        self.warmup_batch_sizes: list[int] = []

    def eval(self) -> _FakeModel:
        self.training = False
        self.mode_calls.append("eval")
        return self

    def train(self, mode: bool = True) -> _FakeModel:
        self.training = mode
        self.mode_calls.append("train" if mode else "eval")
        return self

    def evaluate(self, _position: PenteBoard) -> tuple[np.ndarray, float]:
        return np.empty(0), 0.0

    def evaluate_batch(
        self,
        positions: Sequence[PenteBoard],
    ) -> tuple[np.ndarray, np.ndarray]:
        self.warmup_batch_sizes.append(len(positions))
        return np.empty((len(positions), 0)), np.empty(len(positions))


class _FakeGenerator:
    instances: list[_FakeGenerator] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args
        backend = kwargs["search_backend"]
        if not isinstance(backend, str):
            raise TypeError("fake backend must be a string")
        self.backend = backend
        self.play_calls: list[tuple[int, int]] = []
        _FakeGenerator.instances.append(self)

    def play_games(self, games: int, max_active_games: int) -> tuple[list[object], list[object]]:
        self.play_calls.append((games, max_active_games))
        return [object()], []


def _cpu_metrics(errors: int = 0) -> CpuMetrics:
    return CpuMetrics(4, 2, 40.0, 55.0, 70.0, 100, 120, errors)


def _cuda_metrics(errors: int = 0) -> CudaMetrics:
    return CudaMetrics(2, 20.0, 30.0, 40, 10.0, 20, 100, 120, errors)


def _config(**overrides: object) -> SelfPlayBenchmarkConfig:
    values: dict[str, object] = {
        "board_size": 5,
        "ruleset": PenteRuleset.FREESTYLE,
        "games": 4,
        "max_active_games": 2,
        "simulations": 2,
        "temp_threshold": 3,
        "repeats": 2,
        "warmup_batches": 0,
        "seed": 41,
        "minimum_steady_state_batch_occupancy": 0.8,
        "minimum_native_games_per_second_ratio": 2.0,
    }
    values.update(overrides)
    return SelfPlayBenchmarkConfig(**values)  # type: ignore[arg-type]


class _MetricSource:
    def __init__(self, games: int = 4) -> None:
        self.games = games
        self.calls: list[tuple[str, float]] = []
        self.by_backend = {"python": [1.0, 3.0], "cpp": [6.0, 8.0]}

    def __call__(self, _games: object, _batches: object, _elapsed: float) -> dict[str, int | float]:
        backend = _FakeGenerator.instances[-1].backend
        index = sum(call[0] == backend for call in self.calls)
        speed = self.by_backend[backend][index]
        self.calls.append((backend, speed))
        return {
            "games": self.games,
            "games_per_second": speed,
            "positions_per_second": speed * 10.0,
            "leaf_evaluations_per_second": speed * 20.0,
            "steady_state_mean_batch_occupancy": 0.9,
            "mcts_invalid_policy_fallbacks": 0,
            "mcts_zero_visit_fallbacks": 0,
            "mean_inference_batch_size": speed,
            "duplicate_leaf_rate": 0.1,
            "native_select_seconds": speed / 10.0,
            "unique_trajectories": 2,
        }


def _measure_cpu(operation: Callable[[], object]) -> tuple[object, CpuMetrics]:
    return operation(), _cpu_metrics()


def _measure_cpu_errors(operation: Callable[[], object]) -> tuple[object, CpuMetrics]:
    return operation(), _cpu_metrics(errors=1)


def _measure_cpu_cuda(
    operation: Callable[[], object],
) -> tuple[object, CpuMetrics]:
    return operation(), _cpu_metrics()


def _measure_cuda_cpu(
    _device: torch.device,
    operation: Callable[[], object],
) -> tuple[object, CudaMetrics | None]:
    return operation(), None


class SelfPlayBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeGenerator.instances.clear()
        self.game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
        self.model = _FakeModel()

    def test_alternates_backends_reuses_seed_and_preserves_medians(self) -> None:
        source = _MetricSource()
        config = _config()
        with (
            patch.object(benchmark, "SelfPlayGenerator", _FakeGenerator),
            patch.object(benchmark, "collect_self_play_metrics", side_effect=source),
            patch.object(benchmark, "measure_cpu_operation", side_effect=_measure_cpu),
            patch.object(benchmark, "measure_cuda_operation", side_effect=_measure_cuda_cpu),
        ):
            report = benchmark.run_self_play_benchmark(
                self.model,
                self.game,
                config,
                torch.device("cpu"),
            )

        self.assertEqual(
            [("python", 0, 41), ("cpp", 0, 41), ("cpp", 1, 42), ("python", 1, 42)],
            [(run.backend, run.repeat, run.seed) for run in report.raw_runs],
        )
        self.assertEqual(2, report.python.repeats)
        self.assertEqual(2.0, report.python.metric("games_per_second"))
        self.assertEqual(7.0, report.cpp.metric("games_per_second"))
        self.assertEqual(3.5, report.ratios.native_games_per_second)
        self.assertEqual(3.5, report.ratios.native_positions_per_second)
        self.assertEqual(3.5, report.ratios.native_leaf_evaluations_per_second)
        self.assertEqual(4, dict(report.raw_runs[0].metrics)["games"])
        self.assertEqual(0, dict(report.raw_runs[0].metrics)["self_play_cpu_sampling_errors"])
        self.assertTrue(report.passed, report.failures)
        self.assertTrue(self.model.training)
        self.assertEqual((4, 2), _FakeGenerator.instances[0].play_calls[0])

    def test_warmup_uses_active_batch_and_cuda_sampling_is_conditional(self) -> None:
        source = _MetricSource()
        config = _config(
            repeats=1,
            warmup_batches=2,
            minimum_native_games_per_second_ratio=0.0,
        )
        sync_calls: list[torch.device] = []
        with (
            patch.object(benchmark, "SelfPlayGenerator", _FakeGenerator),
            patch.object(benchmark, "collect_self_play_metrics", side_effect=source),
            patch.object(benchmark, "measure_cpu_operation", side_effect=_measure_cpu),
            patch.object(benchmark, "measure_cuda_operation", side_effect=lambda device, operation: (operation(), _cuda_metrics())),
            patch.object(benchmark, "_synchronize_cuda", side_effect=sync_calls.append),
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            cuda_model = _FakeModel(torch.device("cuda"))
            report = benchmark.run_self_play_benchmark(
                cuda_model,
                self.game,
                config,
                torch.device("cuda"),
            )

        self.assertEqual([2, 2], cuda_model.warmup_batch_sizes)
        self.assertEqual(6, len(sync_calls))
        self.assertIsNotNone(report.raw_runs[0].cuda_metrics)
        self.assertIn("self_play_gpu_utilization_sampling_errors", dict(report.raw_runs[0].metrics))
        self.assertTrue(report.passed, report.failures)

    def test_failures_cover_incomplete_unhealthy_and_sampling_runs(self) -> None:
        source = _MetricSource(games=3)
        source.by_backend = {"python": [1.0, 1.0], "cpp": [1.0, 1.0]}
        config = _config(
            repeats=1,
            minimum_steady_state_batch_occupancy=1.0,
            minimum_native_games_per_second_ratio=2.0,
        )

        def unhealthy_metrics(
            games: object,
            batches: object,
            elapsed: float,
        ) -> dict[str, int | float]:
            values = source(games, batches, elapsed)
            values["mcts_invalid_policy_fallbacks"] = 1
            values["mcts_zero_visit_fallbacks"] = 1
            values["steady_state_mean_batch_occupancy"] = 0.2
            return values

        with (
            patch.object(benchmark, "SelfPlayGenerator", _FakeGenerator),
            patch.object(benchmark, "collect_self_play_metrics", side_effect=unhealthy_metrics),
            patch.object(benchmark, "measure_cpu_operation", side_effect=_measure_cpu_errors),
            patch.object(benchmark, "measure_cuda_operation", side_effect=_measure_cuda_cpu),
        ):
            report = benchmark.run_self_play_benchmark(
                self.model,
                self.game,
                config,
                torch.device("cpu"),
            )

        self.assertFalse(report.passed)
        self.assertEqual(
            11,
            len(report.failures),
        )
        self.assertIn("did not complete", report.failures[0])
        self.assertIn("occupancy", report.failures[1])
        self.assertIn("invalid-policy", report.failures[2])
        self.assertIn("zero-visit", report.failures[3])
        self.assertIn("CPU metric", report.failures[4])

    def test_validation_rejects_invalid_configuration_and_model_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "warmup_batches cannot be negative"):
            _config(warmup_batches=-1)
        with self.assertRaisesRegex(ValueError, "simulations must be positive"):
            _config(simulations=0)
        with self.assertRaisesRegex(ValueError, "board size"):
            benchmark.run_self_play_benchmark(
                self.model,
                PenteGame(9, ruleset=PenteRuleset.FREESTYLE),
                _config(),
                torch.device("cpu"),
            )
        mismatched = SimpleNamespace(
            device=torch.device("cpu"),
            config=SimpleNamespace(board_size=9, action_size=81),
            eval=lambda: None,
            train=lambda mode=True: None,
            evaluate=lambda position: None,
            evaluate_batch=lambda positions: None,
        )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            benchmark.run_self_play_benchmark(
                mismatched,  # type: ignore[arg-type]
                self.game,
                _config(),
                torch.device("cpu"),
            )

    def test_model_mode_and_torch_threads_restore_on_failure(self) -> None:
        config = _config(torch_threads=1)
        with (
            patch.object(benchmark, "SelfPlayGenerator", side_effect=RuntimeError("boom")),
            patch.object(torch, "get_num_threads", return_value=8),
            patch.object(torch, "set_num_threads") as set_threads,
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                benchmark.run_self_play_benchmark(
                    self.model,
                    self.game,
                    config,
                    torch.device("cpu"),
                )
        self.assertTrue(self.model.training)
        self.assertEqual([call(1), call(8)], set_threads.call_args_list)


if __name__ == "__main__":
    unittest.main()
