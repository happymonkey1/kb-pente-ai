from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class SelfPlayHealthThresholds:
    minimum_steady_state_batch_occupancy: float = 0.0
    minimum_mean_root_children_visited: float = 0.0
    maximum_search_collapse_rate: float = 1.0
    maximum_invalid_policy_fallbacks: int = 0
    maximum_zero_visit_fallbacks: int = -1

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_steady_state_batch_occupancy <= 1:
            raise ValueError("Minimum batch occupancy must be between zero and one")
        if self.minimum_mean_root_children_visited < 0:
            raise ValueError("Minimum root breadth cannot be negative")
        if not 0 <= self.maximum_search_collapse_rate <= 1:
            raise ValueError("Maximum collapse rate must be between zero and one")
        if self.maximum_invalid_policy_fallbacks < 0:
            raise ValueError("Maximum invalid-policy fallbacks cannot be negative")
        if self.maximum_zero_visit_fallbacks < -1:
            raise ValueError("Maximum zero-visit fallbacks must be -1 or greater")


def validate_self_play_health(
    metrics: Mapping[str, int | float],
    thresholds: SelfPlayHealthThresholds,
) -> None:
    failures: list[str] = []
    _require_at_most(
        metrics,
        "mcts_invalid_policy_fallbacks",
        thresholds.maximum_invalid_policy_fallbacks,
        failures,
    )
    if thresholds.maximum_zero_visit_fallbacks >= 0:
        _require_at_most(
            metrics,
            "mcts_zero_visit_fallbacks",
            thresholds.maximum_zero_visit_fallbacks,
            failures,
        )
    _require_at_least(
        metrics,
        "steady_state_mean_batch_occupancy",
        thresholds.minimum_steady_state_batch_occupancy,
        failures,
    )
    _require_at_least(
        metrics,
        "mean_root_children_visited",
        thresholds.minimum_mean_root_children_visited,
        failures,
    )
    _require_at_most(
        metrics,
        "search_collapse_rate",
        thresholds.maximum_search_collapse_rate,
        failures,
    )
    _require_at_most(metrics, "self_play_gpu_utilization_sampling_errors", 0, failures)
    _require_at_most(metrics, "self_play_cpu_sampling_errors", 0, failures)
    if failures:
        raise RuntimeError("Self-play health gate failed: " + "; ".join(failures))


def _require_at_least(
    metrics: Mapping[str, int | float],
    name: str,
    threshold: float,
    failures: list[str],
) -> None:
    value = float(metrics.get(name, 0.0))
    if value < threshold:
        failures.append(f"{name}={value:.6g} is below {threshold:.6g}")


def _require_at_most(
    metrics: Mapping[str, int | float],
    name: str,
    threshold: float,
    failures: list[str],
) -> None:
    value = float(metrics.get(name, 0.0))
    if value > threshold:
        failures.append(f"{name}={value:.6g} exceeds {threshold:.6g}")
