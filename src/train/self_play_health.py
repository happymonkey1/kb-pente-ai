from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Mapping

from src.telemetry import MetricSink


logger = logging.getLogger(__name__)


class SelfPlayHealthFailurePolicy(str, Enum):
    """Controls whether a threshold breach stops a self-play iteration."""

    WARN = "warn"
    ERROR = "error"

    @classmethod
    def parse(
        cls,
        value: SelfPlayHealthFailurePolicy | str,
    ) -> SelfPlayHealthFailurePolicy:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Self-play health failure policy must be a string")
        normalized = value.lower()
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(policy.value for policy in cls)
            raise ValueError(
                f"Unknown self-play health failure policy {value!r}; "
                f"expected one of: {choices}"
            ) from error


@dataclass(frozen=True, slots=True)
class SelfPlayHealthReport:
    """Immutable result of evaluating one self-play health sample."""

    failures: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not self.failures

    @property
    def message(self) -> str:
        return "Self-play health gate failed: " + "; ".join(self.failures)


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


def evaluate_self_play_health(
    metrics: Mapping[str, int | float],
    thresholds: SelfPlayHealthThresholds,
) -> SelfPlayHealthReport:
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
    return SelfPlayHealthReport(tuple(failures))


def validate_self_play_health(
    metrics: Mapping[str, int | float],
    thresholds: SelfPlayHealthThresholds,
) -> None:
    """Evaluate health and raise for callers that require strict enforcement."""

    report = evaluate_self_play_health(metrics, thresholds)
    enforce_self_play_health(report, SelfPlayHealthFailurePolicy.ERROR)


def enforce_self_play_health(
    report: SelfPlayHealthReport,
    health_failure_policy: SelfPlayHealthFailurePolicy | str,
) -> None:
    """Apply a health policy to an already evaluated self-play report."""

    policy = SelfPlayHealthFailurePolicy.parse(health_failure_policy)
    if not report.healthy and policy is SelfPlayHealthFailurePolicy.ERROR:
        raise RuntimeError(report.message)


def validate_and_emit_self_play_health(
    metric_sink: MetricSink,
    step: int,
    device_type: str,
    cpu_logical_core_count: int,
    generation_seconds: float,
    metrics: Mapping[str, int | float],
    thresholds: SelfPlayHealthThresholds,
    health_failure_policy: SelfPlayHealthFailurePolicy | str = (
        SelfPlayHealthFailurePolicy.WARN
    ),
) -> SelfPlayHealthReport:
    """Report a breach and enforce the configured policy.

    Health threshold breaches are expected operational signals. Runtime errors
    from self-play, search, model evaluation, or telemetry are intentionally
    outside this boundary and propagate to the caller.
    """

    policy = SelfPlayHealthFailurePolicy.parse(health_failure_policy)
    report = evaluate_self_play_health(metrics, thresholds)
    if report.healthy:
        return report

    failure_metrics: dict[str, int | float | str | bool] = dict(metrics)
    failure_metrics.update(
        {
            "device_type": device_type,
            "cpu_logical_core_count": cpu_logical_core_count,
            "generation_seconds": generation_seconds,
            "error": report.message,
            "failure_count": len(report.failures),
            "health_failure_policy": policy.value,
            "training_continues": policy is SelfPlayHealthFailurePolicy.WARN,
            "healthy": False,
        }
    )
    logger.warning(
        "Self-play health threshold breach at step %s; policy=%s; training_continues=%s: %s",
        step,
        policy.value,
        policy is SelfPlayHealthFailurePolicy.WARN,
        report.message,
    )
    metric_sink.emit("self_play_health_failure", step, failure_metrics)
    enforce_self_play_health(report, policy)
    return report


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
