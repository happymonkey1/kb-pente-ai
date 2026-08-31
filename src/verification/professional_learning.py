from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from src.telemetry import TELEMETRY_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ProfessionalLearningCriteria:
    minimum_cross_entropy_reduction: float = 0.05
    minimum_top_one_gain: float = 0.02
    minimum_top_five_gain: float = 0.05
    maximum_value_mse_ratio: float = 1.05

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_cross_entropy_reduction <= 1:
            raise ValueError("Cross-entropy reduction must be between zero and one")
        if not 0 <= self.minimum_top_one_gain <= 1:
            raise ValueError("Top-one gain must be between zero and one")
        if not 0 <= self.minimum_top_five_gain <= 1:
            raise ValueError("Top-five gain must be between zero and one")
        if self.maximum_value_mse_ratio < 0:
            raise ValueError("Maximum value MSE ratio cannot be negative")


@dataclass(frozen=True, slots=True)
class ProfessionalLearningReport:
    examples: int
    baseline_policy_cross_entropy: float
    final_policy_cross_entropy: float
    policy_cross_entropy_reduction: float
    baseline_policy_top_one_accuracy: float
    final_policy_top_one_accuracy: float
    policy_top_one_gain: float
    baseline_policy_top_five_accuracy: float
    final_policy_top_five_accuracy: float
    policy_top_five_gain: float
    baseline_value_mse: float
    final_value_mse: float
    value_mse_ratio: float
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_professional_learning(
    telemetry_path: str | Path,
    criteria: ProfessionalLearningCriteria | None = None,
) -> ProfessionalLearningReport:
    selected_criteria = criteria or ProfessionalLearningCriteria()
    baseline_metrics: dict[str, Any] | None = None
    final_metrics: dict[str, Any] | None = None

    with Path(telemetry_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid telemetry JSON on line {line_number}") from error
            if record.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
                raise ValueError(f"Unsupported telemetry schema on line {line_number}")
            metrics = record.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"Telemetry metrics must be an object on line {line_number}")
            if record.get("event") == "professional_validation_baseline":
                if baseline_metrics is not None:
                    raise ValueError("Telemetry contains multiple professional baselines")
                baseline_metrics = metrics
            elif (
                record.get("event") == "training_iteration"
                and "professional_validation_examples" in metrics
            ):
                final_metrics = metrics

    if baseline_metrics is None:
        raise ValueError("Telemetry does not contain a professional validation baseline")
    if final_metrics is None:
        raise ValueError("Telemetry does not contain updated professional validation metrics")

    baseline_examples = _positive_integer(baseline_metrics, "professional_validation_examples")
    final_examples = _positive_integer(final_metrics, "professional_validation_examples")
    if baseline_examples != final_examples:
        raise ValueError("Baseline and final validation example counts differ")

    baseline_cross_entropy = _finite_metric(
        baseline_metrics,
        "professional_validation_policy_cross_entropy",
    )
    final_cross_entropy = _finite_metric(
        final_metrics,
        "professional_validation_policy_cross_entropy",
    )
    if baseline_cross_entropy <= 0:
        raise ValueError("Baseline policy cross-entropy must be positive")
    cross_entropy_reduction = (
        baseline_cross_entropy - final_cross_entropy
    ) / baseline_cross_entropy

    baseline_top_one = _probability_metric(
        baseline_metrics,
        "professional_validation_policy_top_one_accuracy",
    )
    final_top_one = _probability_metric(
        final_metrics,
        "professional_validation_policy_top_one_accuracy",
    )
    baseline_top_five = _probability_metric(
        baseline_metrics,
        "professional_validation_policy_top_five_accuracy",
    )
    final_top_five = _probability_metric(
        final_metrics,
        "professional_validation_policy_top_five_accuracy",
    )
    baseline_value_mse = _finite_metric(
        baseline_metrics,
        "professional_validation_value_mse",
    )
    final_value_mse = _finite_metric(
        final_metrics,
        "professional_validation_value_mse",
    )
    if baseline_value_mse < 0 or final_value_mse < 0:
        raise ValueError("Value MSE metrics cannot be negative")
    value_mse_ratio = (
        final_value_mse / baseline_value_mse
        if baseline_value_mse > 0
        else (0.0 if final_value_mse == 0 else math.inf)
    )

    top_one_gain = final_top_one - baseline_top_one
    top_five_gain = final_top_five - baseline_top_five
    failures = []
    if cross_entropy_reduction < selected_criteria.minimum_cross_entropy_reduction:
        failures.append("policy cross-entropy reduction is below threshold")
    if top_one_gain < selected_criteria.minimum_top_one_gain:
        failures.append("top-one accuracy gain is below threshold")
    if top_five_gain < selected_criteria.minimum_top_five_gain:
        failures.append("top-five accuracy gain is below threshold")
    if value_mse_ratio > selected_criteria.maximum_value_mse_ratio:
        failures.append("value MSE ratio exceeds threshold")

    return ProfessionalLearningReport(
        examples=baseline_examples,
        baseline_policy_cross_entropy=baseline_cross_entropy,
        final_policy_cross_entropy=final_cross_entropy,
        policy_cross_entropy_reduction=cross_entropy_reduction,
        baseline_policy_top_one_accuracy=baseline_top_one,
        final_policy_top_one_accuracy=final_top_one,
        policy_top_one_gain=top_one_gain,
        baseline_policy_top_five_accuracy=baseline_top_five,
        final_policy_top_five_accuracy=final_top_five,
        policy_top_five_gain=top_five_gain,
        baseline_value_mse=baseline_value_mse,
        final_value_mse=final_value_mse,
        value_mse_ratio=value_mse_ratio,
        passed=not failures,
        failures=tuple(failures),
    )


def _finite_metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Telemetry metric {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Telemetry metric {name} must be finite")
    return result


def _positive_integer(metrics: dict[str, Any], name: str) -> int:
    value = _finite_metric(metrics, name)
    if not value.is_integer() or value < 1:
        raise ValueError(f"Telemetry metric {name} must be a positive integer")
    return int(value)


def _probability_metric(metrics: dict[str, Any], name: str) -> float:
    value = _finite_metric(metrics, name)
    if not 0 <= value <= 1:
        raise ValueError(f"Telemetry metric {name} must be between zero and one")
    return value
