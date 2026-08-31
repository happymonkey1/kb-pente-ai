from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ValueMetrics:
    calibration_error: float
    negative_outcomes: int
    draw_outcomes: int
    positive_outcomes: int
    negative_mean_prediction: float
    draw_mean_prediction: float
    positive_mean_prediction: float

    def to_metrics(self, prefix: str = "value") -> dict[str, int | float]:
        return {
            f"{prefix}_calibration_error": self.calibration_error,
            f"{prefix}_negative_outcomes": self.negative_outcomes,
            f"{prefix}_draw_outcomes": self.draw_outcomes,
            f"{prefix}_positive_outcomes": self.positive_outcomes,
            f"{prefix}_negative_mean_prediction": self.negative_mean_prediction,
            f"{prefix}_draw_mean_prediction": self.draw_mean_prediction,
            f"{prefix}_positive_mean_prediction": self.positive_mean_prediction,
        }


class ValueMetricsAccumulator:
    def __init__(self, calibration_bins: int = 10) -> None:
        if calibration_bins < 2:
            raise ValueError("Value calibration requires at least two bins")
        self._calibration_bins = calibration_bins
        self._bin_counts = np.zeros(calibration_bins, dtype=np.int64)
        self._bin_predictions = np.zeros(calibration_bins, dtype=np.float64)
        self._bin_outcomes = np.zeros(calibration_bins, dtype=np.float64)
        self._outcome_counts = np.zeros(3, dtype=np.int64)
        self._outcome_prediction_sums = np.zeros(3, dtype=np.float64)

    def add(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        flat_predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        flat_targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        if flat_predictions.shape != flat_targets.shape:
            raise ValueError("Value predictions and targets must have the same shape")
        if not np.isfinite(flat_predictions).all() or not np.isfinite(flat_targets).all():
            raise ValueError("Value metrics require finite predictions and targets")
        if np.any(np.abs(flat_predictions) > 1.000001):
            raise ValueError("Value predictions must be in [-1, 1]")
        if np.any(np.abs(flat_targets) > 1.000001):
            raise ValueError("Value targets must be in [-1, 1]")

        probabilities = np.clip((flat_predictions + 1.0) / 2.0, 0.0, 1.0)
        observed = (flat_targets + 1.0) / 2.0
        bin_indices = np.minimum(
            (probabilities * self._calibration_bins).astype(np.int64),
            self._calibration_bins - 1,
        )
        np.add.at(self._bin_counts, bin_indices, 1)
        np.add.at(self._bin_predictions, bin_indices, probabilities)
        np.add.at(self._bin_outcomes, bin_indices, observed)

        outcome_indices = np.where(flat_targets < -0.5, 0, np.where(flat_targets > 0.5, 2, 1))
        np.add.at(self._outcome_counts, outcome_indices, 1)
        np.add.at(self._outcome_prediction_sums, outcome_indices, flat_predictions)

    def finish(self) -> ValueMetrics:
        total = int(self._bin_counts.sum())
        if total == 0:
            raise RuntimeError("Value metrics require at least one example")
        populated = self._bin_counts > 0
        mean_predictions = np.divide(
            self._bin_predictions,
            self._bin_counts,
            out=np.zeros_like(self._bin_predictions),
            where=populated,
        )
        mean_outcomes = np.divide(
            self._bin_outcomes,
            self._bin_counts,
            out=np.zeros_like(self._bin_outcomes),
            where=populated,
        )
        calibration_error = float(
            np.sum(
                self._bin_counts[populated]
                * np.abs(mean_predictions[populated] - mean_outcomes[populated])
            )
            / total
        )
        outcome_means = np.divide(
            self._outcome_prediction_sums,
            self._outcome_counts,
            out=np.zeros_like(self._outcome_prediction_sums),
            where=self._outcome_counts > 0,
        )
        return ValueMetrics(
            calibration_error=calibration_error,
            negative_outcomes=int(self._outcome_counts[0]),
            draw_outcomes=int(self._outcome_counts[1]),
            positive_outcomes=int(self._outcome_counts[2]),
            negative_mean_prediction=float(outcome_means[0]),
            draw_mean_prediction=float(outcome_means[1]),
            positive_mean_prediction=float(outcome_means[2]),
        )
