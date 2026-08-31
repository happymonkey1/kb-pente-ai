from __future__ import annotations

import unittest

from src.monitoring.device_summary import summarize_device
from src.monitoring.models import MetricValue, TelemetryRecord


def _record(
    metrics: dict[str, MetricValue],
    *,
    event: str = "training_iteration",
    step: int = 1,
) -> TelemetryRecord:
    return TelemetryRecord(
        schema_version=1,
        timestamp_unix=float(step),
        run_id=None,
        event=event,
        step=step,
        metrics=metrics,
    )


class DeviceSummaryTest(unittest.TestCase):
    def test_groups_latest_cuda_metrics_by_training_phase(self) -> None:
        summary = summarize_device(
            [
                _record(
                    {
                        "self_play_gpu_utilization_samples": 8,
                        "self_play_gpu_mean_utilization_percent": 40.0,
                        "self_play_gpu_mean_device_memory_percent": 15.0,
                    }
                ),
                _record(
                    {
                        "self_play_gpu_mean_utilization_percent": 55.0,
                        "learner_gpu_utilization_samples": 4,
                        "learner_gpu_p95_utilization_percent": 91.0,
                        "learner_gpu_peak_memory_allocated_bytes": 3_000_000_000,
                        "learner_gpu_utilization_sampling_errors": 0,
                    },
                    step=2,
                ),
            ]
        )

        self.assertEqual(
            {
                "type": "cuda",
                "logical_core_count": None,
                "phases": {
                    "self_play": {
                        "utilization_samples": 8,
                        "mean_utilization_percent": 55.0,
                        "mean_device_memory_percent": 15.0,
                    },
                    "learner": {
                        "utilization_samples": 4,
                        "p95_utilization_percent": 91.0,
                        "peak_memory_allocated_bytes": 3_000_000_000,
                        "utilization_sampling_errors": 0,
                    },
                },
            },
            summary,
        )

    def test_reports_cpu_without_cuda_metrics(self) -> None:
        summary = summarize_device([_record({"loss": 1.0})])

        self.assertEqual(
            {
                "type": "cpu",
                "logical_core_count": None,
                "phases": {"self_play": None, "learner": None},
            },
            summary,
        )

    def test_reports_unknown_before_a_training_iteration(self) -> None:
        summary = summarize_device([_record({}, event="replay_resume")])

        self.assertEqual(
            {
                "type": "unknown",
                "logical_core_count": None,
                "phases": {"self_play": None, "learner": None},
            },
            summary,
        )

    def test_groups_cpu_process_metrics_by_training_phase(self) -> None:
        summary = summarize_device(
            [
                _record(
                    {
                        "device_type": "cpu",
                        "cpu_logical_core_count": 16,
                        "self_play_cpu_utilization_samples": 12,
                        "self_play_cpu_mean_process_utilization_percent": 43.5,
                        "self_play_cpu_p95_process_utilization_percent": 68.0,
                        "self_play_cpu_max_process_utilization_percent": 72.0,
                        "self_play_cpu_mean_resident_memory_bytes": 1_500_000_000,
                        "self_play_cpu_peak_resident_memory_bytes": 1_800_000_000,
                        "self_play_cpu_sampling_errors": 0,
                    }
                )
            ]
        )

        self.assertEqual(
            {
                "type": "cpu",
                "logical_core_count": 16,
                "phases": {
                    "self_play": {
                        "utilization_samples": 12,
                        "mean_process_utilization_percent": 43.5,
                        "p95_process_utilization_percent": 68.0,
                        "max_process_utilization_percent": 72.0,
                        "mean_resident_memory_bytes": 1_500_000_000,
                        "peak_resident_memory_bytes": 1_800_000_000,
                        "sampling_errors": 0,
                    },
                    "learner": None,
                },
            },
            summary,
        )


if __name__ == "__main__":
    unittest.main()
