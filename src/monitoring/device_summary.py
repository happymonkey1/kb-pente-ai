from __future__ import annotations

from collections.abc import Iterable

from src.monitoring.models import MetricValue, TelemetryRecord


_CUDA_PHASE_PREFIXES = {
    "self_play": "self_play_gpu",
    "learner": "learner_gpu",
}
_CUDA_METRIC_SUFFIXES = (
    "utilization_samples",
    "mean_utilization_percent",
    "p95_utilization_percent",
    "max_utilization_percent",
    "mean_device_memory_percent",
    "max_device_memory_percent",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
    "utilization_sampling_errors",
)
_CPU_PHASE_PREFIXES = {
    "self_play": "self_play_cpu",
    "learner": "learner_cpu",
}
_CPU_METRIC_SUFFIXES = (
    "utilization_samples",
    "mean_process_utilization_percent",
    "p95_process_utilization_percent",
    "max_process_utilization_percent",
    "mean_resident_memory_bytes",
    "peak_resident_memory_bytes",
    "sampling_errors",
)


def summarize_device(records: Iterable[TelemetryRecord]) -> dict[str, object]:
    latest_training_metrics: dict[str, MetricValue] = {}
    has_training_iteration = False
    for record in records:
        if record.event != "training_iteration":
            continue
        has_training_iteration = True
        latest_training_metrics.update(record.metrics)

    cuda_phases = {
        phase: _phase_metrics(latest_training_metrics, prefix, _CUDA_METRIC_SUFFIXES)
        for phase, prefix in _CUDA_PHASE_PREFIXES.items()
    }
    cpu_phases = {
        phase: _phase_metrics(latest_training_metrics, prefix, _CPU_METRIC_SUFFIXES)
        for phase, prefix in _CPU_PHASE_PREFIXES.items()
    }
    reported_type = latest_training_metrics.get("device_type")
    if reported_type in ("cpu", "cuda"):
        device_type = reported_type
    elif any(phase is not None for phase in cuda_phases.values()):
        device_type = "cuda"
    elif any(phase is not None for phase in cpu_phases.values()):
        device_type = "cpu"
    elif has_training_iteration:
        device_type = "cpu"
    else:
        device_type = "unknown"

    if device_type == "cuda":
        phases = cuda_phases
    elif device_type == "cpu":
        phases = cpu_phases
    else:
        phases = {phase: None for phase in _CUDA_PHASE_PREFIXES}

    logical_core_count = latest_training_metrics.get("cpu_logical_core_count")
    if (
        isinstance(logical_core_count, bool)
        or not isinstance(logical_core_count, int)
        or logical_core_count < 1
    ):
        logical_core_count = None

    return {
        "type": device_type,
        "logical_core_count": logical_core_count,
        "phases": phases,
    }


def _phase_metrics(
    metrics: dict[str, MetricValue],
    prefix: str,
    suffixes: tuple[str, ...],
) -> dict[str, int | float] | None:
    result: dict[str, int | float] = {}
    for suffix in suffixes:
        value = metrics.get(f"{prefix}_{suffix}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[suffix] = value
    return result or None
