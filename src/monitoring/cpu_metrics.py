from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Callable, TypeVar

import numpy as np


T = TypeVar("T")
_STATM_PATH = Path("/proc/self/statm")


@dataclass(frozen=True, slots=True)
class CpuMetrics:
    logical_core_count: int
    utilization_samples: int
    mean_process_utilization_percent: float
    p95_process_utilization_percent: float
    max_process_utilization_percent: float
    mean_resident_memory_bytes: int
    peak_resident_memory_bytes: int
    sampling_errors: int

    def to_metrics(self, prefix: str) -> dict[str, int | float]:
        return {
            f"{prefix}_cpu_utilization_samples": self.utilization_samples,
            f"{prefix}_cpu_mean_process_utilization_percent": (
                self.mean_process_utilization_percent
            ),
            f"{prefix}_cpu_p95_process_utilization_percent": (
                self.p95_process_utilization_percent
            ),
            f"{prefix}_cpu_max_process_utilization_percent": (
                self.max_process_utilization_percent
            ),
            f"{prefix}_cpu_mean_resident_memory_bytes": (
                self.mean_resident_memory_bytes
            ),
            f"{prefix}_cpu_peak_resident_memory_bytes": (
                self.peak_resident_memory_bytes
            ),
            f"{prefix}_cpu_sampling_errors": self.sampling_errors,
        }


class CpuMetricSampler:
    """Samples normalized CPU use and RSS for the current process."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        if interval_seconds <= 0:
            raise ValueError("CPU metric sampling interval must be positive")
        self._interval_seconds = interval_seconds
        self._logical_core_count = max(1, os.cpu_count() or 1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._utilization: list[float] = []
        self._resident_memory: list[int] = []
        self._sampling_errors = 0
        self._previous_process_time = 0.0
        self._previous_wall_time = 0.0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("CPU metric sampler is already running")
        self._stop_event.clear()
        self._previous_process_time = time.process_time()
        self._previous_wall_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> CpuMetrics:
        if self._thread is None:
            raise RuntimeError("CPU metric sampler is not running")
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        self._sample()
        return CpuMetrics(
            logical_core_count=self._logical_core_count,
            utilization_samples=len(self._utilization),
            mean_process_utilization_percent=_mean(self._utilization),
            p95_process_utilization_percent=_percentile(self._utilization, 95),
            max_process_utilization_percent=max(self._utilization, default=0.0),
            mean_resident_memory_bytes=_integer_mean(self._resident_memory),
            peak_resident_memory_bytes=max(self._resident_memory, default=0),
            sampling_errors=self._sampling_errors,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._sample()

    def _sample(self) -> None:
        try:
            process_time = time.process_time()
            wall_time = time.perf_counter()
            process_delta = process_time - self._previous_process_time
            wall_delta = wall_time - self._previous_wall_time
            if wall_delta <= 0:
                raise ValueError("CPU sampling clock did not advance")
            utilization = 100.0 * process_delta / (
                wall_delta * self._logical_core_count
            )
            resident_memory = _resident_memory_bytes()
        except Exception:
            self._sampling_errors += 1
            return

        self._previous_process_time = process_time
        self._previous_wall_time = wall_time
        self._utilization.append(float(np.clip(utilization, 0.0, 100.0)))
        self._resident_memory.append(resident_memory)


def _resident_memory_bytes() -> int:
    fields = _STATM_PATH.read_text(encoding="ascii").split()
    if len(fields) < 2:
        raise ValueError("Process memory status does not contain RSS")
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentile(values: list[float], percentile: int) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def _integer_mean(values: list[int]) -> int:
    return int(round(float(np.mean(values)))) if values else 0


def measure_cpu_operation(operation: Callable[[], T]) -> tuple[T, CpuMetrics]:
    sampler = CpuMetricSampler()
    sampler.start()
    try:
        result = operation()
    except BaseException:
        sampler.stop()
        raise
    return result, sampler.stop()
