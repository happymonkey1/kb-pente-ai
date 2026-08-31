from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable, TypeVar

import numpy as np
import torch


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CudaMetrics:
    samples: int
    mean_utilization_percent: float
    p95_utilization_percent: float
    max_utilization_percent: int
    mean_device_memory_percent: float
    max_device_memory_percent: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    sampling_errors: int

    def to_metrics(self, prefix: str = "gpu") -> dict[str, int | float]:
        return {
            f"{prefix}_utilization_samples": self.samples,
            f"{prefix}_mean_utilization_percent": self.mean_utilization_percent,
            f"{prefix}_p95_utilization_percent": self.p95_utilization_percent,
            f"{prefix}_max_utilization_percent": self.max_utilization_percent,
            f"{prefix}_mean_device_memory_percent": self.mean_device_memory_percent,
            f"{prefix}_max_device_memory_percent": self.max_device_memory_percent,
            f"{prefix}_peak_memory_allocated_bytes": self.peak_allocated_bytes,
            f"{prefix}_peak_memory_reserved_bytes": self.peak_reserved_bytes,
            f"{prefix}_utilization_sampling_errors": self.sampling_errors,
        }


class CudaMetricSampler:
    """Samples device-wide CUDA utilization while one training iteration runs."""

    def __init__(self, device: torch.device, interval_seconds: float = 0.25) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA metrics require a CUDA device")
        if interval_seconds <= 0:
            raise ValueError("CUDA metric sampling interval must be positive")
        self._device = device
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._utilization: list[int] = []
        self._device_memory: list[int] = []
        self._sampling_errors = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("CUDA metric sampler is already running")
        self._stop_event.clear()
        torch.cuda.reset_peak_memory_stats(self._device)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> CudaMetrics:
        if self._thread is None:
            raise RuntimeError("CUDA metric sampler is not running")
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        if not self._utilization:
            self._sample()
        return CudaMetrics(
            samples=len(self._utilization),
            mean_utilization_percent=_mean(self._utilization),
            p95_utilization_percent=_percentile(self._utilization, 95),
            max_utilization_percent=max(self._utilization, default=0),
            mean_device_memory_percent=_mean(self._device_memory),
            max_device_memory_percent=max(self._device_memory, default=0),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self._device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self._device),
            sampling_errors=self._sampling_errors,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._sample()

    def _sample(self) -> None:
        try:
            self._utilization.append(int(torch.cuda.utilization(self._device)))
            self._device_memory.append(int(torch.cuda.memory_usage(self._device)))
        except Exception:
            # NVML exception classes vary by driver and PyTorch version.
            self._sampling_errors += 1


def _mean(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentile(values: list[int], percentile: int) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def measure_cuda_operation(
    device: torch.device,
    operation: Callable[[], T],
) -> tuple[T, CudaMetrics | None]:
    if device.type != "cuda":
        return operation(), None

    sampler = CudaMetricSampler(device)
    sampler.start()
    try:
        result = operation()
    except BaseException:
        sampler.stop()
        raise
    return result, sampler.stop()
