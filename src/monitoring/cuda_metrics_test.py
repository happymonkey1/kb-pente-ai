import unittest
from unittest.mock import patch

import torch

from src.monitoring.cuda_metrics import CudaMetricSampler, measure_cuda_operation


class CudaMetricSamplerTest(unittest.TestCase):
    @patch("src.monitoring.cuda_metrics.torch.cuda.max_memory_reserved", return_value=2048)
    @patch("src.monitoring.cuda_metrics.torch.cuda.max_memory_allocated", return_value=1024)
    @patch("src.monitoring.cuda_metrics.torch.cuda.reset_peak_memory_stats")
    @patch("src.monitoring.cuda_metrics.torch.cuda.memory_usage", side_effect=[20, 40])
    @patch("src.monitoring.cuda_metrics.torch.cuda.utilization", side_effect=[50, 90])
    def test_collects_utilization_and_memory_summary(
        self,
        utilization: object,
        memory_usage: object,
        reset_peak_memory_stats: object,
        max_memory_allocated: object,
        max_memory_reserved: object,
    ) -> None:
        sampler = CudaMetricSampler(torch.device("cuda"))

        with patch("src.monitoring.cuda_metrics.threading.Thread"):
            sampler.start()
            sampler._sample()
            sampler._sample()
            metrics = sampler.stop()

        self.assertEqual(2, metrics.samples)
        self.assertEqual(70.0, metrics.mean_utilization_percent)
        self.assertEqual(88.0, metrics.p95_utilization_percent)
        self.assertEqual(90, metrics.max_utilization_percent)
        self.assertEqual(30.0, metrics.mean_device_memory_percent)
        self.assertEqual(40, metrics.max_device_memory_percent)
        self.assertEqual(1024, metrics.peak_allocated_bytes)
        self.assertEqual(2048, metrics.peak_reserved_bytes)
        self.assertEqual(0, metrics.sampling_errors)

    def test_rejects_non_cuda_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            CudaMetricSampler(torch.device("cpu"))

    def test_cpu_operation_runs_without_cuda_metrics(self) -> None:
        result, metrics = measure_cuda_operation(torch.device("cpu"), lambda: 17)

        self.assertEqual(17, result)
        self.assertIsNone(metrics)


if __name__ == "__main__":
    unittest.main()
