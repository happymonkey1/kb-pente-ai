import unittest
from unittest.mock import patch

from src.monitoring.cpu_metrics import CpuMetricSampler, measure_cpu_operation


class CpuMetricSamplerTest(unittest.TestCase):
    @patch("src.monitoring.cpu_metrics._resident_memory_bytes", side_effect=[1000, 3000])
    @patch("src.monitoring.cpu_metrics.os.cpu_count", return_value=4)
    @patch(
        "src.monitoring.cpu_metrics.time.perf_counter",
        side_effect=[10.0, 11.0, 13.0],
    )
    @patch(
        "src.monitoring.cpu_metrics.time.process_time",
        side_effect=[2.0, 4.0, 5.0],
    )
    def test_collects_normalized_utilization_and_resident_memory(
        self,
        process_time: object,
        perf_counter: object,
        cpu_count: object,
        resident_memory: object,
    ) -> None:
        sampler = CpuMetricSampler()

        with patch("src.monitoring.cpu_metrics.threading.Thread"):
            sampler.start()
            sampler._sample()
            metrics = sampler.stop()

        self.assertEqual(4, metrics.logical_core_count)
        self.assertEqual(2, metrics.utilization_samples)
        self.assertEqual(31.25, metrics.mean_process_utilization_percent)
        self.assertEqual(48.125, metrics.p95_process_utilization_percent)
        self.assertEqual(50.0, metrics.max_process_utilization_percent)
        self.assertEqual(2000, metrics.mean_resident_memory_bytes)
        self.assertEqual(3000, metrics.peak_resident_memory_bytes)
        self.assertEqual(0, metrics.sampling_errors)

    @patch("src.monitoring.cpu_metrics._resident_memory_bytes", return_value=1000)
    @patch("src.monitoring.cpu_metrics.os.cpu_count", return_value=1)
    @patch(
        "src.monitoring.cpu_metrics.time.perf_counter",
        side_effect=[1.0, 2.0],
    )
    @patch(
        "src.monitoring.cpu_metrics.time.process_time",
        side_effect=[1.0, 4.0],
    )
    def test_clamps_measurement_noise_to_one_hundred_percent(
        self,
        process_time: object,
        perf_counter: object,
        cpu_count: object,
        resident_memory: object,
    ) -> None:
        sampler = CpuMetricSampler()

        with patch("src.monitoring.cpu_metrics.threading.Thread"):
            sampler.start()
            metrics = sampler.stop()

        self.assertEqual(100.0, metrics.max_process_utilization_percent)

    @patch("src.monitoring.cpu_metrics._resident_memory_bytes", side_effect=OSError)
    def test_sampling_failure_is_reported(self, resident_memory: object) -> None:
        sampler = CpuMetricSampler()

        with patch("src.monitoring.cpu_metrics.threading.Thread"):
            sampler.start()
            metrics = sampler.stop()

        self.assertEqual(0, metrics.utilization_samples)
        self.assertEqual(1, metrics.sampling_errors)

    def test_operation_exception_stops_sampler_thread(self) -> None:
        def fail() -> None:
            raise RuntimeError("operation failed")

        with self.assertRaisesRegex(RuntimeError, "operation failed"):
            measure_cpu_operation(fail)

    def test_metric_names_match_monitoring_contract(self) -> None:
        _, metrics = measure_cpu_operation(lambda: 17)

        encoded = metrics.to_metrics("learner")

        self.assertEqual(
            {
                "learner_cpu_utilization_samples",
                "learner_cpu_mean_process_utilization_percent",
                "learner_cpu_p95_process_utilization_percent",
                "learner_cpu_max_process_utilization_percent",
                "learner_cpu_mean_resident_memory_bytes",
                "learner_cpu_peak_resident_memory_bytes",
                "learner_cpu_sampling_errors",
            },
            set(encoded),
        )


if __name__ == "__main__":
    unittest.main()
