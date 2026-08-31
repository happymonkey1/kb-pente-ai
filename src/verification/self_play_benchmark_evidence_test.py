from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.verification.self_play_benchmark_evidence import (
    INFORMATIONAL_REASON,
    load_historical_self_play_reference,
)


def _metrics(device: str = "cpu") -> dict[str, object]:
    values: dict[str, object] = {
        "device_type": device,
        "games": 8,
        "games_per_second": 2.5,
        "positions_per_second": 240.0,
        "leaf_evaluations_per_second": 480.0,
        "mean_inference_batch_size": 6.0,
        "p95_inference_batch_size": 8.0,
        "max_inference_batch_size": 9.0,
        "duplicate_leaf_rate": 0.25,
        "steady_state_mean_batch_occupancy": 0.875,
        "active_game_target": 8,
        "self_play_cpu_mean_process_utilization_percent": 42.0,
        "cpu_logical_core_count": 16,
    }
    if device == "cuda":
        values.update(
            {
                "self_play_gpu_mean_utilization_percent": 71.0,
                "self_play_gpu_p95_utilization_percent": 89.0,
                "self_play_gpu_max_utilization_percent": 95.0,
            }
        )
    return values


def _record(
    step: int = 1,
    *,
    device: str = "cpu",
    run_id: str | None = "run-id",
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "timestamp_unix": 1000.0 + step,
        "run_id": run_id,
        "event": "training_iteration",
        "step": step,
        "metrics": _metrics(device) if metrics is None else metrics,
    }


def _write(path: Path, records: list[object], *, leading_blank: bool = False) -> bytes:
    encoded = "\n".join(json.dumps(record, separators=(",", ":")) for record in records)
    if leading_blank:
        encoded = "\n" + encoded
    payload = (encoded + "\n").encode("utf-8")
    path.write_bytes(payload)
    return payload


class HistoricalSelfPlayEvidenceTest(unittest.TestCase):
    def test_selects_last_complete_record_and_preserves_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            first = _record(step=2)
            incomplete = _record(step=3, metrics={"device_type": "cpu"})
            last = _record(step=4)
            payload = _write(
                path,
                [{"event": "learner", "metrics": {}}, first, incomplete, last],
                leading_blank=True,
            )

            reference = load_historical_self_play_reference(path)

        self.assertEqual(str(path.resolve()), reference.source_path)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), reference.source_sha256)
        self.assertEqual(5, reference.record_line)
        self.assertEqual(3, reference.record_index)
        self.assertEqual("run-id", reference.run_id)
        self.assertEqual(4, reference.step)
        self.assertEqual(1004.0, reference.timestamp_unix)
        self.assertEqual(8, reference.games)

    def test_optional_values_are_none_and_report_is_serializable_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            _write(path, [_record(run_id=None)])
            reference = load_historical_self_play_reference(path)

        self.assertIsNone(reference.run_id)
        self.assertIsNone(reference.generation_seconds)
        self.assertIsNone(reference.simulations)
        self.assertIsNone(reference.model_blocks)
        self.assertIsNone(reference.model_channels)
        self.assertIsNone(reference.model_hidden_size)
        self.assertIsNone(reference.gpu_mean_utilization_percent)
        self.assertIsNone(reference.gpu_p95_utilization_percent)
        self.assertIsNone(reference.gpu_max_utilization_percent)
        self.assertEqual("informational", reference.comparison_kind)
        self.assertEqual(INFORMATIONAL_REASON, reference.comparison_reason)
        encoded = json.dumps(reference.to_dict(), sort_keys=True, allow_nan=False)
        self.assertIn('"comparison_kind": "informational"', encoded)

    def test_preserves_optional_scalar_context_and_cuda_utilization(self) -> None:
        values = _metrics("cuda")
        values.update(
            {
                "generation_seconds": 12.5,
                "simulations": 64,
                "model_blocks": 6,
                "model_channels": 128,
                "model_hidden_size": 256,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.jsonl"
            _write(path, [_record(device="cuda", metrics=values)])
            reference = load_historical_self_play_reference(path)

        self.assertEqual(12.5, reference.generation_seconds)
        self.assertEqual(64, reference.simulations)
        self.assertEqual(6, reference.model_blocks)
        self.assertEqual(128, reference.model_channels)
        self.assertEqual(256, reference.model_hidden_size)
        self.assertEqual(71.0, reference.gpu_mean_utilization_percent)
        self.assertEqual(89.0, reference.gpu_p95_utilization_percent)
        self.assertEqual(95.0, reference.gpu_max_utilization_percent)

    def test_rejects_cuda_record_without_complete_utilization(self) -> None:
        values = _metrics("cuda")
        del values["self_play_gpu_p95_utilization_percent"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.jsonl"
            _write(path, [_record(device="cuda", metrics=values)])
            with self.assertRaisesRegex(ValueError, "no complete valid"):
                load_historical_self_play_reference(path)

    def test_skips_incomplete_cuda_record_after_complete_cpu_record(self) -> None:
        values = _metrics("cuda")
        del values["self_play_gpu_max_utilization_percent"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            _write(path, [_record(step=1), _record(step=2, device="cuda", metrics=values)])
            reference = load_historical_self_play_reference(path)

        self.assertEqual("cpu", reference.device_type)
        self.assertEqual(1, reference.step)

    def test_rejects_malformed_json_and_non_object_with_line_numbers(self) -> None:
        cases = (
            (b'{"event":"training_iteration"}\n{"broken"\n', "line 2.*malformed JSON"),
            (b'[]\n', "line 1.*JSON object"),
        )
        for payload, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.jsonl"
                path.write_bytes(payload)
                with self.assertRaisesRegex(ValueError, message):
                    load_historical_self_play_reference(path)

    def test_rejects_incomplete_only_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.jsonl"
            _write(path, [_record(metrics={"device_type": "cpu"})])
            with self.assertRaisesRegex(ValueError, "no complete valid"):
                load_historical_self_play_reference(path)

    def test_rejects_invalid_numeric_and_range_values(self) -> None:
        cases = (
            ("games", -1, "games.*positive"),
            ("games_per_second", 0, "games_per_second.*positive"),
            ("duplicate_leaf_rate", 1.1, "duplicate_leaf_rate.*at most"),
            ("steady_state_mean_batch_occupancy", -0.1, "occupancy.*at least"),
            ("self_play_cpu_mean_process_utilization_percent", 101.0, "cpu.*at most"),
        )
        for name, value, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                values = deepcopy(_metrics())
                values[name] = value
                path = Path(directory) / "invalid-values.jsonl"
                _write(path, [_record(metrics=values)])
                with self.assertRaisesRegex(ValueError, message):
                    load_historical_self_play_reference(path)

    def test_rejects_non_regular_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "regular file"):
                load_historical_self_play_reference(path)


if __name__ == "__main__":
    unittest.main()
