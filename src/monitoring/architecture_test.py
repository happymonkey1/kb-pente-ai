from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from typing import Any

from src.monitoring.architecture import RunManifestStore


class RunManifestStoreTest(unittest.TestCase):
    def test_matches_run_and_derives_architecture_metrics(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry_path = root / "metrics" / "training.jsonl"
            telemetry_path.parent.mkdir()
            telemetry_path.write_text("", encoding="utf-8")
            model_root = root / "model"
            model_root.mkdir()
            (model_root / "run-manifest-step-0.json").write_text(
                json.dumps(
                    _manifest(
                        telemetry_path,
                        training_run_id="run-1",
                        start_iteration=0,
                    )
                ),
                encoding="utf-8",
            )
            store = RunManifestStore([root])

            architecture = store.architecture_for(telemetry_path, "run-1")

        self.assertEqual(
            {
                "available": True,
                "model": "PenteNet",
                "config": {
                    "board_size": 5,
                    "action_size": 25,
                    "input_planes": 4,
                    "residual_blocks": 1,
                    "channels": 8,
                    "value_hidden_size": 16,
                },
                "metrics": {
                    "parameter_count": 3_253,
                    "estimated_fp32_bytes": 13_012,
                    "multiply_accumulates_per_position": 38_266,
                    "estimated_flops_per_position": 76_532,
                    "parameterized_layer_count": 13,
                    "trunk_activation_values_per_position": 200,
                    "parameters_by_stage": {
                        "stem": 312,
                        "residual_tower": 1_200,
                        "policy_head": 1_297,
                        "value_head": 444,
                    },
                    "multiply_accumulates_by_stage": {
                        "stem": 7_200,
                        "residual_tower": 28_800,
                        "policy_head": 1_650,
                        "value_head": 616,
                    },
                },
                "runtime": {
                    "device": "cpu",
                    "device_name": None,
                    "torch": "2.8.0",
                    "torch_cuda": None,
                    "compiled": False,
                },
                "ruleset": "freestyle",
                "manifest": {
                    "created_at_utc": "2026-08-30T00:00:00+00:00",
                    "start_iteration": 0,
                },
            },
            architecture,
        )

    def test_prefers_latest_matching_resume_manifest(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            telemetry_path = root / "training.jsonl"
            telemetry_path.write_text("", encoding="utf-8")
            for start_iteration, channels in ((0, 8), (4, 16)):
                value = _manifest(
                    telemetry_path,
                    training_run_id="run-1",
                    start_iteration=start_iteration,
                )
                model = value["model"]
                assert isinstance(model, dict)
                model["num_channels"] = channels
                (root / f"run-manifest-step-{start_iteration}.json").write_text(
                    json.dumps(value),
                    encoding="utf-8",
                )
            store = RunManifestStore([root])

            architecture = store.architecture_for(telemetry_path, "run-1")

        manifest = architecture["manifest"]
        config = architecture["config"]
        assert isinstance(manifest, dict)
        assert isinstance(config, dict)
        self.assertEqual(4, manifest["start_iteration"])
        self.assertEqual(16, config["channels"])

    def test_reports_missing_manifest_without_guessing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            telemetry_path = Path(temporary_directory) / "training.jsonl"
            telemetry_path.write_text("", encoding="utf-8")
            store = RunManifestStore([temporary_directory])

            architecture = store.architecture_for(telemetry_path, "missing")

        self.assertEqual(
            {
                "available": False,
                "reason": "No matching run manifest was found.",
            },
            architecture,
        )

def _manifest(
    telemetry_path: Path,
    *,
    training_run_id: str,
    start_iteration: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": "2026-08-30T00:00:00+00:00",
        "training_run_id": training_run_id,
        "start_iteration": start_iteration,
        "model": {
            "board_size": 5,
            "action_size": 25,
            "input_planes": 4,
            "num_res_blocks": 1,
            "num_channels": 8,
            "hidden_fc_size": 16,
        },
        "outputs": {"telemetry": str(telemetry_path)},
        "runtime": {
            "device": "cpu",
            "device_name": None,
            "torch": "2.8.0",
            "torch_cuda": None,
            "compiled": False,
        },
        "program_arguments": {"ruleset": "freestyle"},
    }


if __name__ == "__main__":
    unittest.main()
