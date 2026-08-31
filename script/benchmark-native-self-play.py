#!/usr/bin/env python

"""Run the loaded-model Python/native self-play benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time

import torch

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.native_backend import load_native_extension
from src.model.model_v1 import PenteNet
from src.run_manifest import source_fingerprint
from src.verification.self_play_benchmark import run_self_play_benchmark
from src.verification.self_play_benchmark_evidence import (
    HistoricalSelfPlayReference,
    load_historical_self_play_reference,
)
from src.verification.self_play_benchmark_reports import (
    SelfPlayBenchmarkConfig,
    SelfPlayBenchmarkReport,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Python and native batched Pente self-play",
    )
    parser.add_argument("checkpoint", help="Checkpoint file to load")
    parser.add_argument("--gpu", action="store_true", help="Run the benchmark on CUDA")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument(
        "--ruleset",
        choices=[ruleset.value for ruleset in PenteRuleset],
        default=PenteRuleset.STANDARD.value,
    )
    parser.add_argument("--model-blocks", type=int, default=4)
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--model-hidden-size", type=int, default=256)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--max-active-games", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--temp-threshold", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--native-search-threads", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument(
        "--minimum-steady-state-batch-occupancy",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--minimum-native-games-per-second-ratio",
        type=float,
        default=2.0,
    )
    parser.add_argument("--maximum-invalid-policy-fallbacks", type=int, default=0)
    parser.add_argument("--maximum-zero-visit-fallbacks", type=int, default=0)
    parser.add_argument("--maximum-cpu-sampling-errors", type=int, default=0)
    parser.add_argument("--maximum-cuda-sampling-errors", type=int, default=0)
    parser.add_argument(
        "--historical-path",
        type=Path,
        default=None,
        help="Optional historical self-play JSONL telemetry",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    checkpoint_path, game, device, config, historical = _validate_args(args)

    # Keep native loading explicit and after all ordinary input validation.
    load_native_extension()
    model, iteration = _load_model(checkpoint_path, device, game, args)
    report = run_self_play_benchmark(model, game, config, device)
    output = _build_output(
        checkpoint_path,
        iteration,
        model,
        device,
        config,
        report,
        historical,
        time.perf_counter() - started,
    )
    print(_stable_json(output))
    return 0 if report.passed else 1


def _validate_args(
    args: argparse.Namespace,
) -> tuple[
    Path,
    PenteGame,
    torch.device,
    SelfPlayBenchmarkConfig,
    HistoricalSelfPlayReference | None,
]:
    checkpoint_path = _regular_file(Path(args.checkpoint), "Checkpoint")
    historical = (
        load_historical_self_play_reference(args.historical_path)
        if args.historical_path is not None
        else None
    )
    if args.gpu and not torch.cuda.is_available():
        raise ValueError("CUDA benchmark requested without an available CUDA device")
    for name in ("model_blocks", "model_channels", "model_hidden_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    ruleset = PenteRuleset.parse(args.ruleset)
    game = PenteGame(args.board_size, ruleset=ruleset)
    config = SelfPlayBenchmarkConfig(
        board_size=game.get_board_size(),
        ruleset=game.ruleset,
        games=args.games,
        max_active_games=args.max_active_games,
        simulations=args.simulations,
        temp_threshold=args.temp_threshold,
        repeats=args.repeats,
        warmup_batches=args.warmup_batches,
        seed=args.seed,
        native_worker_threads=args.native_search_threads,
        torch_threads=args.torch_threads,
        minimum_steady_state_batch_occupancy=(
            args.minimum_steady_state_batch_occupancy
        ),
        minimum_native_games_per_second_ratio=(
            args.minimum_native_games_per_second_ratio
        ),
        maximum_invalid_policy_fallbacks=args.maximum_invalid_policy_fallbacks,
        maximum_zero_visit_fallbacks=args.maximum_zero_visit_fallbacks,
        maximum_cpu_sampling_errors=args.maximum_cpu_sampling_errors,
        maximum_cuda_sampling_errors=args.maximum_cuda_sampling_errors,
    )
    device = torch.device("cuda" if args.gpu else "cpu")
    return checkpoint_path, game, device, config, historical


def _load_model(
    path: Path,
    device: torch.device,
    game: PenteGame,
    args: argparse.Namespace,
) -> tuple[PenteNet, int]:
    model = PenteNet(
        device,
        board_size=game.get_board_size(),
        action_size=game.get_action_size(),
        num_res_blocks=args.model_blocks,
        num_channels=args.model_channels,
        hidden_fc_size=args.model_hidden_size,
    )
    iteration = PenteNet.load_checkpoint_from_path(
        str(path),
        model,
        expected_ruleset=game.ruleset.value,
    )
    return model, iteration


def _build_output(
    checkpoint_path: Path,
    iteration: int,
    model: PenteNet,
    device: torch.device,
    config: SelfPlayBenchmarkConfig,
    report: SelfPlayBenchmarkReport,
    historical: HistoricalSelfPlayReference | None,
    elapsed_seconds: float,
) -> dict[str, object]:
    model_config = asdict(model.config)
    return {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "iteration": iteration,
        },
        "model": {
            "config": model_config,
            "parameter_count": model.get_parameter_count(),
        },
        "configuration": _json_value(asdict(config)),
        "repository": {
            "commit": _git_commit(repository_root),
            "source_fingerprint_sha256": source_fingerprint(repository_root),
        },
        "runtime": _runtime_metadata(device),
        "report": _json_value(asdict(report)),
        "historical_reference": (
            None if historical is None else historical.to_dict()
        ),
        "elapsed_seconds": elapsed_seconds,
    }


def _runtime_metadata(device: torch.device) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    cuda_selected = device.type == "cuda" and cuda_available
    properties = (
        torch.cuda.get_device_properties(device) if cuda_selected else None
    )
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if cuda_selected else None
        ),
        "cuda_device_capability": (
            torch.cuda.get_device_capability(device) if cuda_selected else None
        ),
        "cuda_total_memory_bytes": (
            properties.total_memory if properties is not None else None
        ),
    }


def _regular_file(path: Path, description: str) -> Path:
    resolved = path.resolve()
    try:
        is_regular = stat.S_ISREG(resolved.stat().st_mode)
    except OSError as error:
        raise ValueError(f"{description} must be a regular file: {resolved}") from error
    if not is_regular:
        raise ValueError(f"{description} must be a regular file: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit or None


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _stable_json(value: Mapping[str, object]) -> str:
    return json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    raise SystemExit(main())
