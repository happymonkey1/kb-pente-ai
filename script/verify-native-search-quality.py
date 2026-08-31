#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Sequence

import torch

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTSArgs
from src.mcts.native_backend import load_native_extension
from src.model.model_v1 import PenteNet
from src.run_manifest import source_fingerprint
from src.verification.search_quality import (
    evaluate_search_quality,
    summarize_statistical_parity,
)
from src.verification.search_quality_reports import (
    BackendSearchQualityReport,
    StatisticalParityCriteria,
    StatisticalParityReport,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Python and C++ Pente search quality on identical inputs",
    )
    parser.add_argument("current_checkpoint")
    parser.add_argument("previous_checkpoint")
    parser.add_argument("--gpu", action="store_true", help="Run both backends on CUDA")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument(
        "--ruleset",
        choices=[ruleset.value for ruleset in PenteRuleset],
        default=PenteRuleset.STANDARD.value,
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--native-search-threads", type=int, default=1)
    parser.add_argument("--model-blocks", type=int, default=1)
    parser.add_argument("--model-channels", type=int, default=16)
    parser.add_argument("--model-hidden-size", type=int, default=64)
    parser.add_argument(
        "--maximum-decisive-win-rate-difference",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--maximum-paired-win-rate-difference",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--maximum-tactical-accuracy-difference",
        type=float,
        default=1.0 / 6.0,
    )
    parser.add_argument(
        "--require-wilson-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    game, device, mcts_args, criteria = _validate_args(args)
    current_path = Path(args.current_checkpoint).resolve()
    previous_path = Path(args.previous_checkpoint).resolve()

    # The extension is checked once, after ordinary validation and before
    # loading models or starting either backend. The accepted API still owns
    # its normal backend construction and intentionally receives "cpp".
    load_native_extension()
    started = time.perf_counter()
    current, current_iteration = _load_model(current_path, device, game, args)
    previous, previous_iteration = _load_model(previous_path, device, game, args)
    python_report = evaluate_search_quality(
        current,
        previous,
        game,
        mcts_args,
        games=args.games,
        opening_plies=args.opening_plies,
        seed=args.seed,
        search_backend="python",
        native_worker_threads=args.native_search_threads,
    )
    cpp_report = evaluate_search_quality(
        current,
        previous,
        game,
        mcts_args,
        games=args.games,
        opening_plies=args.opening_plies,
        seed=args.seed,
        search_backend="cpp",
        native_worker_threads=args.native_search_threads,
    )
    parity = summarize_statistical_parity(python_report, cpp_report, criteria)
    output = _build_output(
        current_path,
        previous_path,
        current_iteration,
        previous_iteration,
        current,
        game,
        device,
        args,
        criteria,
        python_report,
        cpp_report,
        parity,
        time.perf_counter() - started,
    )
    print(_stable_json(output))
    return 0 if parity.passed else 1


def _validate_args(
    args: argparse.Namespace,
) -> tuple[PenteGame, torch.device, MCTSArgs, StatisticalParityCriteria]:
    current_path = Path(args.current_checkpoint)
    previous_path = Path(args.previous_checkpoint)
    if not current_path.is_file():
        raise ValueError(f"Current checkpoint not found: {current_path}")
    if not previous_path.is_file():
        raise ValueError(f"Previous checkpoint not found: {previous_path}")
    if args.gpu and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested without an available CUDA device")
    if args.board_size < 9:
        raise ValueError("Search quality requires a board of at least 9 by 9")
    if args.games < 2:
        raise ValueError("games must be at least two")
    if args.games % 2:
        raise ValueError("games must be even for paired openings")
    if args.simulations < 1:
        raise ValueError("simulations must be positive")
    if args.opening_plies < 0:
        raise ValueError("opening_plies cannot be negative")
    if args.seed < 0:
        raise ValueError("seed cannot be negative")
    if args.native_search_threads < 1:
        raise ValueError("native search threads must be positive")
    if args.model_blocks < 1:
        raise ValueError("model blocks must be positive")
    if args.model_channels < 1:
        raise ValueError("model channels must be positive")
    if args.model_hidden_size < 1:
        raise ValueError("model hidden size must be positive")

    ruleset = PenteRuleset.parse(args.ruleset)
    game = PenteGame(args.board_size, ruleset=ruleset)
    device = torch.device("cuda" if args.gpu else "cpu")
    mcts_args = MCTSArgs(num_simulations=args.simulations)
    criteria = StatisticalParityCriteria(
        maximum_decisive_win_rate_difference=args.maximum_decisive_win_rate_difference,
        maximum_paired_win_rate_difference=args.maximum_paired_win_rate_difference,
        maximum_tactical_accuracy_difference=args.maximum_tactical_accuracy_difference,
        require_wilson_overlap=args.require_wilson_overlap,
    )
    return game, device, mcts_args, criteria


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
    current_path: Path,
    previous_path: Path,
    current_iteration: int,
    previous_iteration: int,
    current: PenteNet,
    game: PenteGame,
    device: torch.device,
    args: argparse.Namespace,
    criteria: StatisticalParityCriteria,
    python_report: BackendSearchQualityReport,
    cpp_report: BackendSearchQualityReport,
    parity: StatisticalParityReport,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "checkpoints": {
            "current": {
                "path": str(current_path),
                "sha256": _sha256(current_path),
                "iteration": current_iteration,
            },
            "previous": {
                "path": str(previous_path),
                "sha256": _sha256(previous_path),
                "iteration": previous_iteration,
            },
        },
        "model": asdict(current.config),
        "configuration": {
            "board_size": game.get_board_size(),
            "ruleset": game.ruleset.value,
            "games": args.games,
            "simulations": args.simulations,
            "opening_plies": args.opening_plies,
            "seed": args.seed,
            "native_search_threads": args.native_search_threads,
        },
        "repository": {
            "commit": _git_commit(repository_root),
            "source_fingerprint_sha256": source_fingerprint(repository_root),
        },
        "runtime": _runtime_metadata(device),
        "criteria": asdict(criteria),
        "reports": {
            "python": asdict(python_report),
            "cpp": asdict(cpp_report),
        },
        "parity": asdict(parity),
        "elapsed_seconds": elapsed_seconds,
    }


def _runtime_metadata(device: torch.device) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and cuda_available
            else None
        ),
    }


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


def _stable_json(value: dict[str, object]) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    raise SystemExit(main())
