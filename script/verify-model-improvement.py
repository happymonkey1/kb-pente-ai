#!/usr/bin/env python

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

import torch

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.native_backend import load_native_extension
from src.model.model_v1 import PenteNet
from src.verification.model_match import ModelMatchCriteria, evaluate_model_match


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare candidate and frozen Pente checkpoints on paired openings",
    )
    parser.add_argument("candidate_checkpoint")
    parser.add_argument("baseline_checkpoint")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument(
        "--ruleset",
        choices=[ruleset.value for ruleset in PenteRuleset],
        default=PenteRuleset.STANDARD.value,
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument(
        "--search-backend",
        choices=("python", "cpp"),
        default="python",
        help="MCTS search implementation (Python by default)",
    )
    parser.add_argument(
        "--native-search-threads",
        type=int,
        default=1,
        help="Native MCTS worker threads",
    )
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--model-blocks", type=int, default=1)
    parser.add_argument("--model-channels", type=int, default=16)
    parser.add_argument("--model-hidden-size", type=int, default=64)
    parser.add_argument("--minimum-games", type=int, default=100)
    parser.add_argument("--minimum-confidence-lower", type=float, default=0.5)
    return parser


def _preflight_native_search(search_backend: str, simulations: int) -> None:
    if search_backend == "cpp" and simulations > 0:
        load_native_extension()


def _validate_native_search_threads(native_search_threads: int) -> None:
    if native_search_threads < 1:
        raise ValueError("Native search threads must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested without an available CUDA device")
    _validate_native_search_threads(args.native_search_threads)
    _preflight_native_search(args.search_backend, args.simulations)
    device = torch.device("cuda" if args.gpu else "cpu")
    ruleset = PenteRuleset.parse(args.ruleset)
    game = PenteGame(args.board_size, ruleset=ruleset)

    def load(path: str) -> tuple[PenteNet, int]:
        model = PenteNet(
            device,
            board_size=args.board_size,
            action_size=game.get_action_size(),
            num_res_blocks=args.model_blocks,
            num_channels=args.model_channels,
            hidden_fc_size=args.model_hidden_size,
        )
        iteration = PenteNet.load_checkpoint_from_path(
            path,
            model,
            expected_ruleset=ruleset.value,
        )
        return model, iteration

    candidate, candidate_iteration = load(args.candidate_checkpoint)
    baseline, baseline_iteration = load(args.baseline_checkpoint)
    report = evaluate_model_match(
        candidate,
        baseline,
        game,
        games=args.games,
        seed=args.seed,
        simulations=args.simulations,
        opening_plies=args.opening_plies,
        criteria=ModelMatchCriteria(
            minimum_games=args.minimum_games,
            minimum_paired_win_rate_95pct_lower=args.minimum_confidence_lower,
        ),
        search_backend=args.search_backend,
        native_worker_threads=args.native_search_threads,
    )
    output = report.to_dict()
    output["candidate_checkpoint"] = str(Path(args.candidate_checkpoint).resolve())
    output["baseline_checkpoint"] = str(Path(args.baseline_checkpoint).resolve())
    output["candidate_iteration"] = candidate_iteration
    output["baseline_iteration"] = baseline_iteration
    output["simulations"] = args.simulations
    output["search_backend"] = args.search_backend
    output["native_search_threads"] = args.native_search_threads
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
