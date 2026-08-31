#!/usr/bin/env python

import argparse
import json
from pathlib import Path
import sys

import torch

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

from src.game.pente.rules import PenteRuleset
from src.verification.batched_search_benchmark import (
    BatchedSearchBenchmarkConfig,
    run_batched_search_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare independent and batched neural MCTS on identical roots",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument(
        "--ruleset",
        choices=[ruleset.value for ruleset in PenteRuleset],
        default=PenteRuleset.STANDARD.value,
    )
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--model-blocks", type=int, default=1)
    parser.add_argument("--model-channels", type=int, default=16)
    parser.add_argument("--model-hidden-size", type=int, default=64)
    parser.add_argument("--maximum-policy-difference", type=float, default=0.05)
    parser.add_argument("--minimum-selected-action-agreement", type=float, default=1.0)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device("cuda" if args.gpu else "cpu")
    config = BatchedSearchBenchmarkConfig(
        board_size=args.board_size,
        ruleset=PenteRuleset.parse(args.ruleset),
        games=args.games,
        simulations=args.simulations,
        repeats=args.repeats,
        warmup_batches=args.warmup_batches,
        seed=args.seed,
        model_blocks=args.model_blocks,
        model_channels=args.model_channels,
        model_hidden_size=args.model_hidden_size,
        maximum_policy_difference=args.maximum_policy_difference,
        minimum_selected_action_agreement=args.minimum_selected_action_agreement,
        minimum_speedup=args.minimum_speedup,
    )
    report = run_batched_search_benchmark(config, device)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
