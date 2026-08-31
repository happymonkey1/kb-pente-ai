#!/usr/bin/env python

import argparse
import json
from pathlib import Path
import sys

import torch

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.model.model_v1 import PenteNet
from src.verification.random_play import (
    RandomPlayCriteria,
    evaluate_network_against_random,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a policy network against seeded random play with balanced colors",
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument(
        "--ruleset",
        choices=[ruleset.value for ruleset in PenteRuleset],
        default=PenteRuleset.STANDARD.value,
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--model-blocks", type=int, default=1)
    parser.add_argument("--model-channels", type=int, default=16)
    parser.add_argument("--model-hidden-size", type=int, default=64)
    parser.add_argument("--minimum-games", type=int, default=100)
    parser.add_argument("--minimum-confidence-lower", type=float, default=0.5)
    args = parser.parse_args()
    if args.gpu and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested without an available CUDA device")
    device = torch.device("cuda" if args.gpu else "cpu")
    ruleset = PenteRuleset.parse(args.ruleset)
    game = PenteGame(args.board_size, ruleset=ruleset)
    model = PenteNet(
        device,
        board_size=args.board_size,
        action_size=game.get_action_size(),
        num_res_blocks=args.model_blocks,
        num_channels=args.model_channels,
        hidden_fc_size=args.model_hidden_size,
    )
    checkpoint_iteration = PenteNet.load_checkpoint_from_path(
        args.checkpoint,
        model,
        expected_ruleset=ruleset.value,
    )
    report = evaluate_network_against_random(
        model,
        game,
        games=args.games,
        seed=args.seed,
        opening_plies=args.opening_plies,
        criteria=RandomPlayCriteria(
            minimum_games=args.minimum_games,
            minimum_decisive_win_rate_95pct_lower=args.minimum_confidence_lower,
        ),
    )
    output = report.to_dict()
    output["checkpoint_iteration"] = checkpoint_iteration
    output["checkpoint"] = str(Path(args.checkpoint).resolve())
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
