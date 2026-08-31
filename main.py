from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import uuid

import numpy as np
import torch

from src.device import select_torch_device
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.mcts.mcts_v2 import MCTS, MCTSArgs
from src.model.model_v1 import PenteNet
from src.monitoring.replay_writer import JsonlReplaySampleSink
from src.run_manifest import write_run_manifest
from src.telemetry import JsonlMetricSink
from src.train.arena import Arena
from src.train.nnet_player import NNetPlayer
from src.train.random_player import RandomPlayer
from src.train.self_play import SelfPlayTrainer, SelfPlayTrainerArgs
from src.train.self_play_health import SelfPlayHealthThresholds


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate the corrected Pente policy/value network")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--infer", action="store_true", help="Evaluate a model instead of training")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA when available")
    parser.add_argument("--compile", action="store_true", help="Compile the model on supported CUDA systems")
    parser.add_argument("--infer-games", type=int, default=20)
    parser.add_argument("--model-dir", default="pente-model-v2")
    parser.add_argument("--model", help="Compatible schema-v2 checkpoint to load")
    parser.add_argument("--infer-mcts", action="store_true")
    parser.add_argument("--mcts-sim", type=int, default=64)
    parser.add_argument("--temp-threshold", type=int, default=15)
    parser.add_argument("--batch-games", type=int, default=512)
    parser.add_argument("--active-games", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--arena", action="store_true", help="Measure latest against the prior iteration")
    parser.add_argument("--num-arena-games", type=int, default=40)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--arena-opening-plies", type=int, default=4)
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument(
        "--ruleset",
        choices=[ruleset.value for ruleset in PenteRuleset],
        default=PenteRuleset.STANDARD.value,
    )
    parser.add_argument("--professional-iterations", type=int, default=0)
    parser.add_argument("--self-play-iterations", type=int, default=150)
    parser.add_argument("--raw-dataset", default="data/pente_dataset.txt")
    parser.add_argument("--processed-dataset", default="data/pente-dataset-v3.pkl")
    parser.add_argument("--force-dataset-processing", action="store_true")
    parser.add_argument("--max-training-examples", type=int, default=500_000)
    parser.add_argument("--professional-replay-fraction", type=float, default=0.25)
    parser.add_argument("--professional-value-loss-weight", type=float, default=1.0)
    parser.add_argument("--self-play-value-loss-weight", type=float, default=1.0)
    parser.add_argument("--minimum-batch-occupancy", type=float, default=0.0)
    parser.add_argument("--minimum-mean-root-children", type=float, default=0.0)
    parser.add_argument("--maximum-search-collapse-rate", type=float, default=1.0)
    parser.add_argument("--maximum-invalid-policy-fallbacks", type=int, default=0)
    parser.add_argument("--maximum-zero-visit-fallbacks", type=int, default=-1)
    parser.add_argument("--model-blocks", type=int, default=4)
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--model-hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learner-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--telemetry-file", default="metrics/training.jsonl")
    parser.add_argument("--replay-sample-file")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--replay-checkpoint-interval", type=int, default=5)
    parser.add_argument("--resume-replay")
    parser.add_argument("--seed-replay-from-professional", action="store_true")
    return parser


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    handler = RotatingFileHandler(
        "kb-pente-ai.log",
        mode="a",
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=(handler, logging.StreamHandler()),
    )


def main() -> int:
    program_args = build_parser().parse_args()
    configure_logging(program_args.debug)
    np.random.seed(program_args.seed)
    torch.manual_seed(program_args.seed)

    ruleset = PenteRuleset.parse(program_args.ruleset)
    if program_args.professional_iterations > 0 and program_args.board_size != 19:
        raise ValueError("The checked-in professional dataset requires a 19 by 19 board")

    device = select_torch_device(program_args.gpu)
    use_cuda = device.type == "cuda"
    game = PenteGame(program_args.board_size, ruleset=ruleset)
    net = PenteNet(
        device,
        board_size=program_args.board_size,
        action_size=game.get_action_size(),
        num_res_blocks=program_args.model_blocks,
        num_channels=program_args.model_channels,
        hidden_fc_size=program_args.model_hidden_size,
    )
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=program_args.learning_rate,
        weight_decay=1e-4,
    )
    start_iteration = 0
    training_run_id: str | None = None
    expected_replay_generation: int | None = None
    if program_args.model:
        if program_args.infer:
            PenteNet.load_checkpoint_from_path(
                program_args.model,
                net,
                expected_ruleset=ruleset.value,
            )
        else:
            resume_state = PenteNet.load_training_checkpoint_from_path(
                program_args.model,
                net,
                optimizer,
                expected_ruleset=ruleset.value,
            )
            start_iteration = resume_state.iteration
            training_run_id = resume_state.training_run_id
            expected_replay_generation = resume_state.replay_snapshot_generation
    elif not program_args.infer:
        training_run_id = uuid.uuid4().hex

    trainer_args = SelfPlayTrainerArgs(
        start_iteration=start_iteration,
        professional_games_training_iterations=program_args.professional_iterations,
        self_play_training_iterations=program_args.self_play_iterations,
        temp_threshold=program_args.temp_threshold,
        mcts_args=MCTSArgs(num_simulations=program_args.mcts_sim),
        watch_training_raw_dataset_filepath=program_args.raw_dataset,
        watch_training_processed_dataset_filepath=program_args.processed_dataset,
        force_watch_training_raw_dataset_processing=program_args.force_dataset_processing,
        eval_iteration_interval=program_args.eval_interval,
        num_arena_games=program_args.num_arena_games,
        batch_size=program_args.batch_size,
        batch_games=program_args.batch_games,
        active_games=program_args.active_games,
        checkpoint_dir=program_args.model_dir,
        should_checkpoint=not program_args.no_checkpoint,
        max_training_examples=program_args.max_training_examples,
        debug=program_args.debug,
        should_use_arena=program_args.arena,
        seed=program_args.seed,
        replay_checkpoint_interval=program_args.replay_checkpoint_interval,
        learner_steps_per_iteration=program_args.learner_steps,
        arena_opening_plies=program_args.arena_opening_plies,
        resume_replay_filepath=program_args.resume_replay,
        seed_replay_from_professional=program_args.seed_replay_from_professional,
        professional_replay_fraction=program_args.professional_replay_fraction,
        professional_value_loss_weight=program_args.professional_value_loss_weight,
        self_play_value_loss_weight=program_args.self_play_value_loss_weight,
        training_run_id=training_run_id,
        expected_replay_generation=expected_replay_generation,
        search_health=SelfPlayHealthThresholds(
            minimum_steady_state_batch_occupancy=(
                program_args.minimum_batch_occupancy
            ),
            minimum_mean_root_children_visited=(
                program_args.minimum_mean_root_children
            ),
            maximum_search_collapse_rate=(
                program_args.maximum_search_collapse_rate
            ),
            maximum_invalid_policy_fallbacks=(
                program_args.maximum_invalid_policy_fallbacks
            ),
            maximum_zero_visit_fallbacks=(
                program_args.maximum_zero_visit_fallbacks
            ),
        ),
    )

    if program_args.compile:
        if not use_cuda:
            raise ValueError("--compile requires an available CUDA device selected with --gpu")
        net.compile(fullgraph=True)
        torch.set_float32_matmul_precision("high")

    logger.info(
        "Initialized board=%s ruleset=%s device=%s parameters=%s",
        program_args.board_size,
        ruleset.value,
        device,
        net.get_parameter_count(),
    )
    if program_args.infer:
        return run_inference(program_args, game, net, trainer_args)

    assert training_run_id is not None
    metric_sink = JsonlMetricSink(program_args.telemetry_file, training_run_id)
    replay_sample_sink = (
        JsonlReplaySampleSink(program_args.replay_sample_file)
        if program_args.replay_sample_file
        else None
    )
    trainer = SelfPlayTrainer(
        game,
        net,
        optimizer,
        device,
        trainer_args,
        metric_sink,
        replay_sample_sink,
    )
    manifest_root = (
        Path(program_args.model_dir)
        if not program_args.no_checkpoint
        else Path(program_args.telemetry_file).resolve().parent
    )
    write_run_manifest(
        manifest_root / f"run-manifest-step-{start_iteration}.json",
        Path(__file__).resolve().parent,
        [sys.executable, *sys.argv],
        trainer.training_run_id,
        start_iteration,
        device,
        program_args.compile,
        net,
        trainer_args,
        vars(program_args),
        program_args.telemetry_file,
    )
    trainer.train()
    return 0


def run_inference(
    program_args: argparse.Namespace,
    game: PenteGame,
    net: PenteNet,
    trainer_args: SelfPlayTrainerArgs,
) -> int:
    if not program_args.model:
        raise ValueError("Inference requires --model")
    net.eval()
    mcts = (
        MCTS(game, net, trainer_args.mcts_args, np.random.default_rng(program_args.seed))
        if program_args.infer_mcts
        else None
    )
    arena = Arena(
        NNetPlayer(net, mcts, "network"),
        RandomPlayer(np.random.default_rng(program_args.seed + 1)),
        game,
        opening_plies=program_args.arena_opening_plies,
        rng=np.random.default_rng(program_args.seed + 2),
    )
    stats = arena.play_games(program_args.infer_games)
    logger.info(
        "Evaluation complete: network_wins=%s random_wins=%s draws=%s average_moves=%s",
        stats.p1_wins,
        stats.p2_wins,
        stats.draws,
        stats.avg_moves,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
