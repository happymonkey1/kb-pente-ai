from __future__ import annotations

import copy
from dataclasses import dataclass
import os
import shutil
import tempfile
import time
import uuid

import numpy as np
import torch

from src.evaluation.supervised import (
    evaluate_supervised_examples,
    supervised_evaluation_metrics,
)
from src.game.pente.pente_game import PenteGame
from src.mcts.batched import BatchedSearchTelemetry
from src.mcts.mcts_v2 import MCTSArgs
from src.model.model_v1 import PenteNet
from src.monitoring.cuda_metrics import measure_cuda_operation
from src.telemetry import MetricSink, NullMetricSink
from src.train.iteration_evaluation import evaluate_training_iteration
from src.train.learner import ModelTrainingStats, train_policy_value_model
from src.train.profession_game_loader import ProfessionGameLoader
from src.train.replay_buffer import ReplayBuffer, ReplaySource
from src.train.self_play_generation import (
    PlayedGame,
    SelfPlayGenerator,
    finalize_training_examples,
)
from src.train.self_play_metrics import collect_self_play_metrics
from src.train.training_example import TrainingExample


@dataclass(frozen=True, slots=True)
class SelfPlayTrainerArgs:
    start_iteration: int
    professional_games_training_iterations: int
    self_play_training_iterations: int
    temp_threshold: int
    mcts_args: MCTSArgs
    watch_training_raw_dataset_filepath: str
    watch_training_processed_dataset_filepath: str
    force_watch_training_raw_dataset_processing: bool
    eval_iteration_interval: int = 5
    num_arena_games: int = 40
    batch_size: int = 512
    batch_games: int = 1
    active_games: int = 128
    checkpoint_dir: str = "checkpoints"
    should_checkpoint: bool = True
    max_training_examples: int = 500_000
    debug: bool = False
    should_use_arena: bool = False
    seed: int = 0
    augment_training: bool = True
    replay_checkpoint_interval: int = 5
    learner_steps_per_iteration: int = 256
    arena_opening_plies: int = 4
    resume_replay_filepath: str | None = None
    seed_replay_from_professional: bool = False
    training_run_id: str | None = None
    expected_replay_generation: int | None = None
    professional_replay_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.start_iteration < 0:
            raise ValueError("Start iteration cannot be negative")
        if self.professional_games_training_iterations < 0 or self.self_play_training_iterations < 0:
            raise ValueError("Training iteration counts cannot be negative")
        if self.temp_threshold < 0:
            raise ValueError("Temperature threshold cannot be negative")
        if self.eval_iteration_interval < 1:
            raise ValueError("Evaluation interval must be positive")
        if self.batch_size < 1 or self.batch_games < 1 or self.active_games < 1:
            raise ValueError("Training batch and game counts must be positive")
        if self.max_training_examples < 1:
            raise ValueError("Replay capacity must be positive")
        if self.replay_checkpoint_interval < 1:
            raise ValueError("Replay checkpoint interval must be positive")
        if self.learner_steps_per_iteration < 1:
            raise ValueError("Learner steps per iteration must be positive")
        if self.arena_opening_plies < 0:
            raise ValueError("Arena opening plies cannot be negative")
        if self.start_iteration == 0 and self.resume_replay_filepath is not None:
            raise ValueError("A replay snapshot can only resume a nonzero iteration")
        if self.start_iteration == 0 and self.seed_replay_from_professional:
            raise ValueError("Professional replay seeding is only valid when resuming")
        if self.resume_replay_filepath is not None and self.seed_replay_from_professional:
            raise ValueError("Choose either a replay snapshot or professional replay seeding")
        if self.start_iteration > 0 and not self.training_run_id:
            raise ValueError("Resumed training requires a checkpoint training run identifier")
        if self.start_iteration == 0 and self.expected_replay_generation is not None:
            raise ValueError("A new run cannot expect an existing replay generation")
        if (
            self.expected_replay_generation is not None
            and not 0 <= self.expected_replay_generation <= self.start_iteration
        ):
            raise ValueError("Expected replay generation is outside the checkpoint history")
        if not 0 <= self.professional_replay_fraction <= 1:
            raise ValueError("Professional replay fraction must be between zero and one")


class SelfPlayTrainer:
    def __init__(
        self,
        game: PenteGame,
        net: PenteNet,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        args: SelfPlayTrainerArgs,
        metric_sink: MetricSink | None = None,
    ) -> None:
        self.args = args
        self._replay_snapshot_path = os.path.join(
            args.checkpoint_dir,
            "replay-latest.pkl",
        )
        self.training_run_id = args.training_run_id or uuid.uuid4().hex
        self._replay_source: str | None
        resume_replay_path = args.resume_replay_filepath or self._default_resume_replay_path(
            args.expected_replay_generation,
        )
        if (
            args.start_iteration > 0
            and args.resume_replay_filepath is not None
            and not os.path.exists(resume_replay_path)
        ):
            raise FileNotFoundError(f"Replay snapshot not found: {resume_replay_path}")
        if args.start_iteration > 0 and os.path.exists(resume_replay_path):
            self.replay = ReplayBuffer.load(
                resume_replay_path,
                expected_training_run_id=self.training_run_id,
                expected_board_size=game.get_board_size(),
                expected_ruleset=game.ruleset.value,
            )
            if self.replay.capacity != args.max_training_examples:
                raise ValueError("Replay snapshot capacity does not match trainer configuration")
            assert self.replay.snapshot_generation is not None
            if self.replay.snapshot_generation > args.start_iteration:
                raise ValueError("Replay snapshot is newer than the resumed model checkpoint")
            if (
                args.expected_replay_generation is not None
                and self.replay.snapshot_generation
                != args.expected_replay_generation
            ):
                raise ValueError("Replay generation does not match the model checkpoint")
            self._replay_source = os.path.abspath(resume_replay_path)
        elif args.start_iteration > 0 and not args.seed_replay_from_professional:
            raise ValueError(
                "Resuming training requires a compatible replay snapshot or explicit "
                "professional replay seeding"
            )
        else:
            self.replay = ReplayBuffer(args.max_training_examples)
            self._replay_source = None
        self.net = net
        self.game = game
        self.optimizer = optimizer
        self.device = device
        self.metric_sink = metric_sink if metric_sink is not None else NullMetricSink()
        self.rng = np.random.default_rng(args.seed)
        self.previous_net = copy.deepcopy(self.net)
        self.professional_validation_examples: list[TrainingExample] = []

    def play_game(self) -> PlayedGame:
        self.net.eval()
        return self._self_play_generator().play_game()

    def build_self_play_training_examples(
        self,
    ) -> tuple[list[TrainingExample], list[PlayedGame], list[BatchedSearchTelemetry]]:
        self.net.eval()
        games, batches = self._self_play_generator().play_games(
            self.args.batch_games,
            self.args.active_games,
        )
        examples = [example for game in games for example in game.examples]
        return examples, games, batches

    def _self_play_generator(self) -> SelfPlayGenerator:
        return SelfPlayGenerator(
            self.game,
            self.net,
            self.args.mcts_args,
            self.args.temp_threshold,
            self.rng,
        )

    def _default_resume_replay_path(self, expected_generation: int | None) -> str:
        if expected_generation is not None:
            versioned_path = self._replay_generation_path(expected_generation)
            if os.path.exists(versioned_path):
                return versioned_path
        return self._replay_snapshot_path

    def _replay_generation_path(self, generation: int) -> str:
        return os.path.join(self.args.checkpoint_dir, f"replay-{generation}.pkl")

    def load_professional_training_examples(self) -> list[TrainingExample]:
        loader = ProfessionGameLoader(
            raw_filepath=self.args.watch_training_raw_dataset_filepath,
            processed_filepath=self.args.watch_training_processed_dataset_filepath,
            board_size=self.game.get_board_size(),
            force=self.args.force_watch_training_raw_dataset_processing,
            ruleset=self.game.ruleset,
        )
        examples = loader.load_games()
        self.professional_validation_examples = loader.validation_examples
        if loader.last_stats is not None:
            stats = loader.last_stats
            self.metric_sink.emit(
                "professional_data",
                0,
                {
                    "accepted_games": stats.accepted_games,
                    "rejected_games": stats.rejected_games,
                    "accepted_positions": stats.accepted_positions,
                    "deduplicated_positions": stats.deduplicated_positions,
                    "non_terminal_games": stats.non_terminal_games,
                    "training_games": stats.training_games,
                    "validation_games": stats.validation_games,
                    "training_positions": stats.training_positions,
                    "validation_positions": stats.validation_positions,
                },
            )
        return examples

    def train(self) -> None:
        total_iterations = (
            self.args.professional_games_training_iterations
            + self.args.self_play_training_iterations
        )
        professional_examples: list[TrainingExample] | None = None

        if self.args.start_iteration > 0:
            if self.args.seed_replay_from_professional and not self.replay:
                professional_examples = self.load_professional_training_examples()
                if len(professional_examples) > self.replay.capacity:
                    raise ValueError(
                        "Replay capacity cannot hold the requested professional seed"
                    )
                self.replay.extend(
                    professional_examples,
                    generation=self.args.start_iteration,
                    source=ReplaySource.PROFESSIONAL,
                )
                self.metric_sink.emit(
                    "replay_seed",
                    self.args.start_iteration,
                    {
                        "source": "professional",
                        "positions": len(professional_examples),
                        "capacity": self.replay.capacity,
                    },
                )
            elif self.replay.snapshot_generation is not None:
                self.metric_sink.emit(
                    "replay_resume",
                    self.args.start_iteration,
                    {
                        "source": self._replay_source,
                        "snapshot_generation": self.replay.snapshot_generation,
                        "model_generation": self.args.start_iteration,
                        "generation_lag": (
                            self.args.start_iteration - self.replay.snapshot_generation
                        ),
                        "positions": len(self.replay),
                    },
                )

        if self.args.should_checkpoint and self.args.start_iteration == 0:
            self._save_checkpoint(0)

        for iteration in range(self.args.start_iteration, total_iterations):
            iteration_started = time.perf_counter()
            self.previous_net.load_state_dict(self.net.state_dict())
            self.previous_net.eval()

            is_professional = iteration < self.args.professional_games_training_iterations
            generation_started = time.perf_counter()
            played_games: list[PlayedGame] = []
            search_batches: list[BatchedSearchTelemetry] = []
            if is_professional:
                if professional_examples is None:
                    professional_examples = self.load_professional_training_examples()
                new_examples = professional_examples if iteration == 0 else []
                if iteration == 0 and self.professional_validation_examples:
                    baseline = evaluate_supervised_examples(
                        self.net,
                        self.game,
                        self.professional_validation_examples,
                        self.args.batch_size,
                    )
                    self.metric_sink.emit(
                        "professional_validation_baseline",
                        0,
                        supervised_evaluation_metrics(baseline),
                    )
            else:
                generation_result, generation_cuda_metrics = measure_cuda_operation(
                    self.device,
                    self.build_self_play_training_examples,
                )
                new_examples, played_games, search_batches = generation_result
            generation_seconds = time.perf_counter() - generation_started

            self.replay.extend(
                new_examples,
                generation=iteration + 1,
                source=(
                    ReplaySource.PROFESSIONAL
                    if is_professional
                    else ReplaySource.SELF_PLAY
                ),
            )
            if not self.replay:
                raise RuntimeError("No training examples are available")
            learner_sample = self.replay.sample_mixed(
                self.args.batch_size * self.args.learner_steps_per_iteration,
                self.rng,
                self.args.professional_replay_fraction,
            )
            learner_examples = learner_sample.examples

            training_started = time.perf_counter()
            training_stats, learner_cuda_metrics = measure_cuda_operation(
                self.device,
                lambda: self.train_model(learner_examples),
            )
            training_seconds = time.perf_counter() - training_started
            replay_stats = self.replay.stats(iteration + 1)

            metrics: dict[str, int | float] = {
                "new_positions": len(new_examples),
                "new_unique_positions": len({example.position.state_key() for example in new_examples}),
                "replay_positions": replay_stats.size,
                "replay_unique_positions": replay_stats.unique_positions,
                "replay_professional_positions": replay_stats.professional_positions,
                "replay_self_play_positions": replay_stats.self_play_positions,
                "replay_oldest_age": replay_stats.oldest_age,
                "replay_mean_age": replay_stats.mean_age,
                "learner_positions": len(learner_examples),
                "learner_professional_positions": learner_sample.professional_positions,
                "learner_self_play_positions": learner_sample.self_play_positions,
                "generation_seconds": generation_seconds,
                "training_seconds": training_seconds,
                "iteration_seconds": time.perf_counter() - iteration_started,
                "loss": training_stats.total_loss / training_stats.num_batches,
                "policy_loss": training_stats.total_policy_loss / training_stats.num_batches,
                "policy_kl": training_stats.total_policy_kl / training_stats.num_batches,
                "value_loss": training_stats.total_value_loss / training_stats.num_batches,
                "value_absolute_error": (
                    training_stats.total_value_absolute_error / training_stats.num_batches
                ),
                "value_bias": training_stats.total_value_bias / training_stats.num_batches,
            }
            metrics.update(training_stats.value_metrics.to_metrics())
            if not is_professional and generation_cuda_metrics is not None:
                metrics.update(generation_cuda_metrics.to_metrics("self_play_gpu"))
            if learner_cuda_metrics is not None:
                metrics.update(learner_cuda_metrics.to_metrics("learner_gpu"))
            if played_games:
                metrics.update(
                    collect_self_play_metrics(
                        played_games,
                        search_batches,
                        generation_seconds,
                    )
                )
            if is_professional and self.professional_validation_examples:
                validation = evaluate_supervised_examples(
                    self.net,
                    self.game,
                    self.professional_validation_examples,
                    self.args.batch_size,
                )
                metrics.update(supervised_evaluation_metrics(validation))
            self.metric_sink.emit("training_iteration", iteration + 1, metrics)

            if (
                self.args.should_use_arena
                and self.args.num_arena_games > 0
                and (iteration + 1) % self.args.eval_iteration_interval == 0
            ):
                evaluate_training_iteration(
                    self.game,
                    self.previous_net,
                    self.net,
                    self.args.mcts_args,
                    self.metric_sink,
                    iteration + 1,
                    self.args.num_arena_games,
                    self.args.arena_opening_plies,
                    self.args.debug,
                    self.args.seed,
                )
            if self.args.should_checkpoint:
                self._save_checkpoint(iteration + 1)

    def train_model(self, training_examples: list[TrainingExample]) -> ModelTrainingStats:
        return train_policy_value_model(
            self.net,
            self.optimizer,
            self.device,
            self.game,
            training_examples,
            self.args.batch_size,
            self.args.augment_training,
        )

    def _save_checkpoint(self, iteration: int) -> None:
        if (
            not os.path.exists(self._replay_snapshot_path)
            or iteration <= 1
            or iteration % self.args.replay_checkpoint_interval == 0
        ):
            versioned_replay_path = self._replay_generation_path(iteration)
            self.replay.save(
                versioned_replay_path,
                generation=iteration,
                training_run_id=self.training_run_id,
                board_size=self.game.get_board_size(),
                ruleset=self.game.ruleset.value,
            )
            _replace_with_link_or_copy(
                versioned_replay_path,
                self._replay_snapshot_path,
            )
        state: dict[str, object] = {
            "iteration": iteration,
            "state_dict": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metadata": self.net.checkpoint_metadata(self.game.ruleset.value),
            "replay_snapshot_generation": self.replay.snapshot_generation,
            "training_run_id": self.training_run_id,
        }
        PenteNet.save_checkpoint(
            state,
            self.args.checkpoint_dir,
            self.net.get_checkpoint_file_name(iteration),
        )
        PenteNet.save_checkpoint(state, self.args.checkpoint_dir, "latest.pth.tar")


def _replace_with_link_or_copy(source: str, destination: str) -> None:
    directory = os.path.dirname(os.path.abspath(destination))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(dir=directory)
    os.close(file_descriptor)
    os.unlink(temporary_path)
    try:
        try:
            os.link(source, temporary_path)
        except OSError:
            shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
