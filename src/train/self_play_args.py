from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Literal

from src.mcts.mcts_v2 import MCTSArgs
from src.train.self_play_health import SelfPlayHealthThresholds


SearchBackend = Literal["python", "cpp"]


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
    professional_value_loss_weight: float = 1.0
    self_play_value_loss_weight: float = 1.0
    search_health: SelfPlayHealthThresholds = SelfPlayHealthThresholds()
    search_backend: SearchBackend = "python"
    native_worker_threads: int = 1

    def __post_init__(self) -> None:
        if self.start_iteration < 0:
            raise ValueError("Start iteration cannot be negative")
        if (
            self.professional_games_training_iterations < 0
            or self.self_play_training_iterations < 0
        ):
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
            raise ValueError(
                "Choose either a replay snapshot or professional replay seeding"
            )
        if self.start_iteration > 0 and not self.training_run_id:
            raise ValueError(
                "Resumed training requires a checkpoint training run identifier"
            )
        if self.start_iteration == 0 and self.expected_replay_generation is not None:
            raise ValueError("A new run cannot expect an existing replay generation")
        if (
            self.expected_replay_generation is not None
            and not 0 <= self.expected_replay_generation <= self.start_iteration
        ):
            raise ValueError(
                "Expected replay generation is outside the checkpoint history"
            )
        if not 0 <= self.professional_replay_fraction <= 1:
            raise ValueError(
                "Professional replay fraction must be between zero and one"
            )
        if (
            self.professional_value_loss_weight < 0
            or self.self_play_value_loss_weight < 0
        ):
            raise ValueError("Value loss weights cannot be negative")
        if self.seed < 0:
            raise ValueError("Training seed cannot be negative")
        if self.search_backend not in ("python", "cpp"):
            raise ValueError("Search backend must be 'python' or 'cpp'")
        if isinstance(self.native_worker_threads, bool):
            raise TypeError("Native worker threads must be an integer")
        try:
            native_worker_threads = operator.index(self.native_worker_threads)
        except TypeError as error:
            raise TypeError("Native worker threads must be an integer") from error
        if native_worker_threads < 1:
            raise ValueError("Native worker threads must be positive")
        object.__setattr__(self, "native_worker_threads", native_worker_threads)
