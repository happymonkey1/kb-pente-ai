from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import os
import pickle
import tempfile
from typing import Iterable, Iterator

import numpy as np

from src.artifacts import (
    POSITION_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
    TRAINING_EXAMPLE_SCHEMA_VERSION,
)
from src.train.training_example import TrainingExample


class ReplaySource(str, Enum):
    PROFESSIONAL = "professional"
    SELF_PLAY = "self_play"


@dataclass(frozen=True, slots=True)
class ReplayEntry:
    example: TrainingExample
    generation: int
    source: ReplaySource


@dataclass(frozen=True, slots=True)
class ReplayStats:
    size: int
    capacity: int
    oldest_age: int
    newest_age: int
    mean_age: float
    unique_positions: int
    professional_positions: int
    self_play_positions: int


@dataclass(frozen=True, slots=True)
class ReplaySample:
    examples: list[TrainingExample]
    professional_positions: int
    self_play_positions: int


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    schema_version: int
    capacity: int
    generation: int
    training_run_id: str
    board_size: int
    ruleset: str
    position_schema_version: int
    training_example_schema_version: int
    entries: tuple[ReplayEntry, ...]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self._entries: deque[ReplayEntry] = deque(maxlen=capacity)
        self.snapshot_generation: int | None = None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[TrainingExample]:
        return (entry.example for entry in self._entries)

    def extend(
        self,
        examples: Iterable[TrainingExample],
        generation: int,
        source: ReplaySource,
    ) -> None:
        if generation < 0:
            raise ValueError("Replay generation cannot be negative")
        self._entries.extend(
            ReplayEntry(example, generation, source)
            for example in examples
        )

    def examples(self) -> list[TrainingExample]:
        return [entry.example for entry in self._entries]

    def sample(
        self,
        count: int,
        rng: np.random.Generator,
    ) -> list[TrainingExample]:
        if count < 1:
            raise ValueError("Replay sample count must be positive")
        if not self._entries:
            raise ValueError("Cannot sample an empty replay buffer")
        entries = list(self._entries)
        return [entry.example for entry in self._sample_entries(entries, count, rng)]

    def sample_mixed(
        self,
        count: int,
        rng: np.random.Generator,
        professional_fraction: float,
    ) -> ReplaySample:
        if count < 1:
            raise ValueError("Replay sample count must be positive")
        if not 0 <= professional_fraction <= 1:
            raise ValueError("Professional replay fraction must be between zero and one")
        if not self._entries:
            raise ValueError("Cannot sample an empty replay buffer")
        professional = [
            entry
            for entry in self._entries
            if entry.source is ReplaySource.PROFESSIONAL
        ]
        self_play = [
            entry
            for entry in self._entries
            if entry.source is ReplaySource.SELF_PLAY
        ]
        if professional and self_play:
            professional_count = round(count * professional_fraction)
            self_play_count = count - professional_count
        elif professional:
            professional_count = count
            self_play_count = 0
        else:
            professional_count = 0
            self_play_count = count

        selected = self._sample_entries(professional, professional_count, rng)
        selected.extend(self._sample_entries(self_play, self_play_count, rng))
        ordering = rng.permutation(len(selected))
        selected = [selected[int(index)] for index in ordering]
        return ReplaySample(
            examples=[entry.example for entry in selected],
            professional_positions=professional_count,
            self_play_positions=self_play_count,
        )

    def stats(self, current_generation: int) -> ReplayStats:
        if not self._entries:
            return ReplayStats(0, self.capacity, 0, 0, 0.0, 0, 0, 0)
        ages = np.array(
            [current_generation - entry.generation for entry in self._entries],
            dtype=np.int64,
        )
        if np.any(ages < 0):
            raise ValueError("Current generation predates a replay entry")
        unique_positions = len({entry.example.position.state_key() for entry in self._entries})
        professional_positions = sum(
            entry.source is ReplaySource.PROFESSIONAL
            for entry in self._entries
        )
        return ReplayStats(
            size=len(self._entries),
            capacity=self.capacity,
            oldest_age=int(ages.max()),
            newest_age=int(ages.min()),
            mean_age=float(ages.mean()),
            unique_positions=unique_positions,
            professional_positions=professional_positions,
            self_play_positions=len(self._entries) - professional_positions,
        )

    @staticmethod
    def _sample_entries(
        entries: list[ReplayEntry],
        count: int,
        rng: np.random.Generator,
    ) -> list[ReplayEntry]:
        if count == 0:
            return []
        if not entries:
            raise ValueError("Requested replay source has no entries")
        indices = rng.choice(
            len(entries),
            size=count,
            replace=count > len(entries),
        )
        return [entries[int(index)] for index in np.asarray(indices).reshape(-1)]

    def save(
        self,
        path: str,
        generation: int,
        training_run_id: str,
        board_size: int,
        ruleset: str,
    ) -> None:
        if generation < 0:
            raise ValueError("Replay snapshot generation cannot be negative")
        if not training_run_id:
            raise ValueError("Replay training run identifier cannot be empty")
        if board_size < 5:
            raise ValueError("Replay board size must be at least five")
        if not ruleset:
            raise ValueError("Replay ruleset cannot be empty")
        snapshot = ReplaySnapshot(
            schema_version=REPLAY_SCHEMA_VERSION,
            capacity=self.capacity,
            generation=generation,
            training_run_id=training_run_id,
            board_size=board_size,
            ruleset=ruleset,
            position_schema_version=POSITION_SCHEMA_VERSION,
            training_example_schema_version=TRAINING_EXAMPLE_SCHEMA_VERSION,
            entries=tuple(self._entries),
        )
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as stream:
                temporary_path = stream.name
                pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, path)
            self.snapshot_generation = generation
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @classmethod
    def load(
        cls,
        path: str,
        expected_training_run_id: str,
        expected_board_size: int,
        expected_ruleset: str,
    ) -> ReplayBuffer:
        with open(path, "rb") as stream:
            snapshot = pickle.load(stream)
        if not isinstance(snapshot, ReplaySnapshot):
            raise ValueError("Replay snapshot is a legacy or unknown format")
        if snapshot.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"Replay schema {snapshot.schema_version} is incompatible with schema {REPLAY_SCHEMA_VERSION}"
            )
        if snapshot.training_run_id != expected_training_run_id:
            raise ValueError("Replay snapshot belongs to a different training run")
        if snapshot.board_size != expected_board_size:
            raise ValueError(
                f"Replay board size {snapshot.board_size} does not match {expected_board_size}"
            )
        if snapshot.ruleset != expected_ruleset:
            raise ValueError(
                f"Replay ruleset {snapshot.ruleset!r} does not match {expected_ruleset!r}"
            )
        if (
            snapshot.position_schema_version != POSITION_SCHEMA_VERSION
            or snapshot.training_example_schema_version
            != TRAINING_EXAMPLE_SCHEMA_VERSION
        ):
            raise ValueError("Replay position or training-example schema is incompatible")
        replay = cls(snapshot.capacity)
        replay._entries.extend(snapshot.entries)
        replay.snapshot_generation = snapshot.generation
        return replay
