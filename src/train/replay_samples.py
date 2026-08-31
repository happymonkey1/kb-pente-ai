from __future__ import annotations

from collections.abc import Sequence
import hashlib
import time
from typing import Protocol

from src.monitoring.models import REPLAY_SAMPLE_SCHEMA_VERSION, ReplaySample
from src.train.self_play_generation import PlayedGame


MAX_REPLAY_SAMPLES_PER_ITERATION = 8


class ReplaySampleSink(Protocol):
    def emit(self, sample: ReplaySample) -> None:
        """Persist one validated, browser-safe completed-game sample."""


def emit_replay_samples(
    sink: ReplaySampleSink,
    games: Sequence[PlayedGame],
    run_id: str,
    iteration: int,
    board_size: int,
    ruleset: str,
    maximum_samples: int = MAX_REPLAY_SAMPLES_PER_ITERATION,
) -> int:
    if maximum_samples < 0:
        raise ValueError("Maximum replay samples cannot be negative")
    selected = sorted(games, key=_trajectory_digest)[:maximum_samples]
    recorded_at_unix = time.time()
    for index, game in enumerate(selected):
        digest = _trajectory_digest(game).hex()[:16]
        sink.emit(
            ReplaySample(
                schema_version=REPLAY_SAMPLE_SCHEMA_VERSION,
                run_id=run_id,
                game_id=f"iteration-{iteration}-sample-{index}-{digest}",
                recorded_at_unix=recorded_at_unix,
                board_size=board_size,
                ruleset=ruleset,
                actions=game.actions,
                winner=game.winner,
                win_reason=game.win_reason,
            )
        )
    return len(selected)


def _trajectory_digest(game: PlayedGame) -> bytes:
    encoded = bytearray()
    for action in game.actions:
        encoded.extend(action.to_bytes(2, byteorder="little", signed=False))
    return hashlib.sha256(encoded).digest()
