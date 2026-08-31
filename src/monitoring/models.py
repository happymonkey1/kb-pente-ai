from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast, TypeAlias


MetricValue: TypeAlias = int | float | str | bool | None
TELEMETRY_SCHEMA_VERSION = 1
REPLAY_SAMPLE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ArtifactIssue:
    code: str
    message: str
    line: int | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
        }
        if self.line is not None:
            result["line"] = self.line
        return result


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    schema_version: int
    timestamp_unix: float
    event: str
    step: int
    metrics: dict[str, MetricValue]

    @classmethod
    def from_object(cls, value: object) -> TelemetryRecord:
        record = _object(value, "Telemetry record")
        schema_version = _integer(record.get("schema_version"), "schema_version")
        if schema_version != TELEMETRY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported telemetry schema version {schema_version}; "
                f"expected {TELEMETRY_SCHEMA_VERSION}"
            )

        timestamp_unix = _finite_number(record.get("timestamp_unix"), "timestamp_unix")
        event = _non_empty_string(record.get("event"), "event")
        step = _integer(record.get("step"), "step")
        if step < 0:
            raise ValueError("step cannot be negative")

        raw_metrics = _object(record.get("metrics"), "metrics")
        metrics: dict[str, MetricValue] = {}
        for key, metric in raw_metrics.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Metric names must be non-empty strings")
            if not _is_metric_value(metric):
                raise ValueError(f"Metric {key} must be a finite scalar value")
            metrics[key] = cast(MetricValue, metric)

        return cls(
            schema_version=schema_version,
            timestamp_unix=timestamp_unix,
            event=event,
            step=step,
            metrics=metrics,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "timestamp_unix": self.timestamp_unix,
            "event": self.event,
            "step": self.step,
            "metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class ReplaySample:
    schema_version: int
    run_id: str
    game_id: str
    recorded_at_unix: float
    board_size: int
    ruleset: str
    actions: tuple[int, ...]
    winner: int | None
    win_reason: str | None

    @classmethod
    def from_object(cls, value: object) -> ReplaySample:
        record = _object(value, "Replay sample")
        schema_version = _integer(record.get("schema_version"), "schema_version")
        if schema_version != REPLAY_SAMPLE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported replay schema version {schema_version}; "
                f"expected {REPLAY_SAMPLE_SCHEMA_VERSION}"
            )

        run_id = _non_empty_string(record.get("run_id"), "run_id")
        game_id = _non_empty_string(record.get("game_id"), "game_id")
        recorded_at_unix = _finite_number(
            record.get("recorded_at_unix"),
            "recorded_at_unix",
        )
        board_size = _integer(record.get("board_size"), "board_size")
        if not 5 <= board_size <= 25:
            raise ValueError("board_size must be between 5 and 25")
        ruleset = _non_empty_string(record.get("ruleset"), "ruleset")

        raw_actions = record.get("actions")
        if not isinstance(raw_actions, list):
            raise ValueError("actions must be an array")
        if len(raw_actions) > 4096:
            raise ValueError("actions contains more than 4096 moves")
        actions: list[int] = []
        for index, action in enumerate(raw_actions):
            parsed_action = _integer(action, f"actions[{index}]")
            if not 0 <= parsed_action < board_size * board_size:
                raise ValueError(f"actions[{index}] is outside the board")
            actions.append(parsed_action)

        raw_winner = record.get("winner")
        winner = None if raw_winner is None else _integer(raw_winner, "winner")
        if winner is not None and winner not in (-1, 1):
            raise ValueError("winner must be 1, -1, or null")

        win_reason = record.get("win_reason")
        if win_reason is not None and not isinstance(win_reason, str):
            raise ValueError("win_reason must be a string or null")

        return cls(
            schema_version=schema_version,
            run_id=run_id,
            game_id=game_id,
            recorded_at_unix=recorded_at_unix,
            board_size=board_size,
            ruleset=ruleset,
            actions=tuple(actions),
            winner=winner,
            win_reason=win_reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "game_id": self.game_id,
            "recorded_at_unix": self.recorded_at_unix,
            "board_size": self.board_size,
            "ruleset": self.ruleset,
            "actions": list(self.actions),
            "winner": self.winner,
            "win_reason": self.win_reason,
        }

    def metadata(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "game_id": self.game_id,
            "recorded_at_unix": self.recorded_at_unix,
            "board_size": self.board_size,
            "ruleset": self.ruleset,
            "move_count": len(self.actions),
            "winner": self.winner,
            "win_reason": self.win_reason,
        }


def _object(value: object, name: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _is_metric_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False
