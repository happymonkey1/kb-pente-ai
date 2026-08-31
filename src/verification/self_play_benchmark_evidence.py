"""Immutable historical self-play benchmark evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import stat


INFORMATIONAL_REASON = (
    "Historical self-play telemetry is informational and does not participate "
    "in benchmark acceptance."
)


@dataclass(frozen=True, slots=True)
class HistoricalSelfPlayReference:
    """A complete historical self-play record and its source identity.

    The report intentionally contains no acceptance result. Historical
    telemetry can come from a different device, model, or revision, so it is
    retained as context without being used as a pass/fail criterion.
    """

    source_path: str
    source_sha256: str
    record_line: int
    record_index: int
    run_id: str | None
    step: int
    timestamp_unix: float
    device_type: str
    games: int
    games_per_second: float
    positions_per_second: float
    leaf_evaluations_per_second: float
    cpu_mean_process_utilization_percent: float | None
    cpu_logical_core_count: int | None
    gpu_mean_utilization_percent: float | None
    gpu_p95_utilization_percent: float | None
    gpu_max_utilization_percent: float | None
    mean_inference_batch_size: float
    p95_inference_batch_size: float
    max_inference_batch_size: float
    duplicate_leaf_rate: float
    steady_state_mean_batch_occupancy: float
    active_game_target: int
    generation_seconds: float | None
    simulations: int | None
    model_blocks: int | None
    model_channels: int | None
    model_hidden_size: int | None
    comparison_kind: str = "informational"
    comparison_reason: str = INFORMATIONAL_REASON

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation with stable keys."""

        return asdict(self)


def load_historical_self_play_reference(
    path: Path,
) -> HistoricalSelfPlayReference:
    """Load the last complete self-play record from strict JSONL.

    Every nonblank line must be valid UTF-8 JSON and a JSON object. Records
    for other events are parsed but not examined for metrics. Incomplete
    training records are skipped; a complete record with an invalid value
    raises an error that identifies its source line.
    """

    source = Path(path).resolve()
    try:
        source_stat = source.stat()
    except OSError as error:
        raise ValueError(
            f"Historical self-play telemetry path is not readable: {source}"
        ) from error
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(
            f"Historical self-play telemetry path must be a regular file: {source}"
        )
    try:
        content = source.read_bytes()
    except OSError as error:
        raise ValueError(
            f"Historical self-play telemetry path is not readable: {source}"
        ) from error

    selected: HistoricalSelfPlayReference | None = None
    nonblank_index = 0
    for line_number, encoded_line in enumerate(content.splitlines(), start=1):
        if not encoded_line.strip():
            continue
        try:
            line = encoded_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Historical telemetry line {line_number} is not valid UTF-8"
            ) from error
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            detail = (
                error.msg if isinstance(error, json.JSONDecodeError) else str(error)
            )
            raise ValueError(
                f"Historical telemetry line {line_number} contains malformed JSON: "
                f"{detail}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"Historical telemetry line {line_number} must be a JSON object"
            )
        if value.get("event") == "training_iteration":
            candidate = _parse_training_record(value, line_number, nonblank_index)
            if candidate is not None:
                selected = candidate
        nonblank_index += 1

    if selected is None:
        raise ValueError(
            "Historical telemetry contains no complete valid "
            f"training_iteration record: {source}"
        )
    return replace(
        selected,
        source_path=str(source),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


_REQUIRED_FIELDS = (
    "device_type",
    "games",
    "games_per_second",
    "positions_per_second",
    "leaf_evaluations_per_second",
    "mean_inference_batch_size",
    "p95_inference_batch_size",
    "max_inference_batch_size",
    "duplicate_leaf_rate",
    "steady_state_mean_batch_occupancy",
    "active_game_target",
)


def _parse_training_record(
    record: dict[object, object],
    line_number: int,
    record_index: int,
) -> HistoricalSelfPlayReference | None:
    raw_metrics = record.get("metrics")
    if not isinstance(raw_metrics, dict):
        if raw_metrics is None:
            return None
        raise _line_error(line_number, "metrics must be an object")
    metrics = raw_metrics

    missing = [
        name for name in _REQUIRED_FIELDS if name not in metrics
    ]
    if "step" not in record:
        missing.append("step")
    if "timestamp_unix" not in record:
        missing.append("timestamp_unix")
    if missing:
        return None

    device = _device_type(metrics["device_type"], line_number)
    step = _integer(record["step"], "step", line_number, nonnegative=True)
    timestamp = _number(
        record["timestamp_unix"], "timestamp_unix", line_number, minimum=0.0
    )
    parsed = {
        name: _metric(metrics[name], name, line_number)
        for name in _REQUIRED_FIELDS
        if name != "device_type"
    }
    run_id = _run_id(record.get("run_id"), line_number)

    cpu_mean = _optional_number(
        metrics,
        "self_play_cpu_mean_process_utilization_percent",
        "cpu_mean_process_utilization_percent",
        line_number,
        minimum=0.0,
        maximum=100.0,
    )
    cpu_cores = _optional_integer(
        metrics,
        "cpu_logical_core_count",
        "cpu_logical_core_count",
        line_number,
        positive=True,
    )
    gpu = {
        name: _optional_number(
            metrics,
            source_name,
            name,
            line_number,
            minimum=0.0,
            maximum=100.0,
        )
        for name, source_name in (
            (
                "gpu_mean_utilization_percent",
                "self_play_gpu_mean_utilization_percent",
            ),
            (
                "gpu_p95_utilization_percent",
                "self_play_gpu_p95_utilization_percent",
            ),
            (
                "gpu_max_utilization_percent",
                "self_play_gpu_max_utilization_percent",
            ),
        )
    }
    if device == "cuda" and any(value is None for value in gpu.values()):
        return None

    generation = _optional_number(
        metrics, "generation_seconds", "generation_seconds", line_number, minimum=0.0
    )
    simulations = _optional_context_integer(metrics, "simulations", line_number)
    model_blocks = _optional_context_integer(metrics, "model_blocks", line_number)
    model_channels = _optional_context_integer(metrics, "model_channels", line_number)
    model_hidden_size = _optional_context_integer(metrics, "model_hidden_size", line_number)

    return HistoricalSelfPlayReference(
        source_path="",
        source_sha256="",
        record_line=line_number,
        record_index=record_index,
        run_id=run_id,
        step=step,
        timestamp_unix=timestamp,
        device_type=device,
        games=int(parsed["games"]),
        games_per_second=float(parsed["games_per_second"]),
        positions_per_second=float(parsed["positions_per_second"]),
        leaf_evaluations_per_second=float(parsed["leaf_evaluations_per_second"]),
        cpu_mean_process_utilization_percent=cpu_mean,
        cpu_logical_core_count=cpu_cores,
        gpu_mean_utilization_percent=(
            gpu["gpu_mean_utilization_percent"] if device == "cuda" else None
        ),
        gpu_p95_utilization_percent=(
            gpu["gpu_p95_utilization_percent"] if device == "cuda" else None
        ),
        gpu_max_utilization_percent=(
            gpu["gpu_max_utilization_percent"] if device == "cuda" else None
        ),
        mean_inference_batch_size=float(parsed["mean_inference_batch_size"]),
        p95_inference_batch_size=float(parsed["p95_inference_batch_size"]),
        max_inference_batch_size=float(parsed["max_inference_batch_size"]),
        duplicate_leaf_rate=float(parsed["duplicate_leaf_rate"]),
        steady_state_mean_batch_occupancy=float(
            parsed["steady_state_mean_batch_occupancy"]
        ),
        active_game_target=int(parsed["active_game_target"]),
        generation_seconds=generation,
        simulations=simulations,
        model_blocks=model_blocks,
        model_channels=model_channels,
        model_hidden_size=model_hidden_size,
    )


def _device_type(value: object, line: int) -> str:
    if not isinstance(value, str) or value not in {"cpu", "cuda"}:
        raise _line_error(line, "device_type must be 'cpu' or 'cuda'")
    return value


def _run_id(value: object, line: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _line_error(line, "run_id must be a non-empty string or null")
    return value


def _metric(value: object, name: str, line: int) -> int | float:
    if name in {"games", "active_game_target"}:
        return _integer(value, name, line, positive=True)
    if name in {"duplicate_leaf_rate", "steady_state_mean_batch_occupancy"}:
        return _number(value, name, line, minimum=0.0, maximum=1.0)
    return _number(value, name, line, positive=True)


def _optional_number(
    metrics: dict[object, object], source_name: str, name: str, line: int,
    *, minimum: float | None = None, maximum: float | None = None,
) -> float | None:
    if source_name not in metrics:
        return None
    return _number(metrics[source_name], name, line, minimum=minimum, maximum=maximum)


def _optional_integer(
    metrics: dict[object, object], source_name: str, name: str, line: int,
    *, positive: bool = False,
) -> int | None:
    if source_name not in metrics:
        return None
    return _integer(metrics[source_name], name, line, positive=positive)


def _optional_context_integer(
    metrics: dict[object, object], name: str, line: int
) -> int | None:
    return _optional_integer(metrics, name, name, line, positive=True)


def _integer(
    value: object, name: str, line: int, *, positive: bool = False, nonnegative: bool = False
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _line_error(line, f"{name} must be an integer")
    if positive and value <= 0:
        raise _line_error(line, f"{name} must be positive")
    if nonnegative and value < 0:
        raise _line_error(line, f"{name} cannot be negative")
    return value


def _number(
    value: object, name: str, line: int, *, positive: bool = False,
    minimum: float | None = None, maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _line_error(line, f"{name} must be numeric")
    try:
        parsed = float(value)
    except OverflowError as error:
        raise _line_error(line, f"{name} must be finite") from error
    if not math.isfinite(parsed):
        raise _line_error(line, f"{name} must be finite")
    if positive and parsed <= 0.0:
        raise _line_error(line, f"{name} must be positive")
    if minimum is not None and parsed < minimum:
        raise _line_error(line, f"{name} must be at least {minimum:g}")
    if maximum is not None and parsed > maximum:
        raise _line_error(line, f"{name} must be at most {maximum:g}")
    return parsed


def _line_error(line: int, message: str) -> ValueError:
    return ValueError(f"Historical telemetry line {line}: {message}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


__all__ = [
    "HistoricalSelfPlayReference",
    "INFORMATIONAL_REASON",
    "load_historical_self_play_reference",
]
