from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path


MANIFEST_PATTERN = "run-manifest-step-*.json"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MANIFESTS = 1_000


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    board_size: int
    action_size: int
    input_planes: int
    residual_blocks: int
    channels: int
    value_hidden_size: int

    def to_object(self) -> dict[str, int]:
        return {
            "board_size": self.board_size,
            "action_size": self.action_size,
            "input_planes": self.input_planes,
            "residual_blocks": self.residual_blocks,
            "channels": self.channels,
            "value_hidden_size": self.value_hidden_size,
        }


@dataclass(frozen=True, slots=True)
class RunManifest:
    training_run_id: str
    telemetry_path: Path
    start_iteration: int
    created_at_utc: str | None
    config: ArchitectureConfig
    runtime: dict[str, object]
    ruleset: str | None
    modified_at_unix: float

    def architecture(self) -> dict[str, object]:
        return {
            "available": True,
            "model": "PenteNet",
            "config": self.config.to_object(),
            "metrics": architecture_metrics(self.config),
            "runtime": self.runtime,
            "ruleset": self.ruleset,
            "manifest": {
                "created_at_utc": self.created_at_utc,
                "start_iteration": self.start_iteration,
            },
        }


@dataclass(frozen=True, slots=True)
class CachedManifest:
    signature: tuple[int, int]
    manifest: RunManifest | None


class RunManifestStore:
    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        max_manifest_bytes: int = MAX_MANIFEST_BYTES,
        max_manifests: int = MAX_MANIFESTS,
    ) -> None:
        if max_manifest_bytes < 1:
            raise ValueError("max_manifest_bytes must be positive")
        if max_manifests < 1:
            raise ValueError("max_manifests must be positive")

        self.roots = tuple(dict.fromkeys(Path(root).resolve() for root in roots))
        self.max_manifest_bytes = max_manifest_bytes
        self.max_manifests = max_manifests
        self._cache: dict[Path, CachedManifest] = {}

    def architecture_for(
        self,
        telemetry_path: str | Path,
        training_run_id: str | None,
    ) -> dict[str, object]:
        selected_path = Path(telemetry_path).resolve()
        matches: list[RunManifest] = []
        for path in self._discover():
            manifest = self._load(path)
            if manifest is None:
                continue
            if (
                manifest.telemetry_path == selected_path
                or training_run_id is not None
                and manifest.training_run_id == training_run_id
            ):
                matches.append(manifest)

        if not matches:
            return {
                "available": False,
                "reason": "No matching run manifest was found.",
            }

        selected = max(
            matches,
            key=lambda manifest: (
                manifest.telemetry_path == selected_path,
                manifest.training_run_id == training_run_id,
                manifest.start_iteration,
                manifest.modified_at_unix,
            ),
        )
        return selected.architecture()

    def _discover(self) -> tuple[Path, ...]:
        paths: set[Path] = set()
        for root in self.roots:
            if not root.is_dir():
                continue
            self._add_manifests(paths, root)
            try:
                children = root.iterdir()
            except OSError:
                continue
            for child in children:
                if len(paths) >= self.max_manifests:
                    break
                try:
                    if child.is_dir() and not child.is_symlink():
                        self._add_manifests(paths, child)
                except OSError:
                    continue
            if len(paths) >= self.max_manifests:
                break
        return tuple(sorted(paths))

    def _add_manifests(self, paths: set[Path], directory: Path) -> None:
        try:
            candidates = directory.glob(MANIFEST_PATTERN)
        except OSError:
            return
        for candidate in candidates:
            if len(paths) >= self.max_manifests:
                return
            try:
                resolved = candidate.resolve()
                if resolved.is_file() and any(
                    resolved.is_relative_to(root) for root in self.roots
                ):
                    paths.add(resolved)
            except OSError:
                continue

    def _load(self, path: Path) -> RunManifest | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = self._cache.get(path)
        if cached is not None and cached.signature == signature:
            return cached.manifest

        manifest = None
        if stat.st_size <= self.max_manifest_bytes:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                manifest = _parse_manifest(value, path, stat.st_mtime)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                pass
        self._cache[path] = CachedManifest(signature, manifest)
        return manifest


def architecture_metrics(config: ArchitectureConfig) -> dict[str, object]:
    board_area = config.board_size * config.board_size
    channels = config.channels

    stem_parameters = channels * config.input_planes * 3 * 3 + channels + 2 * channels
    residual_parameters = config.residual_blocks * (
        2 * (channels * channels * 3 * 3 + channels + 2 * channels)
    )
    policy_parameters = (
        2 * channels
        + 2
        + 4
        + config.action_size * (2 * board_area)
        + config.action_size
    )
    value_parameters = (
        channels
        + 1
        + 2
        + config.value_hidden_size * board_area
        + config.value_hidden_size
        + config.value_hidden_size
        + 1
    )
    parameter_count = (
        stem_parameters
        + residual_parameters
        + policy_parameters
        + value_parameters
    )

    stem_macs = board_area * channels * config.input_planes * 3 * 3
    residual_macs = (
        config.residual_blocks * 2 * board_area * channels * channels * 3 * 3
    )
    policy_macs = (
        board_area * 2 * channels
        + config.action_size * 2 * board_area
    )
    value_macs = (
        board_area * channels
        + config.value_hidden_size * board_area
        + config.value_hidden_size
    )
    macs = stem_macs + residual_macs + policy_macs + value_macs

    return {
        "parameter_count": parameter_count,
        "estimated_fp32_bytes": parameter_count * 4,
        "multiply_accumulates_per_position": macs,
        "estimated_flops_per_position": macs * 2,
        "parameterized_layer_count": 4 * config.residual_blocks + 9,
        "trunk_activation_values_per_position": board_area * channels,
        "parameters_by_stage": {
            "stem": stem_parameters,
            "residual_tower": residual_parameters,
            "policy_head": policy_parameters,
            "value_head": value_parameters,
        },
        "multiply_accumulates_by_stage": {
            "stem": stem_macs,
            "residual_tower": residual_macs,
            "policy_head": policy_macs,
            "value_head": value_macs,
        },
    }


def _parse_manifest(value: object, path: Path, modified_at_unix: float) -> RunManifest:
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("Unsupported run manifest")

    training_run_id = value.get("training_run_id")
    outputs = value.get("outputs")
    model = value.get("model")
    if not isinstance(training_run_id, str) or not training_run_id:
        raise ValueError("Run manifest is missing training_run_id")
    if not isinstance(outputs, Mapping) or not isinstance(model, Mapping):
        raise ValueError("Run manifest is missing outputs or model")
    telemetry = outputs.get("telemetry")
    if not isinstance(telemetry, str) or not telemetry:
        raise ValueError("Run manifest is missing outputs.telemetry")

    board_size = _integer(model, "board_size", minimum=5, maximum=25)
    action_size = _integer(model, "action_size", minimum=25, maximum=625)
    if action_size != board_size * board_size:
        raise ValueError("Run manifest action size does not match board size")
    config = ArchitectureConfig(
        board_size=board_size,
        action_size=action_size,
        input_planes=_integer(model, "input_planes", minimum=1, maximum=64),
        residual_blocks=_integer(model, "num_res_blocks", minimum=1, maximum=100),
        channels=_integer(model, "num_channels", minimum=1, maximum=4_096),
        value_hidden_size=_integer(
            model,
            "hidden_fc_size",
            minimum=1,
            maximum=65_536,
        ),
    )

    start_iteration = value.get("start_iteration")
    if isinstance(start_iteration, bool) or not isinstance(start_iteration, int):
        start_iteration = 0
    created_at_utc = value.get("created_at_utc")
    if not isinstance(created_at_utc, str):
        created_at_utc = None

    runtime_value = value.get("runtime")
    runtime = _public_runtime(runtime_value) if isinstance(runtime_value, Mapping) else {}
    arguments = value.get("program_arguments")
    ruleset_value = arguments.get("ruleset") if isinstance(arguments, Mapping) else None
    ruleset = ruleset_value if isinstance(ruleset_value, str) else None
    telemetry_path = Path(telemetry)
    if not telemetry_path.is_absolute():
        telemetry_path = path.parent / telemetry_path

    return RunManifest(
        training_run_id=training_run_id,
        telemetry_path=telemetry_path.resolve(),
        start_iteration=max(0, start_iteration),
        created_at_utc=created_at_utc,
        config=config,
        runtime=runtime,
        ruleset=ruleset,
        modified_at_unix=modified_at_unix,
    )


def _integer(
    value: Mapping[object, object],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    selected = value.get(name)
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or not minimum <= selected <= maximum
    ):
        raise ValueError(f"Run manifest {name} is invalid")
    return selected


def _public_runtime(value: Mapping[object, object]) -> dict[str, object]:
    runtime: dict[str, object] = {}
    for name in ("device", "device_name", "torch", "torch_cuda"):
        selected = value.get(name)
        if isinstance(selected, str) or selected is None:
            runtime[name] = selected
    compiled = value.get("compiled")
    if isinstance(compiled, bool):
        runtime["compiled"] = compiled
    return runtime
