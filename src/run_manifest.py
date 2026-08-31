from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Mapping

import torch

from src.artifacts import (
    CHECKPOINT_SCHEMA_VERSION,
    POSITION_SCHEMA_VERSION,
    PROFESSIONAL_DATA_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
    TRAINING_EXAMPLE_SCHEMA_VERSION,
)
from src.model.model_v1 import PenteNet
from src.train.self_play import SelfPlayTrainerArgs


RUN_MANIFEST_SCHEMA_VERSION = 1

_NATIVE_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cmake",
        ".cpp",
        ".cu",
        ".cuh",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".inc",
        ".inl",
    }
)
_TORCH_SOURCE_SUFFIXES = _NATIVE_SOURCE_SUFFIXES | {
    ".json",
    ".py",
    ".sh",
    ".toml",
}
_GENERATED_NATIVE_DIRECTORIES = frozenset(
    {
        "__pycache__",
        "build",
        "build-debug",
        "build-release",
        "cmake-build-debug",
        "cmake-build-minsizerel",
        "cmake-build-release",
        "cmake-build-relwithdebinfo",
        "dist",
        "target",
    }
)


def write_run_manifest(
    path: str | Path,
    repository_root: str | Path,
    command: list[str],
    training_run_id: str,
    start_iteration: int,
    device: torch.device,
    compiled: bool,
    net: PenteNet,
    trainer_args: SelfPlayTrainerArgs,
    program_arguments: Mapping[str, object],
    telemetry_path: str | Path,
) -> dict[str, object]:
    if not training_run_id:
        raise ValueError("Run manifest requires a training run identifier")
    root = Path(repository_root).resolve()
    manifest: dict[str, object] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_run_id": training_run_id,
        "start_iteration": start_iteration,
        "command": command,
        "repository": _repository_metadata(root),
        "runtime": _runtime_metadata(device, compiled),
        "artifact_schemas": {
            "position": POSITION_SCHEMA_VERSION,
            "training_example": TRAINING_EXAMPLE_SCHEMA_VERSION,
            "professional_data": PROFESSIONAL_DATA_SCHEMA_VERSION,
            "replay": REPLAY_SCHEMA_VERSION,
            "checkpoint": CHECKPOINT_SCHEMA_VERSION,
        },
        "model": asdict(net.config),
        "trainer": _json_value(asdict(trainer_args)),
        "program_arguments": _json_value(dict(program_arguments)),
        "outputs": {
            "telemetry": str(Path(telemetry_path).resolve()),
            "checkpoint_directory": str(Path(trainer_args.checkpoint_dir).resolve()),
        },
    }
    _atomic_json_write(Path(path), manifest)
    return manifest


def source_fingerprint(repository_root: str | Path) -> str:
    root = Path(repository_root).resolve()
    candidates = [root / "main.py", root / "pyproject.toml", root / "uv.lock"]
    candidates.extend((root / "src").rglob("*.py"))
    candidates.extend((root / "script").glob("*.py"))
    native_root = root / "native"
    candidates.append(native_root / "CMakeLists.txt")
    for directory_name in ("include", "src", "tests", "bench"):
        candidates.extend(
            _native_source_files(native_root / directory_name, _NATIVE_SOURCE_SUFFIXES)
        )
    torch_root = native_root / "torch"
    if torch_root.is_dir():
        candidates.extend(
            path
            for path in torch_root.iterdir()
            if path.is_file() and path.suffix.lower() in _TORCH_SOURCE_SUFFIXES
        )
    digest = hashlib.sha256()
    for path in sorted({path for path in candidates if path.is_file()}):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _native_source_files(
    directory: Path,
    suffixes: frozenset[str],
) -> list[Path]:
    if not directory.is_dir():
        return []

    paths: list[Path] = []
    for current_directory, child_directories, filenames in os.walk(
        directory,
        topdown=True,
    ):
        child_directories[:] = sorted(
            name
            for name in child_directories
            if name.lower() not in _GENERATED_NATIVE_DIRECTORIES
        )
        paths.extend(
            Path(current_directory) / filename
            for filename in filenames
            if Path(filename).suffix.lower() in suffixes
        )
    return paths


def _repository_metadata(root: Path) -> dict[str, object]:
    status = _git(root, "status", "--short")
    return {
        "root": str(root),
        "commit": _git(root, "rev-parse", "HEAD").strip() or None,
        "branch": _git(root, "branch", "--show-current").strip() or None,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "source_fingerprint_sha256": source_fingerprint(root),
    }


def _runtime_metadata(device: torch.device, compiled: bool) -> dict[str, object]:
    cuda = device.type == "cuda"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else None,
        "device_total_memory_bytes": (
            torch.cuda.get_device_properties(device).total_memory if cuda else None
        ),
        "compiled": compiled,
        "torch_threads": torch.get_num_threads(),
    }


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _atomic_json_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = stream.name
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
