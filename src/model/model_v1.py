from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import tempfile
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.artifacts import (
    CHECKPOINT_SCHEMA_VERSION,
    POSITION_SCHEMA_VERSION,
    TRAINING_EXAMPLE_SCHEMA_VERSION,
)
from src.game.pente.pente_board import PenteBoard


@dataclass(frozen=True, slots=True)
class PenteNetConfig:
    board_size: int = 19
    action_size: int = 19 * 19
    input_planes: int = 4
    num_res_blocks: int = 4
    num_channels: int = 64
    hidden_fc_size: int = 256

    def __post_init__(self) -> None:
        if self.input_planes != 4:
            raise ValueError("The current Pente state contract requires exactly four input planes")
        if self.action_size != self.board_size * self.board_size:
            raise ValueError("Action size must equal board_size squared")


@dataclass(frozen=True, slots=True)
class CheckpointTrainingState:
    iteration: int
    replay_snapshot_generation: int
    training_run_id: str


def positions_to_tensor(positions: Sequence[PenteBoard], device: torch.device) -> torch.Tensor:
    if not positions:
        raise ValueError("At least one position is required")
    features = np.stack([position.feature_planes() for position in positions], axis=0)
    return torch.from_numpy(features).to(device=device, dtype=torch.float32)


class PenteNet(nn.Module):
    def __init__(
        self,
        device: torch.device,
        board_size: int = 19,
        action_size: int = 19 * 19,
        num_res_blocks: int = 4,
        num_channels: int = 64,
        hidden_fc_size: int = 256,
    ) -> None:
        super().__init__()
        self.device = device
        self.config = PenteNetConfig(
            board_size=board_size,
            action_size=action_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            hidden_fc_size=hidden_fc_size,
        )
        self.board_size = board_size
        self.action_size = action_size
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.hidden_fc_size = hidden_fc_size

        self.conv_in = nn.Conv2d(self.config.input_planes, num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * board_size * board_size, action_size)

        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value_in = nn.Linear(board_size * board_size, hidden_fc_size)
        self.fc_value_out = nn.Linear(hidden_fc_size, 1)
        self.to(device)

    @staticmethod
    def from_existing_model(net: PenteNet) -> PenteNet:
        config = net.config
        return PenteNet(
            net.device,
            config.board_size,
            config.action_size,
            config.num_res_blocks,
            config.num_channels,
            config.hidden_fc_size,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.bn_in(self.conv_in(inputs)))
        for block in self.res_blocks:
            hidden = block(hidden)

        policy = F.relu(self.bn_policy(self.conv_policy(hidden)))
        policy = self.fc_policy(policy.flatten(start_dim=1))

        value = F.relu(self.bn_value(self.conv_value(hidden)))
        value = F.relu(self.fc_value_in(value.flatten(start_dim=1)))
        value = torch.tanh(self.fc_value_out(value))
        return policy, value

    def evaluate(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        policies, values = self.evaluate_batch((position,))
        return policies[0], float(values[0])

    def evaluate_batch(self, positions: Sequence[PenteBoard]) -> tuple[np.ndarray, np.ndarray]:
        use_autocast = self.device.type == "cuda"
        autocast_dtype = (
            torch.bfloat16
            if use_autocast and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.inference_mode():
            inputs = positions_to_tensor(positions, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                policy_logits, values = self.forward(inputs)
                policies = F.softmax(policy_logits, dim=1)
        return (
            policies.detach().float().cpu().numpy(),
            values.detach().float().cpu().numpy().reshape(-1),
        )

    def predict(self, position: PenteBoard) -> tuple[np.ndarray, float]:
        return self.evaluate(position)

    def get_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def checkpoint_metadata(self, ruleset: str) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "position_schema_version": POSITION_SCHEMA_VERSION,
            "training_example_schema_version": TRAINING_EXAMPLE_SCHEMA_VERSION,
            "ruleset": ruleset,
            "model_config": asdict(self.config),
        }

    @staticmethod
    def save_checkpoint(state: dict[str, object], checkpoint_dir: str, filename: str = "checkpoint.pth.tar") -> None:
        metadata = state.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Checkpoint state is missing compatible schema metadata")
        os.makedirs(checkpoint_dir, exist_ok=True)
        destination = os.path.join(checkpoint_dir, filename)
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=checkpoint_dir,
                delete=False,
            ) as stream:
                temporary_path = stream.name
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def load_checkpoint(
        checkpoint_dir: str,
        net: PenteNet,
        filename: str = "checkpoint.pth.tar",
        optimizer: torch.optim.Optimizer | None = None,
        expected_ruleset: str | None = None,
    ) -> int:
        return PenteNet.load_checkpoint_from_path(
            os.path.join(checkpoint_dir, filename),
            net,
            optimizer,
            expected_ruleset,
        )

    @staticmethod
    def load_checkpoint_from_path(
        filepath: str,
        net: PenteNet,
        optimizer: torch.optim.Optimizer | None = None,
        expected_ruleset: str | None = None,
    ) -> int:
        checkpoint = PenteNet._load_checkpoint_payload(
            filepath,
            net,
            optimizer,
            expected_ruleset,
        )
        return PenteNet._checkpoint_iteration(checkpoint)

    @staticmethod
    def load_training_checkpoint_from_path(
        filepath: str,
        net: PenteNet,
        optimizer: torch.optim.Optimizer,
        expected_ruleset: str,
    ) -> CheckpointTrainingState:
        checkpoint = PenteNet._load_checkpoint_payload(
            filepath,
            net,
            optimizer,
            expected_ruleset,
        )
        iteration = PenteNet._checkpoint_iteration(checkpoint)
        training_run_id = checkpoint.get("training_run_id")
        replay_generation = checkpoint.get("replay_snapshot_generation")
        if not isinstance(training_run_id, str) or not training_run_id:
            raise ValueError("Checkpoint is missing a training run identifier")
        if (
            isinstance(replay_generation, bool)
            or not isinstance(replay_generation, int)
            or not 0 <= replay_generation <= iteration
        ):
            raise ValueError("Checkpoint has an invalid replay snapshot generation")
        return CheckpointTrainingState(
            iteration=iteration,
            replay_snapshot_generation=replay_generation,
            training_run_id=training_run_id,
        )

    @staticmethod
    def _checkpoint_iteration(checkpoint: dict[str, object]) -> int:
        iteration = checkpoint.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Checkpoint iteration must be a non-negative integer")
        return iteration

    @staticmethod
    def _load_checkpoint_payload(
        filepath: str,
        net: PenteNet,
        optimizer: torch.optim.Optimizer | None,
        expected_ruleset: str | None,
    ) -> dict[str, object]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=net.device)
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                "Legacy checkpoint has no schema metadata and is incompatible with the corrected four-plane model"
            )
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema {metadata.get('schema_version')} is incompatible with "
                f"schema {CHECKPOINT_SCHEMA_VERSION}"
            )
        if (
            metadata.get("position_schema_version") != POSITION_SCHEMA_VERSION
            or metadata.get("training_example_schema_version")
            != TRAINING_EXAMPLE_SCHEMA_VERSION
        ):
            raise ValueError("Checkpoint state and training-example schemas are incompatible")
        if metadata.get("model_config") != asdict(net.config):
            raise ValueError("Checkpoint model configuration does not match the requested model")
        if expected_ruleset is not None and metadata.get("ruleset") != expected_ruleset:
            raise ValueError(
                f"Checkpoint ruleset {metadata.get('ruleset')!r} does not match {expected_ruleset!r}"
            )

        net.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint

    def get_checkpoint_file_name(self, iteration: int) -> str:
        config = self.config
        return (
            f"checkpoint-{iteration}_{config.board_size}_{config.action_size}_"
            f"{config.num_res_blocks}_{config.num_channels}_{config.hidden_fc_size}.pth.tar"
        )


class ResBlock(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = F.relu(self.bn1(self.conv1(inputs)))
        hidden = self.bn2(self.conv2(hidden))
        return F.relu(hidden + residual)
