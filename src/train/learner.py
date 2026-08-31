from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.evaluation.value_metrics import ValueMetrics
from src.evaluation.value_metrics import ValueMetricsAccumulator
from src.game.pente.pente_game import PenteGame
from src.model.model_v1 import PenteNet
from src.pente_dataloader import PenteDataset
from src.train.training_example import TrainingExample


@dataclass(frozen=True, slots=True)
class ModelTrainingStats:
    total_loss: float
    total_policy_loss: float
    total_policy_kl: float
    total_value_loss: float
    total_value_absolute_error: float
    total_value_bias: float
    num_batches: int
    value_metrics: ValueMetrics


def train_policy_value_model(
    net: PenteNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    game: PenteGame,
    training_examples: list[TrainingExample],
    batch_size: int,
    augment: bool,
) -> ModelTrainingStats:
    net.train()
    pin_memory = device.type != "cpu"
    dataloader = DataLoader(
        PenteDataset(training_examples, game, augment=augment),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    total_loss = 0.0
    total_policy_loss = 0.0
    total_policy_kl = 0.0
    total_value_loss = 0.0
    total_value_absolute_error = 0.0
    total_value_bias = 0.0
    num_batches = 0
    value_metrics = ValueMetricsAccumulator()

    for states, target_policies, target_values in dataloader:
        states = states.to(device, non_blocking=pin_memory)
        target_policies = target_policies.to(device, non_blocking=pin_memory)
        target_values = target_values.to(device, non_blocking=pin_memory).view(-1, 1)

        optimizer.zero_grad()
        policy_logits, predicted_values = net(states)
        policy_loss = -torch.sum(
            target_policies * torch.log_softmax(policy_logits, dim=1),
            dim=1,
        ).mean()
        target_entropy = -torch.sum(
            torch.where(
                target_policies > 0,
                target_policies * torch.log(target_policies.clamp_min(1e-12)),
                torch.zeros_like(target_policies),
            ),
            dim=1,
        ).mean()
        policy_kl = policy_loss - target_entropy
        value_loss = torch.nn.functional.mse_loss(predicted_values, target_values)
        loss = policy_loss + value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Training produced a non-finite loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_policy_loss += float(policy_loss.item())
        total_policy_kl += float(policy_kl.item())
        total_value_loss += float(value_loss.item())
        total_value_absolute_error += float(
            torch.mean(torch.abs(predicted_values - target_values)).item()
        )
        total_value_bias += float(torch.mean(predicted_values - target_values).item())
        value_metrics.add(
            predicted_values.detach().float().cpu().numpy(),
            target_values.detach().float().cpu().numpy(),
        )
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Training produced no batches")
    return ModelTrainingStats(
        total_loss=total_loss,
        total_policy_loss=total_policy_loss,
        total_policy_kl=total_policy_kl,
        total_value_loss=total_value_loss,
        total_value_absolute_error=total_value_absolute_error,
        total_value_bias=total_value_bias,
        num_batches=num_batches,
        value_metrics=value_metrics.finish(),
    )
