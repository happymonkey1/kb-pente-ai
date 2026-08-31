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
class SupervisedEvaluationStats:
    examples: int
    policy_cross_entropy: float
    policy_top_one_accuracy: float
    policy_top_five_accuracy: float
    value_mse: float
    value_metrics: ValueMetrics


def supervised_evaluation_metrics(
    stats: SupervisedEvaluationStats,
    prefix: str = "professional_validation",
) -> dict[str, int | float]:
    if not prefix:
        raise ValueError("Metric prefix cannot be empty")
    metrics: dict[str, int | float] = {
        f"{prefix}_examples": stats.examples,
        f"{prefix}_policy_cross_entropy": stats.policy_cross_entropy,
        f"{prefix}_policy_top_one_accuracy": stats.policy_top_one_accuracy,
        f"{prefix}_policy_top_five_accuracy": stats.policy_top_five_accuracy,
        f"{prefix}_value_mse": stats.value_mse,
    }
    metrics.update(stats.value_metrics.to_metrics(f"{prefix}_value"))
    return metrics


def evaluate_supervised_examples(
    net: PenteNet,
    game: PenteGame,
    examples: list[TrainingExample],
    batch_size: int,
) -> SupervisedEvaluationStats:
    if not examples:
        raise ValueError("Supervised evaluation requires examples")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    net.eval()
    dataloader = DataLoader(
        PenteDataset(examples, game, augment=False),
        batch_size=batch_size,
        shuffle=False,
    )
    total_cross_entropy = 0.0
    total_top_one = 0
    total_top_five = 0
    total_value_squared_error = 0.0
    total_examples = 0
    value_metrics = ValueMetricsAccumulator()

    with torch.inference_mode():
        for states, target_policies, target_values in dataloader:
            states = states.to(net.device)
            target_policies = target_policies.to(net.device)
            target_values = target_values.to(net.device).view(-1, 1)
            policy_logits, predicted_values = net(states)
            batch_count = states.shape[0]
            cross_entropy = -torch.sum(
                target_policies * torch.log_softmax(policy_logits, dim=1),
                dim=1,
            )
            expected_actions = torch.argmax(target_policies, dim=1)
            top_five = torch.topk(policy_logits, k=min(5, game.get_action_size()), dim=1).indices

            total_cross_entropy += float(cross_entropy.sum().item())
            total_top_one += int((torch.argmax(policy_logits, dim=1) == expected_actions).sum().item())
            total_top_five += int((top_five == expected_actions[:, None]).any(dim=1).sum().item())
            total_value_squared_error += float(
                torch.sum((predicted_values - target_values) ** 2).item()
            )
            value_metrics.add(
                predicted_values.detach().float().cpu().numpy(),
                target_values.detach().float().cpu().numpy(),
            )
            total_examples += batch_count

    return SupervisedEvaluationStats(
        examples=total_examples,
        policy_cross_entropy=total_cross_entropy / total_examples,
        policy_top_one_accuracy=total_top_one / total_examples,
        policy_top_five_accuracy=total_top_five / total_examples,
        value_mse=total_value_squared_error / total_examples,
        value_metrics=value_metrics.finish(),
    )
