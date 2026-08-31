from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.game.pente.pente_board import PenteBoard
from src.game.pente.pente_game import PenteGame
from src.game.pente.rules import PenteRuleset
from src.model.model_v1 import PenteNet, positions_to_tensor


@dataclass(frozen=True, slots=True)
class TinyLearningReport:
    examples: int
    steps: int
    initial_loss: float
    final_loss: float
    policy_accuracy: float
    value_mse: float

    @property
    def passed(self) -> bool:
        return (
            self.final_loss < self.initial_loss * 0.1
            and self.policy_accuracy >= 0.95
            and self.value_mse <= 0.02
        )


def run_tiny_learning_verification(seed: int = 11, steps: int = 300) -> TinyLearningReport:
    if steps < 1:
        raise ValueError("Training steps must be positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    game = PenteGame(5, ruleset=PenteRuleset.FREESTYLE)
    positions, target_policies, target_values = _build_examples(game, rng, 32)
    model = PenteNet(
        device,
        board_size=5,
        action_size=25,
        num_res_blocks=2,
        num_channels=32,
        hidden_fc_size=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    states = positions_to_tensor(positions, device)
    policies = torch.from_numpy(target_policies).to(device)
    values = torch.from_numpy(target_values).to(device).view(-1, 1)

    model.train()
    initial_loss = float(_loss(model, states, policies, values)[0].item())
    for _ in range(steps):
        optimizer.zero_grad()
        loss, _, _ = _loss(model, states, policies, values)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        final_loss, _, value_loss = _loss(model, states, policies, values)
        policy_logits, predicted_values = model(states)
        predicted_actions = torch.argmax(policy_logits, dim=1)
        expected_actions = torch.argmax(policies, dim=1)
        policy_accuracy = float((predicted_actions == expected_actions).float().mean().item())
    return TinyLearningReport(
        examples=len(positions),
        steps=steps,
        initial_loss=initial_loss,
        final_loss=float(final_loss.item()),
        policy_accuracy=policy_accuracy,
        value_mse=float(value_loss.item()),
    )


def _build_examples(
    game: PenteGame,
    rng: np.random.Generator,
    count: int,
) -> tuple[list[PenteBoard], np.ndarray, np.ndarray]:
    positions: list[PenteBoard] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    keys: set[bytes] = set()

    while len(positions) < count:
        position = game.init_board()
        move_count = 1 + len(positions) % 10
        for _ in range(move_count):
            legal = np.flatnonzero(game.get_valid_moves(position, position.current_player))
            action = int(rng.choice(legal))
            position, _ = game.apply_action(position, position.current_player, action)
            if game.check_game_end(position).is_terminal:
                break
        if game.check_game_end(position).is_terminal or position.state_key() in keys:
            continue

        legal = np.flatnonzero(game.get_valid_moves(position, position.current_player))
        target_action = int(rng.choice(legal))
        policy = np.zeros(game.get_action_size(), dtype=np.float32)
        policy[target_action] = 1.0
        keys.add(position.state_key())
        positions.append(position)
        policies.append(policy)
        values.append(1.0 if rng.random() >= 0.5 else -1.0)

    return (
        positions,
        np.stack(policies, axis=0),
        np.asarray(values, dtype=np.float32),
    )


def _loss(
    model: PenteNet,
    states: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_logits, predicted_values = model(states)
    policy_loss = -torch.sum(
        target_policies * torch.log_softmax(policy_logits, dim=1),
        dim=1,
    ).mean()
    value_loss = torch.nn.functional.mse_loss(predicted_values, target_values)
    return policy_loss + value_loss, policy_loss, value_loss
