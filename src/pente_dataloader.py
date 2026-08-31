from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.game.pente.pente_game import PenteGame
from src.train.training_example import TrainingExample


class PenteDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        examples: list[TrainingExample],
        game: PenteGame,
        augment: bool = False,
    ) -> None:
        self.examples = examples
        self.game = game
        self.augment = augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        position = example.position
        policy = example.policy
        if self.augment:
            symmetry = int(torch.randint(0, 8, ()).item())
            position, policy = self.game.get_symmetry(position, policy, symmetry)

        state_tensor = torch.from_numpy(position.feature_planes())
        policy_tensor = torch.from_numpy(policy.copy())
        value_tensor = torch.tensor(example.value, dtype=torch.float32)
        return state_tensor, policy_tensor, value_tensor
