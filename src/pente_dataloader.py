import torch
from torch.utils.data import Dataset
import numpy as np

from src.game.board_utils import board_to_tensor


class PenteDataset(Dataset):
    def __init__(self, games):
        self.games = games

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        state, policy, outcome = self.games[idx]

        tensor = board_to_tensor(state)

        if policy.ndim == 2:
            policy_flat = policy.ravel()
        else:
            policy_flat = policy.astype(np.float32)

        return tensor, torch.from_numpy(policy_flat), torch.tensor(outcome, dtype=torch.float32)