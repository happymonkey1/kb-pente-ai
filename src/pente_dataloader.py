import torch
from torch.utils.data import Dataset
import numpy as np

def board_to_tensor(board: np.ndarray) -> np.ndarray:
    current_mask = (board == 1).astype(np.float32)
    opponent_mask = (board == 2).astype(np.float32)
    return np.stack([current_mask, opponent_mask], axis=0)

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

        return torch.from_numpy(tensor), torch.from_numpy(policy_flat), torch.tensor(outcome, dtype=torch.float32)