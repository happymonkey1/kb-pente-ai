import torch
from torch.utils.data import Dataset
import numpy as np

from src.game.pente.pente_game import PenteGame

def board_to_cpu_tensor(board: np.ndarray) -> torch.Tensor:
    # TODO: this only supports two players
    p1_mask = (board == PenteGame.PLAYER_ONE)
    p2_mask = (board == PenteGame.PLAYER_TWO)
    return torch.from_numpy(np.stack([p1_mask, p2_mask], axis=0)).type(dtype=torch.float32)

class PenteDataset(Dataset):
    def __init__(self, games: list[tuple[np.ndarray, np.ndarray, float]]):
        self.games = games

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        state, policy, outcome = self.games[idx]

        state_tensor = board_to_cpu_tensor(state)
        policy_tensor = torch.from_numpy(policy)
        outcome_tensor = torch.tensor(outcome, dtype=torch.float32)

        return state_tensor, policy_tensor, outcome_tensor