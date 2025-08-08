import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from src.game.board_utils import board_to_tensor

logger = logging.getLogger(__name__)

def _debug_check(state, idx):
    if not isinstance(state, np.ndarray) or state.shape != (19, 19):
        logger.error(f"\n\nCRITICAL ERROR IN DATA PIPELINE AT INDEX {idx}")
        logger.error(f"Expected a (19, 19) NumPy array, but got:")
        logger.error(f"  - Type: {type(state)}")
        logger.error(f"  - Shape: {getattr(state, 'shape', 'N/A')}")
        raise TypeError("Invalid data type found in the 'games' list. See details above.")


class PenteDataset(Dataset):
    def __init__(self, games: list[tuple[np.ndarray, np.ndarray, float]]):
        self.games = games

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        state, policy, outcome = self.games[idx]
        _debug_check(state, idx)

        state_tensor = board_to_tensor(state).squeeze(0)
        policy_tensor = torch.from_numpy(policy)
        outcome_tensor = torch.tensor(outcome, dtype=torch.float32)

        return state_tensor, policy_tensor, outcome_tensor

