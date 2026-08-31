from __future__ import annotations

import numpy as np
import torch


def seed_training_iteration(seed: int, iteration: int) -> np.random.Generator:
    if seed < 0 or iteration < 0:
        raise ValueError("Training seed and iteration cannot be negative")
    numpy_seed, torch_seed = np.random.SeedSequence((seed, iteration)).spawn(2)
    torch.manual_seed(int(torch_seed.generate_state(1, dtype=np.uint64)[0]))
    return np.random.default_rng(numpy_seed)
