from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.game.pente.pente_board import PenteBoard


@dataclass(frozen=True, slots=True, eq=False)
class TrainingExample:
    position: PenteBoard
    policy: np.ndarray
    value: float

    def __post_init__(self) -> None:
        policy = np.asarray(self.policy, dtype=np.float32).reshape(-1)
        action_size = self.position.board.size
        if policy.shape != (action_size,):
            raise ValueError(f"Policy must have shape ({action_size},), found {policy.shape}")
        if not np.isfinite(policy).all() or np.any(policy < 0):
            raise ValueError("Policy must contain finite non-negative probabilities")
        if not math.isclose(float(policy.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"Policy must sum to one, found {float(policy.sum())}")
        occupied = self.position.board.reshape(-1) != 0
        if float(policy[occupied].sum()) > 1e-6:
            raise ValueError("Policy assigns probability to an occupied action")
        if not math.isfinite(self.value) or not -1.0 <= self.value <= 1.0:
            raise ValueError(f"Value must be finite and in [-1, 1], found {self.value}")

        copied_policy: np.ndarray = np.array(policy, dtype=np.float32, copy=True)
        copied_policy.flags.writeable = False
        object.__setattr__(self, "policy", copied_policy)
        object.__setattr__(self, "value", float(self.value))
