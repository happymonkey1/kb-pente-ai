import numpy as np
from typing import Union

class MCTSNode:
    def __init__(self,
                 prior: float,
                 board: np.ndarray | None = None,
                 player: int | None = None,
                 captures: dict[int, int] | None = None,
                 parent: Union['MCTSNode', None] = None):

        if captures is None:
            captures = {1: 0, 2: 0}
        self.prior = prior
        self.board = board
        self.player = player
        self.captures: dict[int, int] = captures
        self.parent = parent

        self.value_sum = 0.0
        self.visit_count = 0
        self.children: dict[tuple[int, int], 'MCTSNode'] = {}

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def __str__(self):
        return f"MCTSNode(prior={self.prior}, board={self.board}, player={self.player}, captures={self.captures}, parent={self.parent}, value={self.value}, visit_count={self.visit_count}, children={self.children})"
