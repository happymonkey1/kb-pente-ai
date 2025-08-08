import numpy as np

class MCTSNode:
    def __init__(self,
                 prior: float,
                 board: np.ndarray | None = None,
                 player: int | None = None,
                 captures: dict[int, int] | None = None,
                 parent: 'MCTSNode' | None = None):

        if captures is None:
            captures = {1: 0, 2: 0}
        self.prior = prior
        self.board = board
        self.player = player
        self.captures: dict[int, int] = captures
        self.parent = parent

        self.value_sum = 0.0
        self.visit_count = 0
        self.children: dict[int, 'MCTSNode'] = {}

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count