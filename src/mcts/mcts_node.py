
class MCTSNode:
    def __init__(self, prior: float):
        self.prior = prior
        self.value_sum = 0.0
        self.visit_count = 0
        self.children = {}

    @property
    def value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count