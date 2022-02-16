from torchmetrics import Metric
from torch import Tensor


class KLDivergence(Metric):
    def __init__(self, X: Tensor) -> None:
        super().__init__()
        self.X = X

        self.add_state("x_generated", default=[], dist_reduce_fx="cat")

    def update(self, x_generated: Tensor) -> None:
        self.x_generated.append(x_generated)

    def compute(self) -> Tensor:
        return
