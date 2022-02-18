from torch import Tensor
from torch import nn
from typing import Tuple

from .coupling_layer import CouplingLayer


class RealNVP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        mask: Tensor,
        n_layers: int,
    ):
        super().__init__()
        self.module_list = nn.ModuleList(
            [
                CouplingLayer(
                    input_size,
                    hidden_size,
                    n_hidden_layers,
                    mask if i % 2 else 1 - mask,
                )
                for i in range(n_layers)
            ]
        )
        self.module = nn.Sequential(*self.module_list)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.module((x, None))

    def inverse(self, h: Tensor) -> Tensor:
        for module in reversed(self.module_list):
            h = module.inverse(h)
        return h
