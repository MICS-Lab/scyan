import torch
from torch import Tensor
from torch import nn
from typing import Tuple

from .coupling_layer import CouplingLayer


class RealNVP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_hidden_layers: int,
        n_layers: int,
    ):
        super().__init__()
        self.module_list = nn.ModuleList(
            [
                CouplingLayer(
                    input_size,
                    hidden_size,
                    output_size,
                    n_hidden_layers,
                    self._mask(output_size, i),
                )
                for i in range(n_layers)
            ]
        )
        self.module = nn.Sequential(*self.module_list)

    def _mask(self, output_size: int, shift: int):
        return (((torch.arange(output_size) - shift) % 3) > 0).to(int)

    def forward(self, x: Tensor, covariates: Tensor) -> Tuple[Tensor, Tensor]:
        return self.module((x, covariates, None))

    def inverse(self, h: Tensor, covariates: Tensor) -> Tensor:
        for module in reversed(self.module_list):
            h = module.inverse(h, covariates)
        return h
