import torch
from torch import Tensor
from torch import nn
from typing import Tuple, List, Union


class CouplingLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        mask: Tensor,
    ):
        super().__init__()
        self.sfun = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            *self._hidden_layers(hidden_size, n_hidden_layers),
            nn.Linear(hidden_size, input_size),
            nn.Tanh(),
        )
        self.tfun = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            *self._hidden_layers(hidden_size, n_hidden_layers),
            nn.Linear(hidden_size, input_size),
        )
        self.mask = mask

    def _hidden_layers(self, hidden_size: int, n_hidden_layers: int) -> List[nn.Module]:
        return [
            module
            for _ in range(n_hidden_layers)
            for module in [nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)]
        ]

    def forward(
        self, inputs: Tuple[Tensor, Union[Tensor, None]]
    ) -> Tuple[Tensor, Tensor]:
        x, ldj_sum = inputs

        x_m = x * self.mask
        s_out = self.sfun(x_m)
        t_out = self.tfun(x_m)

        y = x_m + (1 - self.mask) * (x * torch.exp(s_out) + t_out)
        ldj_sum = (
            ldj_sum + s_out.sum(dim=1) if ldj_sum is not None else s_out.sum(dim=1)
        )

        return y, ldj_sum

    def inverse(self, y: Tensor) -> Tensor:
        y_m = y * self.mask
        x = y_m + (1 - self.mask) * (y * (1 - self.mask) - self.tfun(y_m)) * torch.exp(
            -self.sfun(y_m)
        )
        return x
