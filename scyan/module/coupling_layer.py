import torch
from torch import Tensor
from torch import nn
from typing import Tuple, List, Union
import pytorch_lightning as pl


class CouplingLayer(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_hidden_layers: int,
        mask: Tensor,
    ):
        """One single coupling layer module.

        Args:
            input_size (int): Input size, i.e. number of markers + covariates
            hidden_size (int): Neural networks hidden size
            output_size (int): Output size, i.e. number of markers
            n_hidden_layers (int): Number of s and t hidden layers
            mask (Tensor): Mask used to separate x into (x1, x2)
        """
        super().__init__()
        self.sfun = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            *self._hidden_layers(hidden_size, n_hidden_layers),
            nn.Linear(hidden_size, output_size),
            nn.Tanh(),
        )
        self.tfun = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            *self._hidden_layers(hidden_size, n_hidden_layers),
            nn.Linear(hidden_size, output_size),
        )
        self.register_buffer("mask", mask)

    def _hidden_layers(self, hidden_size: int, n_hidden_layers: int) -> List[nn.Module]:
        return [
            module
            for _ in range(n_hidden_layers)
            for module in [nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)]
        ]

    def forward(
        self, inputs: Tuple[Tensor, Tensor, Union[Tensor, None]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Coupling layer forward function

        Args:
            inputs (Tuple[Tensor, Tensor, Union[Tensor, None]]): Tuple of (inputs, covariates, lod_det_jacobian sum)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple of (outputs, covariates, lod_det_jacobian sum)
        """
        x, covariates, ldj_sum = inputs

        x_m = x * self.mask
        st_input = torch.cat([x_m, covariates], dim=1)

        s_out = self.sfun(st_input)
        t_out = self.tfun(st_input)

        y = x_m + (1 - self.mask) * (x * torch.exp(s_out) + t_out)
        ldj_sum = (
            ldj_sum + s_out.sum(dim=1) if ldj_sum is not None else s_out.sum(dim=1)
        )

        return y, covariates, ldj_sum

    def inverse(self, y: Tensor, covariates: Tensor) -> Tensor:
        """Goes through the coupling layer in reverse direction

        Args:
            y (Tensor): Inputs
            covariates (Tensor): Covariates

        Returns:
            Tensor: Outputs
        """
        y_m = y * self.mask
        st_input = torch.cat([y_m, covariates], dim=1)

        return y_m + (1 - self.mask) * (
            y * (1 - self.mask) - self.tfun(st_input)
        ) * torch.exp(-self.sfun(st_input))
