from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn


class CouplingLayer(pl.LightningModule):
    """One single coupling layer module.

    Attributes:
        sfun (nn.Module): `s` Multi-Layer-Perceptron.
        tfun (nn.Module): `t` Multi-Layer-Perceptron.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_hidden_layers: int,
        mask: Tensor,
    ):
        """
        Args:
            input_size: Input size, i.e. number of markers + covariates
            hidden_size: MLP (`s` and `t`) hidden size.
            output_size: Output size, i.e. number of markers.
            n_hidden_layers: Number of hidden layers for the MLP (`s` and `t`).
            mask: Mask used to separate $x$ into $(x^{(1)}, x^{(2)})$
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
        self, inputs: Tuple[Tensor, Tensor, Optional[Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Coupling layer forward function.

        Args:
            inputs: cell-marker expressions, covariates, lod_det_jacobian sum

        Returns:
            outputs, covariates, lod_det_jacobian sum
        """
        x, covariates, ldj_sum = inputs

        x_m = x * self.mask
        st_input = torch.cat([x_m, covariates], dim=1)

        s_out = self.sfun(st_input)
        t_out = self.tfun(st_input)

        y = x_m + (1 - self.mask) * (x * torch.exp(s_out) + t_out)
        ldj = (s_out * (1 - self.mask)).sum(dim=1)
        ldj_sum = ldj_sum + ldj if ldj_sum is not None else ldj

        return y, covariates, ldj_sum

    def inverse(self, y: Tensor, covariates: Tensor) -> Tensor:
        """Go through the coupling layer in reverse direction.

        Args:
            y: Inputs tensor or size $(B, M)$.
            covariates: Covariates tensor of size $(B, M_c)$.

        Returns:
            Outputs tensor.
        """
        y_m = y * self.mask
        st_input = torch.cat([y_m, covariates], dim=1)

        return y_m + (1 - self.mask) * (
            y * (1 - self.mask) - self.tfun(st_input)
        ) * torch.exp(-self.sfun(st_input))
