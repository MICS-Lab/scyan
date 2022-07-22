from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from . import CouplingLayer


class RealNVP(pl.LightningModule):
    """Normalizing flow module (more specifically the RealNVP transformation $f_{\phi}$).

    Attributes:
        module (nn.Sequential): Sequence of [coupling layers][scyan.module.CouplingLayer].
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_hidden_layers: int,
        n_layers: int,
    ):
        """
        Args:
            input_size: Input size, i.e. number of markers + covariates
            hidden_size: MLP (`s` and `t`) hidden size.
            output_size: Output size, i.e. number of markers.
            n_hidden_layers: Number of hidden layers for the MLP (`s` and `t`).
            n_layers: Number of coupling layers.
        """
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

    def forward(self, x: Tensor, covariates: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward implementation, i.e. $f_{\phi}$.

        Args:
            x: Inputs of size $(B, M)$.
            covariates: Covariates of size $(B, M_c)$

        Returns:
            Tuple of (outputs, covariates, lod_det_jacobian sum)
        """
        return self.module((x, covariates, None))

    def inverse(self, u: Tensor, covariates: Tensor) -> Tensor:
        """Go through the RealNVP in reverse direction, i.e. $f_{\phi}^{-1}$.

        Args:
            u: Latent expressions of size $(B, M)$.
            covariates: Covariates of size $(B, M_c)$

        Returns:
            Outputs of size $(B, M)$.
        """
        for module in reversed(self.module_list):
            u = module.inverse(u, covariates)
        return u
