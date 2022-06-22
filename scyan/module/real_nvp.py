import torch
from torch import Tensor
from torch import nn
from typing import Tuple
import pytorch_lightning as pl
import numpy as np

from .coupling_layer import CouplingLayer


class RealNVP(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_hidden_layers: int,
        n_layers: int,
    ):
        """Complete RealNVP module containg many coupling layers

        Args:
            input_size (int): Input size, i.e. number of markers + covariates
            hidden_size (int): Neural networks hidden size
            output_size (int): Output size, i.e. number of markers
            n_hidden_layers (int): Number of s and t hidden layers
            n_layers (int): number of coupling layers
        """
        super().__init__()
        self.build_masks(output_size, n_layers)

        self.module_list = nn.ModuleList(
            [
                CouplingLayer(
                    input_size,
                    hidden_size,
                    output_size,
                    n_hidden_layers,
                    self.masks[i],
                )
                for i in range(n_layers)
            ]
        )
        self.module = nn.Sequential(*self.module_list)

    def build_masks(self, output_size: int, n_layers: int):
        self.masks = []
        for _ in range((n_layers + 1) // 2):
            mask = np.array([j % 2 for j in range(output_size)])
            np.random.shuffle(mask)
            mask = torch.tensor(mask)
            self.masks.extend([mask, 1 - mask])

    def forward(self, x: Tensor, covariates: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward implementation

        Args:
            x (Tensor): Inputs
            covariates (Tensor): Covariates

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple of (outputs, covariates, lod_det_jacobian sum)
        """
        return self.module((x, covariates, None))

    def inverse(self, u: Tensor, covariates: Tensor) -> Tensor:
        """Goes through the RealNVP in reverse direction

        Args:
            u (Tensor): Inputs
            covariates (Tensor): Covariates

        Returns:
            Tensor: Outputs
        """
        for module in reversed(self.module_list):
            u = module.inverse(u, covariates)
        return u
