import torch
from torch import nn
from torch import Tensor
import pytorch_lightning as pl
from anndata import AnnData
import pandas as pd
from typing import Union
import os

from scyan.module.scyan_module import ScyanModule


class Scyan(pl.LightningModule):
    def __init__(
        self,
        adata: AnnData,
        marker_pop_matrix: pd.DataFrame,
        hidden_size: int = 64,
        n_hidden_layers: int = 1,
        alpha_dirichlet: float = 1.02,
        n_layers: int = 6,
        prior_var: float = 0.2,
        lr: float = 5e-3,
        batch_size: int = 16384,
    ):
        super().__init__()
        self.adata = adata
        self.marker_pop_matrix = marker_pop_matrix
        self.save_hyperparameters(ignore=["adata", "marker_pop_matrix"])

        self.x = torch.tensor(adata.X)

        self.module = ScyanModule(
            torch.tensor(marker_pop_matrix.values.T),
            hidden_size,
            n_hidden_layers,
            alpha_dirichlet,
            n_layers,
            prior_var,
            lr,
            batch_size,
        )

        self.init_weights()

    def training_step(self, x: Tensor, _):
        loss = self.module.loss(x)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    @torch.no_grad()
    def predict(
        self, x: Union[Tensor, None] = None, key_added: str = "scyan_pop"
    ) -> pd.Series:
        df = self.predict_proba(x=x)
        populations = df.idxmax(axis=1).astype("category")

        if key_added:
            self.adata.obs[key_added] = populations.values

        return populations

    @torch.no_grad()
    def predict_proba(self, x: Union[Tensor, None] = None) -> pd.DataFrame:
        predictions, *_ = self.module.compute_probabilities(self.x if x is None else x)
        return pd.DataFrame(predictions.numpy(), columns=self.marker_pop_matrix.columns)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.x, batch_size=self.hparams.batch_size)

    def init_weights(self) -> None:
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        for k in [-2, -1]:
            self.module.real_nvp.module_list[k].tfun[-2].register_forward_hook(
                get_activation(k)
            )

            x_ = torch.zeros((2, self.adata.n_vars))
            x_[1] = 1
            h_neg, h_pos = self.module(x_)[0]
            t_neg, t_pos = activation[k]
            x = t_pos - t_neg
            u = 2 - h_pos + h_neg
            delta_weight = (x / x.dot(x))[None, :] * u[:, None]
            b = 1 - h_pos - t_pos @ delta_weight.T

            linear = self.module.real_nvp.module_list[k].tfun[-1]
            linear.weight = nn.Parameter(
                linear.weight + delta_weight, requires_grad=True
            )
            linear.bias = nn.Parameter(linear.bias + b, requires_grad=True)
