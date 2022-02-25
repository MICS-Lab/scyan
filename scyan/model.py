import torch
from torch import Tensor
import pytorch_lightning as pl
from anndata import AnnData
import pandas as pd
from typing import Union, Tuple

from .module.scyan_module import ScyanModule
from .metric import AnnotationMetrics


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
        n_samples: int = 10000,
        n_components: int = 5,
    ):
        super().__init__()
        self.marker_pop_matrix = marker_pop_matrix
        self.adata = adata[:, self.marker_pop_matrix.columns]

        self.save_hyperparameters(ignore=["adata", "marker_pop_matrix"])

        self.x = torch.tensor(self.adata.X)

        self.metric = AnnotationMetrics(self, n_samples, n_components)

        self.module = ScyanModule(
            torch.tensor(marker_pop_matrix.values, dtype=torch.float32),
            hidden_size,
            n_hidden_layers,
            alpha_dirichlet,
            n_layers,
            prior_var,
            lr,
            batch_size,
        )

    def forward(self) -> Tensor:
        return self.module(self.x)[0]

    @torch.no_grad()
    def sample(self, n_samples: int) -> Tuple[Tensor, Tensor]:
        return self.module.sample(n_samples)

    @torch.no_grad()
    def on_train_epoch_start(self):
        self.module._update_log_pi(self.x)

    def training_step(self, x: Tensor, _):
        loss = self.module.loss(x)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def training_epoch_end(self, outputs):
        self.metric()

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
        return pd.DataFrame(predictions.numpy(), columns=self.marker_pop_matrix.index)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.x, batch_size=self.hparams.batch_size)
