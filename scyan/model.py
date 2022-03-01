import torch
from torch import Tensor
import pytorch_lightning as pl
from anndata import AnnData
import pandas as pd
from typing import Union, Tuple, List
import numpy as np
import random

from .module.scyan_module import ScyanModule
from .metric import AnnotationMetrics
from .data import AdataDataset


class Scyan(pl.LightningModule):
    def __init__(
        self,
        adata: AnnData,
        marker_pop_matrix: pd.DataFrame,
        continuous_covariate_keys: List[str] = [],
        categorical_covariate_keys: List[str] = [],
        hidden_size: int = 64,
        n_hidden_layers: int = 1,
        alpha_dirichlet: float = 1.02,
        n_layers: int = 6,
        prior_std: float = 0.5,
        prior_std_nan: float = 2,
        lr: float = 5e-3,
        batch_size: int = 16384,
        n_samples: int = 100,
        n_components: int = 5,
    ):
        super().__init__()
        self.marker_pop_matrix = marker_pop_matrix
        self.adata = adata[:, self.marker_pop_matrix.columns].copy()
        self.continuous_covariate_keys = continuous_covariate_keys
        self.categorical_covariate_keys = categorical_covariate_keys

        self.save_hyperparameters(
            ignore=[
                "adata",
                "marker_pop_matrix",
                "continuous_covariate_keys",
                "categorical_covariate_keys",
            ]
        )

        self.init_dataset()
        self.metric = AnnotationMetrics(self, n_samples, n_components)

        self.module = ScyanModule(
            torch.tensor(marker_pop_matrix.values, dtype=torch.float32),
            self.covariates.shape[1],
            hidden_size,
            n_hidden_layers,
            alpha_dirichlet,
            n_layers,
            prior_std,
            prior_std_nan,
            lr,
            batch_size,
        )

    def init_dataset(self) -> None:
        self.x = torch.tensor(self.adata.X)

        for key in self.categorical_covariate_keys:  # enforce dtype category
            self.adata.obs[key] = self.adata.obs[key].astype("category")

        categorical_covariate_embedding = (
            pd.get_dummies(self.adata.obs[self.categorical_covariate_keys]).values
            if self.categorical_covariate_keys
            else np.empty((self.adata.n_obs, 0))
        )

        continuous_covariate_embedding = (
            self.adata.obs[self.continuous_covariate_keys].values
            if self.continuous_covariate_keys
            else np.empty((self.adata.n_obs, 0))
        )

        self.adata.obsm["covariates"] = np.concatenate(
            [
                categorical_covariate_embedding,
                continuous_covariate_embedding,
            ],
            axis=1,
        )

        self.covariates = torch.tensor(
            self.adata.obsm["covariates"],
            dtype=torch.float32,
        )

        self.dataset = AdataDataset(self.x, self.covariates)

    def forward(self) -> Tensor:
        return self.module(self.x, self.covariates)[0]

    @torch.no_grad()
    def sample(self, n_samples: int) -> Tuple[Tensor, Tensor]:
        # TODO: allow to choose sampling method for covariates
        indices = torch.tensor(random.sample(range(len(self.x)), n_samples))
        covariates_sample = self.covariates[indices]
        return self.module.sample(n_samples, covariates_sample)

    @torch.no_grad()
    def on_train_epoch_start(self):
        self.module._update_log_pi(self.x, self.covariates)

    def training_step(self, batch, _):
        loss = self.module.loss(*batch)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def training_epoch_end(self, _):
        self.metric()

    @torch.no_grad()
    def predict(
        self, x=None, covariates=None, key_added: str = "scyan_pop"
    ) -> pd.Series:
        df = self.predict_proba(x, covariates)
        populations = df.idxmax(axis=1).astype("category")

        if key_added:
            self.adata.obs[key_added] = populations.values

        return populations

    @torch.no_grad()
    def predict_proba(self, x=None, covariates=None) -> pd.DataFrame:
        predictions, *_ = self.module.compute_probabilities(
            self.x if x is None else x,
            self.covariates if covariates is None else covariates,
        )
        return pd.DataFrame(predictions.numpy(), columns=self.marker_pop_matrix.index)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.hparams.batch_size
        )
