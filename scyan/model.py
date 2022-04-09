import torch
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from anndata import AnnData
import pandas as pd
from typing import Union, Tuple, List
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
import logging

from .module import ScyanModule
from .metric import AnnotationMetrics
from .data import AdataDataset
from .utils import _process_pop_sample

log = logging.getLogger(__name__)


class Scyan(pl.LightningModule):
    def __init__(
        self,
        adata: AnnData,
        marker_pop_matrix: pd.DataFrame,
        continuous_covariate_keys: List[str] = [],
        categorical_covariate_keys: List[str] = [],
        hidden_size: int = 16,
        n_hidden_layers: int = 5,
        ratio_threshold: float = 5e-4,
        n_layers: int = 5,
        prior_std: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 16384,
        n_samples: int = 100,  # TODO: remove
        n_components: int = 5,  # TODO: remove
        alpha: float = 1.0,
    ):
        """Scyan model

        Args:
            adata (AnnData): AnnData object containing the FCS data
            marker_pop_matrix (pd.DataFrame): Marker-population table (expert knowledge)
            continuous_covariate_keys (List[str], optional): List of continuous covariable in adata.obs. Defaults to [].
            categorical_covariate_keys (List[str], optional): List of categorical covariable in adata.obs. Defaults to [].
            hidden_size (int, optional): Neural networks (s and t) hidden size. Defaults to 64.
            n_hidden_layers (int, optional): Neural networks (s and t) number of hidden layers. Defaults to 1.
            ratio_threshold (float, optional): Minimum ratio of cells to be observed for each population. Defaults to 1e-4.
            n_layers (int, optional): Number of coupling layers. Defaults to 6.
            prior_std (float, optional): Standard deviation of the base distribution (H). Defaults to 0.25.
            lr (float, optional): Learning rate. Defaults to 5e-3.
            batch_size (int, optional): Batch size. Defaults to 16384.
            n_samples (int, optional): TODO: remove. Defaults to 100.
            n_components (int, optional): TODO: remove. Defaults to 5.
            alpha (float, optional): Constraint term weight in the loss function. Defaults to 1.0.
        """
        super().__init__()
        log.info("The provided adata is copied, prefer to use model.adata from now on.")

        self.marker_pop_matrix = marker_pop_matrix
        self.adata = adata[:, self.marker_pop_matrix.columns].copy()
        self.continuous_covariate_keys = list(continuous_covariate_keys)
        self.categorical_covariate_keys = list(categorical_covariate_keys)
        self.n_pops = len(self.marker_pop_matrix.index)

        self.save_hyperparameters(
            ignore=[
                "adata",
                "marker_pop_matrix",
                "continuous_covariate_keys",
                "categorical_covariate_keys",
            ]
        )

        self.init_data_covariates()

        self.module = ScyanModule(
            torch.tensor(marker_pop_matrix.values, dtype=torch.float32),
            self.covariates.shape[1],
            hidden_size,
            n_hidden_layers,
            n_layers,
            prior_std,
            lr,
            batch_size,
            ratio_threshold,
            alpha,
        )

        self.metric = AnnotationMetrics(self, n_samples, n_components)

        log.info(f"Initialized {self}")

    def __repr__(self) -> str:
        if not self.continuous_covariate_keys and not self.categorical_covariate_keys:
            cov_repr = "No covariate provided."
        else:
            cov_repr = f"Covariates: {', '.join(self.continuous_covariate_keys + self.categorical_covariate_keys)}"
        return f"Scyan model with N={self.adata.n_obs} cells, P={self.n_pops} populations and M={self.adata.n_vars} markers. {cov_repr}"

    def init_data_covariates(self) -> None:
        """Initializes the data and the covariates"""
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

    def forward(self) -> Tensor:
        """Model forward function

        Returns:
            Tensor: Dataset latent representation
        """
        return self.module(self.x, self.covariates)[0]

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        covariates_sample: Union[Tensor, None] = None,
        pop: Union[str, List[str], int, Tensor, None] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample cells

        Args:
            n_samples (int): Number of cells to be sampled
            covariates_sample (Union[Tensor, None], optional): Sample of cobariates. Defaults to None.
            pop (Union[str, List[str], int, Tensor, None], optional): Sample of population. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Pair of (cell expressions, population)
        """
        z_pop = _process_pop_sample(self, pop)

        if covariates_sample is None:
            indices = torch.tensor(
                random.sample(range(len(self.x)), n_samples)
            )  # TODO: sample where pop
            covariates_sample = self.covariates[indices]

        return self.module.sample(n_samples, covariates_sample, z_pop=z_pop)

    def training_step(self, batch, _):
        """PyTorch lightning training_step implementation"""
        loss = self.module.loss(*batch)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def training_epoch_end(self, _):
        """PyTorch lightning training_epoch_end implementation"""
        self.metric()

    @torch.no_grad()
    def predict(
        self,
        x: Union[Tensor, None] = None,
        covariates: Union[Tensor, None] = None,
        key_added: str = "scyan_pop",
    ) -> pd.Series:
        """Model predictions

        Args:
            x (Union[Tensor, None], optional): Model inputs. Defaults to None.
            covariates (Union[Tensor, None], optional): Model covariates. Defaults to None.
            key_added (str, optional): Key added to model.adata.obs. Defaults to "scyan_pop".

        Returns:
            pd.Series: Series of predictions
        """
        df = self.predict_proba(x, covariates)
        populations = df.idxmax(axis=1).astype("category")

        if key_added:
            self.adata.obs[key_added] = populations.values

        return populations

    def knn_predict(self, n_neighbors: int = 64, key_added: str = "scyan_knn_pop"):
        """Model k-neirest-neighbors predictions

        Args:
            n_neighbors (int, optional): Number of neirest neighbors. Defaults to 64.
            key_added (str, optional): Key added to model.adata.obs. Defaults to "scyan_knn_pop".
        """
        assert (
            "scyan_pop" in self.adata.obs
        ), "Key scyan_pop must be in model.adata.obs - Have you run model.predict before?"

        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        neigh.fit(self.adata.X, self.adata.obs.scyan_pop)
        self.adata.obs[key_added] = neigh.predict(self.adata.X)

    @torch.no_grad()
    def predict_proba(
        self, x: Union[Tensor, None] = None, covariates: Union[Tensor, None] = None
    ) -> pd.DataFrame:
        """Model proba predictions for each population

        Args:
            x (Union[Tensor, None], optional): Model inputs. Defaults to None.
            covariates (Union[Tensor, None], optional): Model covariates. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe of probabilities for each population
        """
        predictions, *_ = self.module.compute_probabilities(
            self.x if x is None else x,
            self.covariates if covariates is None else covariates,
        )
        return pd.DataFrame(predictions.numpy(), columns=self.marker_pop_matrix.index)

    @property
    @torch.no_grad()
    def pi_hat(self) -> Tensor:
        """Model observed population weights

        Returns:
            Tensor: Population weights
        """
        predictions, *_ = self.module.compute_probabilities(self.x, self.covariates)
        return predictions.mean(dim=0)

    def configure_optimizers(self):
        """PyTorch lightning configure_optimizers implementation"""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        """PyTorch lightning train_dataloader implementation"""
        print(self.device)
        self.x = self.x.to(self.device)
        self.covariates = self.covariates.to(self.device)
        self.dataset = AdataDataset(self.x, self.covariates)

        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.hparams.batch_size
        )

    def fit(
        self,
        max_epochs: int = 100,
        min_delta: float = 0.5,
        patience: int = 4,
        callbacks: List[pl.Callback] = [],
        trainer: Union[pl.Trainer, None] = None,
    ) -> None:
        """Train Scyan

        Args:
            max_epochs (int, optional): Maximum number of epochs. Defaults to 100.
            min_delta (float, optional): min_delta parameters used for EarlyStopping. See Pytorch Lightning docs. Defaults to 0.5.
            patience (int, optional): Number of epochs with no loss improvement before to stop training. Defaults to 4.
            callbacks (List[pl.Callback], optional): Additionnal Pytorch Lightning callbacks.
            trainer (Union[pl.Trainer, None], optional): Pytorch Lightning Trainer. Warning: it will replace the default Trainer and all ther arguments will be unused. Defaults to None.
        """
        if trainer is not None:
            trainer.fit(self)
            return

        esc = EarlyStopping(
            monitor="loss_epoch",
            min_delta=min_delta,
            patience=patience,
            check_on_train_epoch_end=True,
        )
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[esc] + callbacks)
        trainer.fit(self)
