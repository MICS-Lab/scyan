import torch
from torch import Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from anndata import AnnData
import pandas as pd
from typing import Union, Tuple, List
import numpy as np
import random
from sklearn.metrics import accuracy_score
import logging

from .module import ScyanModule
from .data import AdataDataset, RandomSampler, _prepare_data
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
        n_hidden_layers: int = 7,
        n_layers: int = 7,
        prior_std: float = 0.15,
        lr: float = 1e-3,
        batch_size: int = 16384,
        alpha: float = 200,
        alpha_batch_effect: float = 200,
        temperature_mmd: float = 1.5,
        temp_lr_weights: float = 8,
        mmd_max_samples: int = 2048,
        max_samples: Union[int, None] = None,
        batch_key: Union[str, None] = None,
        batch_ref: Union[str, int, None] = None,
    ):
        """Scyan model

        Args:
            adata (AnnData): AnnData object containing the FCS data
            marker_pop_matrix (pd.DataFrame): Marker-population table (expert knowledge)
            continuous_covariate_keys (List[str], optional): List of continuous covariable in adata.obs. Defaults to [].
            categorical_covariate_keys (List[str], optional): List of categorical covariable in adata.obs. Defaults to [].
            hidden_size (int, optional): Neural networks (s and t) hidden size. Defaults to 64.
            n_hidden_layers (int, optional): Neural networks (s and t) number of hidden layers. Defaults to 1.
            n_layers (int, optional): Number of coupling layers. Defaults to 6.
            prior_std (float, optional): Standard deviation of the base distribution (H). Defaults to 0.25.
            lr (float, optional): Learning rate. Defaults to 5e-3.
            batch_size (int, optional): Batch size. Defaults to 16384.
            alpha (float, optional): Constraint term weight in the loss function. Defaults to 1.0.
        """
        super().__init__()
        log.info("The provided adata is copied, prefer to use model.adata from now on.")

        self.marker_pop_matrix = marker_pop_matrix
        self.adata = adata[:, self.marker_pop_matrix.columns].copy()
        self.continuous_covariate_keys = list(continuous_covariate_keys)
        self.categorical_covariate_keys = list(categorical_covariate_keys)
        self.n_pops = len(self.marker_pop_matrix.index)
        self._is_fitted = False

        self.save_hyperparameters(
            ignore=[
                "adata",
                "marker_pop_matrix",
                "continuous_covariate_keys",
                "categorical_covariate_keys",
            ]
        )

        self.prepare_data()

        self.module = ScyanModule(
            torch.tensor(marker_pop_matrix.values, dtype=torch.float32),
            self.covariates.shape[1],
            hidden_size,
            n_hidden_layers,
            n_layers,
            prior_std,
            alpha,
            temperature_mmd,
            temp_lr_weights,
            mmd_max_samples,
        )

        log.info(f"Initialized {self}")

    def __repr__(self) -> str:
        if not self.continuous_covariate_keys and not self.categorical_covariate_keys:
            cov_repr = "No covariate provided."
        else:
            cov_repr = f"Covariates: {', '.join(self.continuous_covariate_keys + self.categorical_covariate_keys)}"
        return f"Scyan model with N={self.adata.n_obs} cells, P={self.n_pops} populations and M={self.adata.n_vars} markers. {cov_repr}"

    @property
    def pop_names(self):
        return self.marker_pop_matrix.index

    @property
    def var_names(self):
        return self.adata.var_names

    def prepare_data(self) -> None:
        """Initializes the data and the covariates"""
        x, covariates, batch = _prepare_data(
            self.adata,
            self.hparams.batch_key,
            self.categorical_covariate_keys,
            self.continuous_covariate_keys,
        )

        self.register_buffer("x", x)
        self.register_buffer("covariates", covariates)
        self.register_buffer("batch", batch)

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
        return_z: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sample cells

        Args:
            n_samples (int): Number of cells to be sampled
            covariates_sample (Union[Tensor, None], optional): Sample of cobariates. Defaults to None.
            pop (Union[str, List[str], int, Tensor, None], optional): Sample of population. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Pair of (cell expressions, population)
        """
        z = _process_pop_sample(self, pop)

        if covariates_sample is None:
            # TODO: sample where pop
            indices = random.sample(range(len(self.x)), n_samples)
            covariates_sample = self.covariates[indices]

        return self.module.sample(n_samples, covariates_sample, z=z, return_z=return_z)

    def training_step(self, data, _):
        """PyTorch lightning training_step implementation"""
        kl, weighted_mmd, batch_mmd = self.module.losses(*data, self.hparams.batch_ref)
        batch_mmd = self.hparams.alpha_batch_effect * batch_mmd
        loss = kl + weighted_mmd + batch_mmd

        self.log("kl", kl, on_step=True, prog_bar=True)
        self.log("mmd", weighted_mmd, on_step=True, prog_bar=True)
        self.log("batch_mmd", batch_mmd, on_step=True, prog_bar=True)
        self.log("loss", loss, on_epoch=True, on_step=True)

        return loss

    def training_epoch_end(self, _):
        """PyTorch lightning training_epoch_end implementation"""
        if "cell_type" in self.adata.obs:
            if len(self.x) > 500000:
                indices = random.sample(range(len(self.x)), 500000)
                x = self.x[indices]
                covariates = self.covariates[indices]
                labels = self.adata.obs.cell_type[indices]
                acc = accuracy_score(
                    labels, self.predict(x, covariates, key_added=None).values
                )
            else:
                acc = accuracy_score(
                    self.adata.obs.cell_type, self.predict(key_added=None).values
                )
            self.log("accuracy_score", acc, prog_bar=True)

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
            self.adata.obs[key_added] = pd.Categorical(populations.values)

        return populations

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
    def pi_hat(self) -> Tensor:  # TODO: remove?
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
        self.dataset = AdataDataset(self.x, self.covariates, self.batch)
        sampler = RandomSampler(self.dataset, max_samples=self.hparams.max_samples)

        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
        )

    def fit(
        self,
        max_epochs: int = 100,
        min_delta: float = 1,
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
        log.info(f"Training scyan with the following hyperparameters:\n{self.hparams}\n")

        if trainer is not None:
            trainer.fit(self)
            return self

        esc = EarlyStopping(
            monitor="loss_epoch",
            min_delta=min_delta,
            patience=patience,
            check_on_train_epoch_end=True,
        )
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[esc] + callbacks)
        trainer.fit(self)

        self._is_fitted = True

        return self
