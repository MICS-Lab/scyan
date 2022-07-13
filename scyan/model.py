import logging
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from anndata import AnnData
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader

from .data import AdataDataset, RandomSampler, _prepare_data
from .module import ScyanModule
from .utils import _process_pop_sample, _requires_fit, _validate_inputs

log = logging.getLogger(__name__)


class Scyan(pl.LightningModule):
    """
    ```py
    ### Usage example
    import scyan

    adata, marker_pop_matrix = scyan.data.load("aml")

    model = scyan.Scyan(adata, marker_pop_matrix)
    model.fit()
    model.predict()
    ```
    Scyan, a.k.a Single-cell Cytometry Annotation Network.
    It is a wrapper to the ScyanModule that contains the core logic (the loss implementation, the forward function, ...).
    While ScyanModule works on tensors, this class works directly on AnnData objects.

    Attributes:
        adata (AnnData): The provided `adata`
        marker_pop_matrix (pd.Dataframe): The table knowledge
        n_pops (int): Number of populations considered
        hparams (object): Model hyperparameters
        module (ScyanModule): A [ScyanModule][scyan.module.ScyanModule] object
    """

    def __init__(
        self,
        adata: AnnData,
        marker_pop_matrix: pd.DataFrame,
        continuous_covariate_keys: Optional[List[str]] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        hidden_size: int = 16,
        n_hidden_layers: int = 7,
        n_layers: int = 7,
        prior_std: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 16_384,
        alpha_batch_effect: float = 50.0,
        temperature: float = 1.0,
        mmd_max_samples: int = 2048,
        modulo_temp: int = 2,
        max_samples: Optional[int] = 200_000,
        batch_key: Optional[str] = None,
        batch_ref: Union[str, int, None] = None,
    ):
        """
        Args:
            adata: `AnnData` object containing the FCS data. **Warning**: it has to be preprocessed (e.g. `asinh` or `logicle`) and standardised.
            marker_pop_matrix: Dataframe representing the biological knowledge about markers and populations.
            continuous_covariate_keys: Optional list of keys in `adata.obs` that refers to continuous variables to use during the training.
            categorical_covariate_keys: Optional list of keys in `adata.obs` that refers to categorical variables to use during the training.
            hidden_size: Hidden size of the MLP (`s`, `t`).
            n_hidden_layers: Number of hidden layers in the MLP.
            n_layers: Number of coupling layers.
            prior_std: Standard deviation $\sigma$ of the cell-specific random variable $H$.
            lr: Model learning rate.
            batch_size: Model batch size.
            alpha_batch_effect: Weight provided to the batch effect correction loss term.
            temperature: Temperature to favour small populations.
            mmd_max_samples: Maximum number of samples to give to the MMD.
            modulo_temp: At which frequency temperature has to be applied.
            max_samples: Maximum number of samples per epoch.
            batch_key: Key in `adata.obs` that refers to the cell batch variable.
            batch_ref: Batch that will be considered as the reference. By default, choose the batch with the higher number of cells.
        """
        super().__init__()
        self.adata, self.marker_pop_matrix = _validate_inputs(adata, marker_pop_matrix)
        self.continuous_covariate_keys = continuous_covariate_keys or []
        self.categorical_covariate_keys = categorical_covariate_keys or []
        self.n_pops = len(self.marker_pop_matrix)

        self._is_fitted = False
        self._num_workers = 0

        self.save_hyperparameters(
            ignore=[
                "adata",
                "marker_pop_matrix",
                "continuous_covariate_keys",
                "categorical_covariate_keys",
            ]
        )

        self._prepare_data()

        self.module = ScyanModule(
            torch.tensor(marker_pop_matrix.values, dtype=torch.float32),
            self.covariates.shape[1],
            self.other_batches,
            hidden_size,
            n_hidden_layers,
            n_layers,
            prior_std,
            temperature,
            mmd_max_samples,
            self.batch_ref_id,
        )

        log.info(f"Initialized {self}")

    def __repr__(self) -> str:
        if not self.continuous_covariate_keys and not self.categorical_covariate_keys:
            cov_repr = "No covariate provided."
        else:
            cov_repr = f"Covariates: {', '.join(self.continuous_covariate_keys + self.categorical_covariate_keys)}"
        return f"Scyan model with N={self.adata.n_obs} cells, P={self.n_pops} populations and M={self.adata.n_vars} markers. {cov_repr}"

    @property
    def pop_names(self) -> pd.Index:
        """Name of the populations considered in the knowledge table"""
        return self.marker_pop_matrix.index

    @property
    def var_names(self) -> pd.Index:
        """Name of the markers considered in the knowledge table"""
        return self.marker_pop_matrix.columns

    def _prepare_data(self) -> None:
        """Initialize the data and the covariates"""
        if self.hparams.batch_key is None:
            assert (
                self.hparams.batch_ref is None
            ), "To correct batch effect, please profide a batch_key (received only a batch_ref)."
        else:
            batches = self.adata.obs[self.hparams.batch_key]

            if self.hparams.batch_ref is None:
                self.hparams.batch_ref = batches.value_counts().index[0]
                log.warn(
                    f"No batch_ref was provided, using {self.hparams.batch_ref} as reference."
                )

            assert self.hparams.batch_ref in set(
                batches
            ), f"Batch reference '{self.hparams.batch_ref}' is not an existing batch."

        x, covariates, batches, self.other_batches, self.batch_to_id = _prepare_data(
            self.adata,
            self.var_names,
            self.hparams.batch_key,
            self.hparams.batch_ref,
            self.categorical_covariate_keys,
            self.continuous_covariate_keys,
        )

        self.register_buffer("x", x)
        self.register_buffer("covariates", covariates)
        self.register_buffer("batches", batches)

    @property
    def batch_ref_id(self):
        return self.batch_to_id.get(self.hparams.batch_ref)

    def forward(self) -> Tensor:
        """Model forward function

        Returns:
            Tensor: Full dataset latent representation.
        """
        return self.module(self.x, self.covariates)[0]

    @torch.no_grad()
    @_requires_fit
    def sample(
        self,
        n_samples: int,
        covariates_sample: Optional[Tensor] = None,
        pop: Union[str, List[str], int, Tensor, None] = None,
        return_z: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sampling cells by sampling from the prior distribution and going into the normalizing flow.

        Args:
            n_samples: Number of cells to sample.
            covariates_sample: Optional tensor of covariates.
            pop: Populations to sample from. If `str`, then a population name. If `int`, a population index. If `List[str]`, a list of population names. If `Tensor`, a tensor of population indices. By default, samples from all populations.
            return_z: Whether to return the population Tensor.

        Returns:
            Sampled cells expressions and, if `return_z`, the populations associated to these cells.
        """
        z = _process_pop_sample(self, pop)

        if covariates_sample is None:
            # TODO: sample where pop
            indices = random.sample(range(self.adata.n_obs), n_samples)
            covariates_sample = self.covariates[indices]

        return self.module.sample(n_samples, covariates_sample, z=z, return_z=return_z)

    @torch.no_grad()
    @_requires_fit
    def batch_effect_correction(self) -> Tensor:
        """Correct batch effect by going into the latent space, setting the reference covariate to all cells, and then reversing the flow.

        Returns:
            The corrected marker expressions on the original space. **Warning**, as we standardised data for training, this result is standardised too.
        """
        assert (
            self.hparams.batch_key is not None
        ), "Scyan model was trained with no batch_key, thus not correcting batch effect"

        u = self()

        ref_covariate = self.covariates[
            self.adata.obs[self.hparams.batch_key] == self.hparams.batch_ref
        ][0]
        ref_covariates = ref_covariate.repeat((self.adata.n_obs, 1))

        return self.module.inverse(u, ref_covariates)

    def training_step(self, data, _):
        """PyTorch lightning `training_step` implementation (i.e. returning the loss). See [ScyanModule][scyan.module.ScyanModule] for more details."""
        use_temp = self.current_epoch % self.hparams.modulo_temp > 0
        kl, mmd = self.module.losses(*data, use_temp)

        mmd = self.hparams.alpha_batch_effect * mmd
        loss = kl + mmd

        self.log("kl", kl, on_step=True, prog_bar=True)
        self.log("mmd", mmd, on_step=True, prog_bar=True)
        self.log("loss", loss, on_epoch=True, on_step=True)

        return loss

    # def training_epoch_end(self, _):
    #     """PyTorch lightning `training_epoch_end` implementation"""
    #     if "cell_type" in self.adata.obs:  # TODO: remove?
    #         if len(self.x) > 500000:
    #             indices = random.sample(range(len(self.x)), 500000)
    #             x = self.x[indices]
    #             covariates = self.covariates[indices]
    #             labels = self.adata.obs.cell_type[indices]
    #             acc = accuracy_score(
    #                 labels, self.predict(x, covariates, key_added=None).values
    #             )
    #         else:
    #             acc = accuracy_score(
    #                 self.adata.obs.cell_type, self.predict(key_added=None).values
    #             )
    #         self.log("accuracy_score", acc, prog_bar=True)

    @_requires_fit
    @torch.no_grad()
    def predict(
        self,
        x: Optional[Tensor] = None,
        covariates: Optional[Tensor] = None,
        key_added: Optional[str] = "scyan_pop",
    ) -> pd.Series:
        """Model population predictions, i.e. one population is assigned for each cell. Predictions are saved into `adata.obs.scyan_pop` by default.

        Args:
            x: Model inputs.
            covariates: Model covariates.
            key_added: Key added to `model.adata.obs` to save the predictions. If `None`, then the predictions will not be saved.

        Returns:
            pd.Series: Series of predictions.
        """
        df = self.predict_proba(x, covariates)
        populations = df.idxmax(axis=1).astype("category")

        if key_added is not None:
            self.adata.obs[key_added] = pd.Categorical(populations.values)

        missing_pops = self.n_pops - len(populations.cat.categories)
        if missing_pops:
            log.info(
                f"{missing_pops} population(s) were not predicted. It may be due to errors in the knowledge table, the model hyperparameters choices (see https://mics_biomathematics.pages.centralesupelec.fr/biomaths/scyan/parameters/), or maybe these populations are really absent from this dataset."
            )

        return populations

    @_requires_fit
    @torch.no_grad()
    def predict_proba(
        self, x: Optional[Tensor] = None, covariates: Optional[Tensor] = None
    ) -> pd.DataFrame:
        """Soft predictions (i.e. an array of probability per population) for each cell.

        Args:
            x: Model inputs. If `None`, use every cell.
            covariates: Model covariates. If `None`, use every cell.

        Returns:
            Dataframe of probabilities for each population.
        """
        log_probs, *_ = self.module.compute_probabilities(
            self.x if x is None else x,
            self.covariates if covariates is None else covariates,
        )
        probs = torch.softmax(log_probs, dim=1)

        return pd.DataFrame(probs.cpu().numpy(), columns=self.pop_names)

    @property
    @torch.no_grad()
    def pi_hat(self) -> Tensor:  # TODO: remove?
        log_probs, *_ = self.module.compute_probabilities(self.x, self.covariates)
        return torch.softmax(log_probs, dim=1).mean(dim=0)

    def configure_optimizers(self):
        """PyTorch lightning `configure_optimizers` implementation"""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        """PyTorch lightning `train_dataloader` implementation"""
        self.dataset = AdataDataset(self.x, self.covariates, self.batches)
        sampler = RandomSampler(self.dataset, max_samples=self.hparams.max_samples)

        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            num_workers=self._num_workers,
        )

    def fit(
        self,
        max_epochs: int = 100,
        min_delta: float = 1,
        patience: int = 4,
        num_workers: int = 0,
        callbacks: Optional[List[pl.Callback]] = None,
        trainer: Optional[pl.Trainer] = None,
    ) -> "Scyan":
        """Train the `Scyan` model. On interactive Python (e.g. Jupyter Notebooks), training can be interrupted at any time without crashing.

        Args:
            max_epochs: Maximum number of epochs.
            min_delta: min_delta parameters used for `EarlyStopping`. See Pytorch Lightning docs.
            patience: Number of epochs with no loss improvement before to stop training.
            callbacks: Additionnal Pytorch Lightning callbacks.
            trainer: Optional Pytorch Lightning Trainer. **Warning**: it will replace the default Trainer and every other arguments will be unused.

        Returns:
            The trained model itself.
        """
        log.info(f"Training scyan with the following hyperparameters:\n{self.hparams}\n")

        self._num_workers = num_workers

        if trainer is not None:
            trainer.fit(self)
            return self

        esc = EarlyStopping(
            monitor="loss_epoch",
            min_delta=min_delta,
            patience=patience,
            check_on_train_epoch_end=True,
        )
        _callbacks = [esc] + (callbacks or [])

        trainer = pl.Trainer(
            max_epochs=max_epochs, callbacks=_callbacks, log_every_n_steps=10
        )
        trainer.fit(self)

        self._is_fitted = True
        log.info("Successfully ended traning.")

        return self
