import importlib
import logging
import random
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from anndata import AnnData
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .data import AdataDataset, RandomSampler, _prepare_data
from .module import ScyanModule
from .utils import (
    _add_level_predictions,
    _get_pop_index,
    _process_pop_sample,
    _requires_fit,
    _validate_inputs,
)

log = logging.getLogger(__name__)


class Scyan(pl.LightningModule):
    """
    Scyan, a.k.a Single-cell Cytometry Annotation Network.
    It is a wrapper to the ScyanModule that contains the core logic (the loss implementation, the forward function, ...).
    While ScyanModule works on tensors, this class works directly on AnnData objects.

    Attributes:
        adata (AnnData): The provided `adata`
        marker_pop_matrix (pd.Dataframe): The table knowledge
        n_pops (int): Number of populations considered, i.e. $P$
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
        prior_std: float = 0.3,
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
            adata: `AnnData` object containing the FCS data ($N$ cells). **Warning**: it has to be preprocessed (e.g. `asinh` or `logicle`) and standardised.
            marker_pop_matrix: Dataframe of shape $(P, M)$ representing the biological knowledge about markers and populations.
            continuous_covariate_keys: Optional list of keys in `adata.obs` that refers to continuous variables to use during the training.
            categorical_covariate_keys: Optional list of keys in `adata.obs` that refers to categorical variables to use during the training.
            hidden_size: Hidden size of the MLP (`s`, `t`).
            n_hidden_layers: Number of hidden layers in the MLP.
            n_layers: Number of coupling layers.
            prior_std: Standard deviation $\sigma$ of the cell-specific random variable $H$.
            lr: Model learning rate.
            batch_size: Model batch size.
            alpha_batch_effect: Weight provided to the batch effect correction loss term.
            temperature: Temperature to favor small populations.
            mmd_max_samples: Maximum number of samples to give to the MMD.
            modulo_temp: At which frequency temperature has to be applied.
            max_samples: Maximum number of samples per epoch.
            batch_key: Key in `adata.obs` referring to the cell batch variable.
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
        return f"Scyan model with N={self.adata.n_obs} cells, P={self.n_pops} populations and M={len(self.var_names)} markers. {cov_repr}"

    @property
    def pop_names(self) -> pd.Index:
        """Name of the populations considered in the knowledge table"""
        return self.marker_pop_matrix.index.get_level_values(0)

    @property
    def var_names(self) -> pd.Index:
        """Name of the markers considered in the knowledge table"""
        return self.marker_pop_matrix.columns

    @property
    def level_names(self):
        """All population hierarchical level names, if existing."""
        if not isinstance(self.marker_pop_matrix.index, pd.MultiIndex):
            log.warn(
                "The provided knowledge table has no population hierarchical level. See: https://mics-lab.github.io/scyan/tutorials/advanced/#hierarchical-population-display"
            )
            return []

        return list(self.marker_pop_matrix.index.names[1:])

    def pops(
        self,
        level: Union[str, int, None] = None,
        parent_of: Optional[str] = None,
        children_of: Optional[str] = None,
    ) -> Union[set, str]:
        """Get the name of the populations that match a given contraint (only available if a hierarchical populations are provided, see [this tutorial](https://mics-lab.github.io/scyan/tutorials/advanced/#hierarchical-population-display)). If `level` is provided, returns all populations at this level. If `parent_of`, returns the parent of the given pop. If `children_of`, returns the children of the given pop.

        !!! note
            If you want to get the names of the leaves populations, you can simply use `model.pop_names`, which is equivalent to `model.pops(level=0)`.

        Args:
            level: If `str`, level name. If `int`, level index (0 corresponds to leaves).
            parent_of: name of the population of which we want to get the parent in the tree.
            children_of: name of the population of which we want to get the children populations in the tree.

        Returns:
            Set of all populations that match the contraint, or one name if `parent_of` is not `None`.
        """

        assert (
            self.level_names
        ), "The provided knowledge table has no population hierarchical level. See the doc."

        assert (
            sum(arg is not None for arg in [level, parent_of, children_of]) == 1
        ), "One and exactly one argument has to be provided. Choose one among 'level', 'parent_of', and 'children_of'."

        if level is not None:
            assert (
                isinstance(level, int) or level in self.level_names
            ), f"Level has to be one of [{', '.join(self.level_names)}]. Found {level}."

            return set(self.marker_pop_matrix.index.get_level_values(level))

        name = parent_of or children_of
        index = _get_pop_index(name, self.marker_pop_matrix)
        where = self.marker_pop_matrix.index.get_level_values(index) == name

        if children_of is not None:
            if index == 0:
                return set()
            return set(self.marker_pop_matrix.index.get_level_values(index - 1)[where])

        assert (
            index < self.marker_pop_matrix.index.nlevels - 1
        ), "Can not get parent of highest level population."

        return self.marker_pop_matrix.index.get_level_values(index + 1)[where][0]

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

    def forward(self, indices: Optional[np.ndarray] = None) -> Tensor:
        """Model forward function (not used during training, see `training_step`instead).

        !!! note
            The core logic and the functions used for training are implemented in [ScyanModule][scyan.module.ScyanModule] (or see [scyan.Scyan.training_step][]).

        Args:
            indices: Indices of the cells to forward. By default, use all cells.

        Returns:
            Latent representation of the considered cells.
        """
        if indices is None:
            indices = np.arange(self.adata.n_obs)

        x = self.x[indices]
        cov = self.covariates[indices]

        return self.dataset_apply(lambda *batch: self.module(*batch)[0], (x, cov))

    def _repeat_ref_covariates(self, k: Optional[int] = None) -> Tensor:
        """Repeat the covariates from the reference batch along axis 0.

        Args:
            k: Number of repetitions. By default, the number of cells $N$.

        Returns:
            A tensor of covariates of shape $(k, M_c)$
        """
        n_repetitions = self.adata.n_obs if k is None else k

        ref_covariate = self.covariates[
            self.adata.obs[self.hparams.batch_key] == self.hparams.batch_ref
        ][0]
        return ref_covariate.repeat((n_repetitions, 1))

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
            covariates_sample: Optional tensor of covariates. If not provided: if the model was trained for batch correction then the reference covariates are repeated, else we sample from all the covariates.
            pop: Optional population to sample from (by default, sample from all populations). If `str`, then a population name. If `int`, a population index. If `List[str]`, a list of population names. If `Tensor`, a tensor of population indices.
            return_z: Whether to return the population `Tensor` (i.e., a tensor of population indices, whose order corresponds to `model.pop_names`).

        Returns:
            Sampled cells expressions and, if `return_z`, the populations associated to these cells.
        """
        z = _process_pop_sample(self, pop)

        if covariates_sample is None:
            if self.hparams.batch_key is None:
                indices = random.sample(range(self.adata.n_obs), n_samples)
                covariates_sample = self.covariates[indices]
            else:
                covariates_sample = self._repeat_ref_covariates(n_samples)

        return self.module.sample(n_samples, covariates_sample, z=z, return_z=return_z)

    @torch.no_grad()
    @_requires_fit
    def batch_effect_correction(self) -> Tensor:
        """Correct batch effect by going into the latent space, setting the reference covariate to all cells, and then reversing the flow.

        !!! warning
            As we standardised data for training, the resulting tensor is standardised too. You can save the tensor as a numpy layer of `adata` and use [scyan.tools.unscale][] to unscale it.

        Returns:
            The corrected marker expressions on the original space.
        """
        assert (
            self.hparams.batch_key is not None
        ), "Scyan model was trained with no batch_key, thus not correcting batch effect"

        u = self()
        ref_covariates = self._repeat_ref_covariates()

        return self.dataset_apply(self.module.inverse, (u, ref_covariates))

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

    def predict_step(self, data: Tuple[Tensor], _) -> Tensor:
        """Compute log probabilities for each population and for each cell. Do not use directly, prefer `predict` or `predict_proba` instead.

        Args:
            data: One mini-batch of data representing $B$ cells.

        Returns:
            Log probabilities as a tensor of shape $(B, P)$.
        """
        x, cov, _ = data
        log_probs, *_ = self.module.compute_probabilities(x, cov)
        return log_probs

    @_requires_fit
    @torch.no_grad()
    def predict(
        self,
        key_added: Optional[str] = "scyan_pop",
        add_levels: bool = True,
    ) -> pd.Series:
        """Model population predictions, i.e. one population is assigned for each cell. Predictions are saved in `adata.obs.scyan_pop` by default.

        Args:
            key_added: Key added to `model.adata.obs` to save the predictions. If `None`, then the predictions will not be saved.
            add_levels: If `True`, and if [hierarchical population names](../../tutorials/advanced/#hierarchical-population-display) were provided, then it also saves the prediction for every population level.

        Returns:
            Population predictions (pandas `Series` of length $N$).
        """
        df = self.predict_proba()
        populations = df.idxmax(axis=1).astype("category")

        if key_added is not None:
            self.adata.obs[key_added] = pd.Categorical(populations)
            if add_levels and isinstance(self.marker_pop_matrix.index, pd.MultiIndex):
                _add_level_predictions(self, key_added)

        missing_pops = self.n_pops - len(populations.cat.categories)
        if missing_pops:
            log.info(
                f"{missing_pops} population(s) were not predicted. It may be due to:\n  - Errors in the knowledge table (see https://mics-lab.github.io/scyan/advanced/advice/)\n  - The model hyperparameters choice (see https://mics-lab.github.io/scyan/advanced/parameters/)\n  - Or maybe these populations are really absent from this dataset."
            )

        return populations

    @_requires_fit
    @torch.no_grad()
    def predict_proba(self) -> pd.DataFrame:
        """Soft predictions (i.e. an array of probability per population) for each cell.

        Returns:
            Dataframe of shape `(N, P)` with probabilities for each population.
        """
        probs = torch.softmax(
            torch.cat(self.trainer.predict(self, self.predict_dataloader()), dim=0), dim=1
        )

        return pd.DataFrame(probs.cpu().numpy(), columns=self.pop_names)

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

    def predict_dataloader(self):
        """PyTorch lightning `predict_dataloader` implementation"""
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self._num_workers,
        )

    def dataset_apply(self, func: Callable, data: Tuple[Tensor] = None) -> Tensor:
        """Apply a function on a dataset using a PyTorch DataLoader and with a progress bar display. It concatenates the results along the first axis.

        Args:
            func: Function to be applied. It takes a batch, and returns a Tensor.
            data: Optional tuple of tensors to load from (we create a TensorDataset). By default, uses the main dataset.

        Returns:
            Tensor of concatenated results.
        """
        if importlib.util.find_spec("ipywidgets") is not None:
            from tqdm.auto import tqdm as _tqdm
        else:
            from tqdm import tqdm as _tqdm

        if not data:
            loader = self.predict_dataloader()
        else:
            loader = DataLoader(
                TensorDataset(*data),
                batch_size=self.hparams.batch_size,
                num_workers=self._num_workers,
            )

        return torch.cat(
            [func(*batch) for batch in _tqdm(loader, desc="DataLoader")], dim=0
        )

    def fit(
        self,
        max_epochs: int = 100,
        min_delta: float = 1,
        patience: int = 2,
        num_workers: int = 0,
        callbacks: Optional[List[pl.Callback]] = None,
        trainer: Optional[pl.Trainer] = None,
    ) -> "Scyan":
        """Train the `Scyan` model. On interactive Python (e.g., Jupyter Notebooks), training can be interrupted at any time without crashing.

        !!! note
            Depending on your machine, you may have a warning about some performance issues. You can simply set `num_workers` to the number indicated by the warning.

        Args:
            max_epochs: Maximum number of epochs.
            min_delta: min_delta parameters used for `EarlyStopping`. See Pytorch Lightning docs.
            patience: Number of epochs with no loss improvement before stopping training.
            num_workers: Pytorch DataLoader `num_workers` argument, i.e. how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
            callbacks: Additional Pytorch Lightning callbacks.
            trainer: Optional Pytorch Lightning Trainer. **Warning**: it will replace the default Trainer, and every other argument will be unused.

        Returns:
            The trained model itself.
        """
        log.info(f"Training scyan with the following hyperparameters:\n{self.hparams}\n")

        self._num_workers = num_workers

        if trainer is None:
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

        self.trainer = trainer
        self.trainer.fit(self)

        self._is_fitted = True
        log.info("Successfully ended traning.")

        return self
