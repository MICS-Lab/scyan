import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from . import utils
from .data import _prepare_data
from .module.distribution import PriorDistribution

log = logging.getLogger(__name__)


class Baseline:
    """Baseline of Scyan (i.e., without the normalizing flow)."""

    def __init__(
        self,
        adata: AnnData,
        table: pd.DataFrame,
        prior_std: float = 0.3,
    ):
        """
        Args:
            adata: `AnnData` object containing the FCS data of $N$ cells. **Warning**: it has to be preprocessed (e.g. `asinh` or `logicle`) and scaled (see https://mics-lab.github.io/scyan/tutorials/preprocessing/).
            table: Dataframe of shape $(P, M)$ representing the biological knowledge about markers and populations. The columns names corresponds to marker that must be in `adata.var_names`.
            prior_std: Standard deviation $\sigma$ of the cell-specific random variable $H$.
        """
        super().__init__()
        self.adata, self.table, self.continuum_markers = utils._validate_inputs(
            adata, table, []
        )
        self.prior_std = prior_std
        self.n_pops, self.n_markers = self.table.shape

        self._prepare_data()

        self.prior = PriorDistribution(
            torch.tensor(table.values, dtype=torch.float32),
            torch.full((self.n_markers,), False),
            self.prior_std,
            self.n_markers,
        )

        log.info(f"Initialized {self}")

    @property
    def pop_names(self) -> pd.Index:
        """Name of the populations considered in the knowledge table"""
        return self.table.index.get_level_values(0)

    @property
    def var_names(self) -> pd.Index:
        """Name of the markers considered in the knowledge table"""
        return self.table.columns

    def __repr__(self) -> str:
        return f"Baseline model with N={self.adata.n_obs} cells, P={self.n_pops} populations and M={len(self.var_names)} markers."

    def _prepare_data(self) -> None:
        """Initialize the data"""
        self.x, _ = _prepare_data(
            self.adata,
            self.table.columns,
            None,
            [],
            [],
        )

    def predict(
        self,
        key_added: Optional[str] = "baseline_pop",
        add_levels: bool = True,
        log_prob_th: float = -50,
    ) -> pd.Series:
        """Model population predictions, i.e. one population is assigned for each cell. Predictions are saved in `adata.obs.scyan_pop` by default.

        !!! note
            Some cells may not be annotated, if their log probability is lower than `log_prob_th` for all populations. Then, the predicted label will be `np.nan`.

        Args:
            key_added: Column name used to save the predictions in `adata.obs`. If `None`, then the predictions will not be saved.
            add_levels: If `True`, and if [hierarchical population names](../../tutorials/usage/#working-with-hierarchical-populations) were provided, then it also saves the prediction for every population level.
            log_prob_th: If the log-probability of the most probable population for one cell is below this threshold, this cell will not be annotated (`np.nan`).

        Returns:
            Population predictions (pandas `Series` of length $N$ cells).
        """
        df = self.predict_proba()

        populations = df.iloc[:, : self.n_pops].idxmax(axis=1).astype("category")
        populations[df["max_log_prob"] < log_prob_th] = np.nan

        if key_added is not None:
            self.adata.obs[key_added] = pd.Categorical(
                populations, categories=self.pop_names
            )
            if add_levels and isinstance(self.table.index, pd.MultiIndex):
                utils._add_level_predictions(self, key_added)

        missing_pops = self.n_pops - len(populations.cat.categories)
        if missing_pops:
            log.warning(
                f"{missing_pops} population(s) were not predicted. It may be due to:\n  - Errors in the knowledge table (see https://mics-lab.github.io/scyan/advice/#advice-for-the-creation-of-the-table)\n  - The model hyperparameters choice (see https://mics-lab.github.io/scyan/advanced/parameters/)\n  - Or maybe these populations are really absent from this dataset."
            )

        return populations

    def predict_proba(self) -> pd.DataFrame:
        """Soft predictions (i.e. an array of probability per population) for each cell.

        Returns:
            Dataframe of shape `(N, P)` with probabilities for each population.
        """
        log_probs = self.prior.log_prob(self.x) - torch.log(torch.tensor(self.n_pops))
        probs = torch.softmax(log_probs, dim=1)

        df = pd.DataFrame(probs.numpy(force=True), columns=self.pop_names)

        max_log_probs = log_probs.max(1)
        df["max_log_prob"] = max_log_probs.values.numpy(force=True)

        return df
