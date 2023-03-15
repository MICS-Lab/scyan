from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch import Tensor


class RandomSampler(torch.utils.data.Sampler):
    """Random sampling during training. It stops the epoch when we reached `max_samples` samples (if provided)."""

    def __init__(self, n_obs: int, n_samples: int):
        """
        Args:
            n_obs: Total number of cells.
            n_samples: Number of samples per epoch.
            batch_size: Mini-batch size.
            batches: Tensor of batch assignment for every cell.
            corr_mode: Whether batch correction is enabled or not.
        """
        self.n_obs = n_obs
        self.n_samples = n_samples

    def __iter__(self):
        yield from torch.randperm(self.n_obs)[: self.n_samples]

    def __len__(self):
        return self.n_samples


def _prepare_data(
    adata: AnnData,
    markers: List[str],
    batch_key: Union[str, int, None],
    categorical_covariates: List[str],
    continuous_covariates: List[str],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Initialize the data and the covariates"""
    x = torch.tensor(adata[:, markers].X, dtype=torch.float32)

    if (batch_key is not None) and batch_key not in categorical_covariates:
        categorical_covariates.append(batch_key)

    for key in list(categorical_covariates) + list(continuous_covariates):
        assert key in adata.obs, f"Covariate {key} in not an existing column of adata.obs"

    for key in categorical_covariates:  # enforce dtype category
        adata.obs[key] = adata.obs[key].astype("category")

    categorical_covariate_embedding = (
        pd.get_dummies(adata.obs[categorical_covariates]).values
        if categorical_covariates
        else np.empty((adata.n_obs, 0))
    )

    continuous_covariate_embedding = (
        adata.obs[continuous_covariates].values
        if continuous_covariates
        else np.empty((adata.n_obs, 0))
    )

    covariates = np.concatenate(
        [
            categorical_covariate_embedding,
            continuous_covariate_embedding,
        ],
        axis=1,
    )
    covariates = torch.tensor(
        covariates,
        dtype=torch.float32,
    )

    return x, covariates
