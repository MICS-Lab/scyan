from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch import Tensor


class AdataDataset(torch.utils.data.Dataset):
    """Pyorch Dataset"""

    def __init__(self, x: Tensor, covariates: Tensor):
        super().__init__()
        self.x = x
        self.covariates = covariates

    def __getitem__(self, index):
        return self.x[index], self.covariates[index]

    def __len__(self):
        return len(self.x)


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
    categorical_covariate_keys: List[str],
    continuous_covariate_keys: List[str],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Initialize the data and the covariates"""
    x = torch.tensor(adata[:, markers].X)

    if (batch_key is not None) and batch_key not in categorical_covariate_keys:
        categorical_covariate_keys.append(batch_key)

    for key in categorical_covariate_keys:  # enforce dtype category
        adata.obs[key] = adata.obs[key].astype("category")

    categorical_covariate_embedding = (
        pd.get_dummies(adata.obs[categorical_covariate_keys]).values
        if categorical_covariate_keys
        else np.empty((adata.n_obs, 0))
    )

    continuous_covariate_embedding = (
        adata.obs[continuous_covariate_keys].values
        if continuous_covariate_keys
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
