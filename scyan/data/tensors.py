from typing import List, Optional, Sized, Tuple, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch import Tensor


class AdataDataset(torch.utils.data.Dataset):
    """Pyorch Dataset"""

    def __init__(self, x: Tensor, covariates: Tensor, batch: Tensor):
        super().__init__()
        self.x = x
        self.covariates = covariates
        self.batch = batch

    def __getitem__(self, index):
        return self.x[index], self.covariates[index], self.batch[index]

    def __len__(self):
        return len(self.x)


class RandomSampler(torch.utils.data.Sampler):
    """Random sampling during training. It stops the epoch when we reached `max_samples` samples (if provided)."""

    def __init__(self, data_source: Sized, max_samples: Optional[int]):
        self.data_source = data_source
        self.n_samples = len(data_source)
        self.max_samples = self.n_samples if max_samples is None else max_samples

    def __iter__(self):
        return iter(torch.randperm(self.n_samples)[: self.max_samples])

    def __len__(self):
        return self.max_samples


def _prepare_data(
    adata: AnnData,
    markers: List[str],
    batch_key: Union[str, int, None],
    batch_ref: Union[str, int, None],
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

    if batch_key is not None:
        batch_to_id = {b: i for i, b in enumerate(adata.obs[batch_key].cat.categories)}
        batches = torch.tensor([batch_to_id[b] for b in adata.obs[batch_key]])
        other_batches = [
            batch_to_id[b] for b in adata.obs[batch_key].cat.categories if b != batch_ref
        ]

        return x, covariates, batches, other_batches, batch_to_id

    return x, covariates, torch.empty((adata.n_obs,)), [], {}
