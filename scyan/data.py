import torch
from torch import Tensor
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import List, Tuple, Union, Sized
import numpy as np

from .utils import root_path


class AdataDataset(torch.utils.data.Dataset):
    def __init__(self, x: Tensor, covariates: Tensor, batch: Tensor):
        """PyTorch dataset

        Args:
            x (Tensor): Inputs
            covariates (Tensor): Covariates
        """
        super().__init__()
        self.x = x
        self.covariates = covariates
        self.batch = batch

    def __getitem__(self, index):
        return self.x[index], self.covariates[index], self.batch[index]

    def __len__(self):
        return len(self.x)


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: Sized, max_samples: Union[int, None]):
        self.data_source = data_source
        self.n_samples = len(data_source)
        self.max_samples = self.n_samples if max_samples is None else max_samples

    def __iter__(self):
        return iter(torch.randperm(self.n_samples)[: self.max_samples])

    def __len__(self):
        return self.max_samples


def load(dataset: str, size: str = "default") -> Union[AnnData, pd.DataFrame]:
    """Loads a dataset

    Args:
        dataset (str): Name of the dataset. Either "AML" or "BMMC".

    Returns:
        Union[AnnData, pd.DataFrame]: AnnData instance and the marker-population matrix
    """
    data_path = root_path() / "data" / dataset

    adata = sc.read_h5ad(data_path / f"{size}.h5ad")
    marker_pop_matrix = pd.read_csv(data_path / "table.csv", index_col=0)

    return adata, marker_pop_matrix


def _prepare_data(
    adata: AnnData,
    markers: List[str],
    batch_key: Union[str, int, None],
    categorical_covariate_keys: List[str],
    continuous_covariate_keys: List[str],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Initializes the data and the covariates"""
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

    batch = (
        torch.tensor(adata.obs[batch_key].astype(int).values)
        if batch_key
        else torch.empty(adata.n_obs)
    )

    return x, covariates, batch
