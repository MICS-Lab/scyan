import torch
from torch import Tensor
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Union, Sized

from .utils import root_path


class AdataDataset(torch.utils.data.Dataset):
    def __init__(self, x: Tensor, covariates: Tensor):
        """PyTorch dataset

        Args:
            x (Tensor): Inputs
            covariates (Tensor): Covariates
        """
        super().__init__()
        self.x = x
        self.covariates = covariates

    def __getitem__(self, index):
        return self.x[index], self.covariates[index]

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
