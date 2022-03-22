import torch
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Union

from .utils import root_path


class AdataDataset(torch.utils.data.Dataset):
    def __init__(self, x, covariates):
        super().__init__()
        self.x = x
        self.covariates = covariates

    def __getitem__(self, index):
        return self.x[index], self.covariates[index]

    def __len__(self):
        return len(self.x)


def load(dataset: str) -> Union[AnnData, pd.DataFrame]:
    data_path = root_path() / "data"

    adata = sc.read_h5ad(data_path / f"{dataset}.h5ad")
    marker_pop_matrix = pd.read_csv(data_path / f"{dataset}.csv", index_col=0)

    return adata, marker_pop_matrix
