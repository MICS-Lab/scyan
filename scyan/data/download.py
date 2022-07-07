import logging
import urllib
from pathlib import Path
from typing import Union
from urllib import request

import pandas as pd
import scanpy as sc
from anndata import AnnData

from ..utils import _root_path

log = logging.getLogger(__name__)


def get_local_file(data_path: Path, dataset: str, name: str, is_table: bool):
    filename = f"{name}.{'csv' if is_table else 'h5ad'}"
    filepath = data_path / filename

    if not filepath.is_file():
        base_url = "https://gitlab-research.centralesupelec.fr/mics_biomathematics/biomaths/scyan_data"
        url = f"{base_url}/-/raw/main/data/{dataset}/{filename}?inline=false"

        try:
            log.info(
                f"Downloading {filename} from dataset {dataset}. It can take dozens of seconds."
            )
            request.urlretrieve(url, filepath)
            log.info(f"Successfully downloaded and saved locally at {filepath}")
        except urllib.error.HTTPError as e:
            if e.code == "404":
                raise FileNotFoundError(
                    f"File data/{dataset}/{filename} not existing on the repository {base_url}"
                )
            raise e

    if is_table:
        return pd.read_csv(filepath, index_col=0)
    return sc.read_h5ad(filepath)


def load(
    dataset: str, size: str = "default", table: str = "default"
) -> Union[AnnData, pd.DataFrame]:
    """Loads a dataset, i.e. its `AnnData` object and its knowledge table.

    Args:
        dataset: Name of the dataset. Datasets available are: `"aml"`, `"bmmc"`, `"debarcoding"`.
        size: Size of the `anndata` object that should be loaded. By default only one size is available, but you can add some.
        table: Name of the knowledge table that should be loaded. By default only one table is available, but you can add some.

    Returns:
        `AnnData` instance and the marker-population matrix
    """
    data_path = _root_path() / "data" / dataset

    assert (
        data_path.is_dir()
    ), f"{data_path} is not an existing directory. Valid dataset values are 'aml', 'bmmc', 'debarcoding'"

    marker_pop_matrix = get_local_file(data_path, dataset, table, True)
    adata = get_local_file(data_path, dataset, size, False)

    return adata, marker_pop_matrix
