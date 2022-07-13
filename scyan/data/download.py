import logging
import urllib
from pathlib import Path
from typing import Tuple, Union
from urllib import request

import pandas as pd
import scanpy as sc
from anndata import AnnData

from ..utils import _root_path

log = logging.getLogger(__name__)


def get_local_file(
    dataset_path: Path, dataset: str, name: str, is_table: bool
) -> Union[AnnData, pd.DataFrame]:
    """Get an `anndata` or a `csv` file into memory. If the file does not exist locally, it is downloaded from Gitlab.

    Args:
        dataset_path: Local path to the dataset folder.
        dataset: Name of the dataset.
        name: Name of the file (without extension).
        is_table: Whether a `csv` or an `anndata` has to be loaded.

    Raises:
        FileNotFoundError: If the file does not exist on Gitlab.
        e: Other error from the Gitlab download.

    Returns:
        An `anndata` or a `csv` object.
    """
    filename = f"{name}.{'csv' if is_table else 'h5ad'}"
    filepath = dataset_path / filename

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
) -> Tuple[AnnData, pd.DataFrame]:
    """Load a dataset, i.e. its `AnnData` object and its knowledge table.
    If the dataset was not loaded yet, it is automatically downloaded (requires internet connection).
    !!! note
        If `scyan` repository was cloned, then the data will be saved in the `data` folder of the repository, else at `<home_path>/.scyan_data`
    !!! note
        You can add other datasets inside the data folder (see note above) and load them with this function.
        Just make sure you create a specific folder for your dataset, and save your `anndata` object and your table in `h5ad` and `csv` formats respectively.

    Args:
        dataset: Name of the dataset. Datasets available are: `"aml"`, `"bmmc"`, `"debarcoding"`.
        size: Size of the `anndata` object that should be loaded. By default only one size is available, but you can add some.
        table: Name of the knowledge table that should be loaded. By default only one table is available, but you can add some.

    Returns:
        `AnnData` instance and the knowledge table.
    """
    data_path = _root_path() / "data"

    if not data_path.is_dir():
        # Repository was not clone, or not installed in editable mode
        data_path = Path.home() / ".scyan_data"

    dataset_path = data_path / dataset
    dataset_path.mkdir(parents=True, exist_ok=True)

    adata = get_local_file(dataset_path, dataset, size, False)
    marker_pop_matrix = get_local_file(dataset_path, dataset, table, True)

    return adata, marker_pop_matrix
