import logging
import urllib
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib import request

import anndata
import joblib
import pandas as pd
import umap
from anndata import AnnData

from ..utils import _root_path

log = logging.getLogger(__name__)


def get_local_file(
    dataset_path: Path, dataset_name: str, name: str, kind: str
) -> Union[AnnData, pd.DataFrame, umap.UMAP]:
    """Get an `anndata` or a `csv` file into memory. If the file does not exist locally, it is downloaded from Gitlab.

    Args:
        dataset_path: Local path to the dataset folder.
        dataset_name: Name of the dataset.
        name: Name of the file (without extension).
        kind: `csv`, `h5ad` or `umap`.

    Raises:
        FileNotFoundError: If the file does not exist on Gitlab.
        e: Other error from the Gitlab download.

    Returns:
        An `anndata` or a `csv` object.
    """
    filename = f"{name}.{kind}"
    filepath = dataset_path / filename

    if not filepath.is_file():
        base_url = "https://github.com/MICS-Lab/scyan_data"
        url = f"{base_url}/raw/main/data/{dataset_name}/{filename}"

        try:
            log.info(
                f"Downloading {filename} from dataset {dataset_name}. It can take dozens of seconds."
            )
            request.urlretrieve(url, filepath)
            log.info(f"Successfully downloaded and saved locally at {filepath}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(
                    f"File data/{dataset_name}/{filename} not existing on the repository {base_url}"
                )
            raise e

    if kind == "csv":
        df = pd.read_csv(filepath, index_col=0)
        if df.columns[0] == "Group name":
            return pd.read_csv(filepath, index_col=[0, 1])
        return df

    if kind == "h5ad":
        return anndata.read_h5ad(filepath)

    return joblib.load(filepath)


def get_data_path() -> Path:
    """Get the path to the folder where datasets are saved. Either at the root of the repository, or at `<your-home>/.scyan_data`"""
    data_path = _root_path() / "data"

    if not data_path.is_dir():
        return Path.home() / ".scyan_data"

    return data_path


def add(
    dataset_name: str,
    *objects: List[Union[AnnData, pd.DataFrame, umap.UMAP]],
    filenames: Union[str, List[str]] = "default",
) -> None:
    """Add an object to a dataset (or create it if not existing). Objects can be `AnnData` objects, a knowledge-table (i.e. a `pd.DataFrame`), or a `UMAP` reducer.

    !!! note
        You will be able to load this dataset with [scyan.data.load](./load.md) as long as you added at least a knowledge-table and a `adata` object.

    Args:
        dataset_name: Name of the dataset in which the object will be saved.
        *objects: Object(s) to save.
        filenames: Name of the file(s) created. The default value (`"default"`) is the filename loaded by default with `scyan.data.load`. If a list is provided, it should have the length of the number of objects provided.

    Raises:
        ValueError: If the object type is not one of the three required.
    """
    data_path = get_data_path()

    dataset_path = data_path / dataset_name
    if not dataset_path.is_dir():
        log.info(f"Creating new dataset folder at {dataset_path}")
        dataset_path.mkdir(parents=True)

    if isinstance(filenames, str):
        filenames = [filenames] * len(objects)

    for obj, filename in zip(objects, filenames):
        if isinstance(obj, AnnData):
            path = dataset_path / f"{filename}.h5ad"
            obj.write_h5ad(path)
        elif isinstance(obj, pd.DataFrame):
            path = dataset_path / f"{filename}.csv"
            obj.to_csv(path)
        elif isinstance(obj, umap.UMAP):
            path = dataset_path / f"{filename}.umap"
            joblib.dump(obj, path)
        else:
            raise ValueError(
                f"Can't save object of type {type(obj)}. It must be an AnnData object, a DataFrame or a UMAP."
            )

        log.info(f"Created file {path}")


def load(
    dataset_name: str,
    size: str = "default",
    table: str = "default",
    reducer: Optional[str] = None,
) -> Tuple[AnnData, pd.DataFrame]:
    """Load a dataset, i.e. its `AnnData` object and its knowledge table.
    If the dataset was not loaded yet, it is automatically downloaded (requires internet connection).
    !!! note
        If you want to load your own dataset, you first have to [create it](../../advanced/data).
    !!! note
        If `scyan` repository was cloned, then the data will be saved in the `data` folder of the repository, else at `<home_path>/.scyan_data`

    Args:
        dataset_name: Name of the dataset. Datasets available are: `"aml"`, `"bmmc"`, `"debarcoding"`.
        size: Size of the `anndata` object that should be loaded. By default only one size is available, but you can add some.
        table: Name of the knowledge table that should be loaded. By default only one table is available, but you can add some.
        reducer: Optional: name of the umap reducer that should be loaded.

    Returns:
        `AnnData` instance and the knowledge table. If `reducer` is not None, also return a `UMAP` object.
    """
    data_path = get_data_path()

    dataset_path = data_path / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    adata = get_local_file(dataset_path, dataset_name, size, "h5ad")
    marker_pop_matrix = get_local_file(dataset_path, dataset_name, table, "csv")

    if reducer is not None:
        umap_reducer = get_local_file(dataset_path, dataset_name, reducer, "umap")
        return adata, marker_pop_matrix, umap_reducer

    return adata, marker_pop_matrix
