import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import tqdm
import umap
from anndata import AnnData

from ..utils import _root_path

log = logging.getLogger(__name__)


def _download(url, filepath):
    import requests

    r = requests.get(url, stream=True)

    if r.status_code == 404:
        raise FileNotFoundError(f"File at url {url} was not found")

    file_size = int(r.headers["Content-Length"])
    chunk_size = 1024
    num_bars = file_size // chunk_size

    with open(filepath, "wb") as fp:
        for chunk in tqdm.tqdm(
            r.iter_content(chunk_size=chunk_size),
            total=num_bars,
            unit="KB",
            desc=f"Downloading {num_bars // 1024} MB",
        ):
            fp.write(chunk)


def get_local_file(
    dataset_path: Path, dataset_name: str, name: str, ext: str
) -> Union[AnnData, pd.DataFrame, umap.UMAP]:
    """Get an `anndata` or a `csv` file into memory. If the file does not exist locally, it is downloaded from Gitlab.

    Args:
        dataset_path: Local path to the dataset folder.
        dataset_name: Name of the dataset.
        name: Name of the file (without extension).
        ext: `csv`, `h5ad` or `umap`.

    Raises:
        FileNotFoundError: If the file does not exist on Gitlab.
        e: Other error from the Gitlab download.

    Returns:
        An `anndata` or a `csv` object.
    """
    filename = f"{name}.{ext}"
    filepath = dataset_path / filename

    if not filepath.is_file():
        assert dataset_name in [
            "aml",
            "bmmc",
            "debarcoding",
            "poised",
        ], f"File {filepath} not existing."

        base_url = "https://github.com/MICS-Lab/scyan_data"
        url = f"{base_url}/raw/main/data/{dataset_name}/{filename}"

        log.info(
            f"File not found locally. Trying to load {filename} from dataset {dataset_name} on github."
        )
        try:
            _download(url, filepath)
        except KeyboardInterrupt as e:
            filepath.unlink()
            raise e
        log.info(f"Successfully downloaded and saved locally at {filepath}")

    if ext == "csv":
        df = pd.read_csv(filepath, index_col=0)
        level_indices = np.where(df.columns.str.lower().str.contains("level"))[0]
        if level_indices.size:
            return pd.read_csv(filepath, index_col=[0] + list(1 + level_indices))
        return df

    if ext == "h5ad":
        return anndata.read_h5ad(filepath)

    import joblib

    return joblib.load(filepath)


def get_data_path() -> Path:
    """Get the path to the folder where datasets are saved. Either at the root of the repository, or at `<your-home>/.scyan_data`"""
    data_path = _root_path() / "data"

    if not data_path.is_dir():
        return Path.home() / ".scyan_data"

    return data_path


def _check_can_write(path: Path, overwrite: bool) -> None:
    assert (
        overwrite or not path.exists()
    ), f"File {path} already exists and 'overwrite' is False. You can either change the 'filename' argument, or set 'overwrite=True' to force its creation."


def add(
    dataset_name: str,
    *objects: List[Union[AnnData, pd.DataFrame, umap.UMAP]],
    filename: Union[str, List[str]] = "default",
    overwrite: bool = False,
) -> None:
    """Add an object to a dataset (or create it if not existing). Objects can be `AnnData` objects, a knowledge-table (i.e. a `pd.DataFrame`), or a `UMAP` reducer. The provided filenames are the one you use when loading data with [scyan.data.load][].

    !!! note
        You will be able to load this dataset with [scyan.data.load][] as long as you added at least a knowledge-table and a `adata` object.

    Args:
        dataset_name: Name of the dataset in which the object will be saved.
        *objects: Object(s) to save.
        filename: Name(s) without extension of the file(s) to create. The default value (`"default"`) is the filename loaded by default with `scyan.data.load`. If a list is provided, it should have the length of the number of objects provided and should have the same order. If a string, then use the same name for all objects (but different extensions).
        overwrite: If `True`, it will overwrite files that already exist.

    Raises:
        ValueError: If the object type is not one of the three required.
    """
    data_path = get_data_path()

    dataset_path = data_path / dataset_name
    if not dataset_path.is_dir():
        log.info(f"Creating new dataset folder at {dataset_path}")
        dataset_path.mkdir(parents=True)

    filenames = [filename] * len(objects) if isinstance(filename, str) else filename

    for obj, filename in zip(objects, filenames):
        if isinstance(obj, AnnData):
            path = dataset_path / f"{filename}.h5ad"
            _check_can_write(path, overwrite)
            obj.write_h5ad(path)
        elif isinstance(obj, pd.DataFrame):
            path = dataset_path / f"{filename}.csv"
            _check_can_write(path, overwrite)
            obj.to_csv(path)
        elif isinstance(obj, umap.UMAP):
            path = dataset_path / f"{filename}.umap"
            _check_can_write(path, overwrite)
            import joblib

            joblib.dump(obj, path)
        else:
            raise ValueError(
                f"Can't save object of type {type(obj)}. It must be an AnnData object, a DataFrame or a UMAP."
            )

        log.info(f"Created file {path}")


def add_remote_names(filenames: dict, *names: str) -> None:
    for name in names:
        stem, suffix = name.split(".")
        filenames[f".{suffix}"].add(stem)


def list_path(dataset_path: Path) -> None:
    dataset_name = dataset_path.stem
    print(f"\nDataset name: {dataset_name}")

    names = {".h5ad": set(), ".csv": set(), ".umap": set()}

    for file_path in dataset_path.iterdir():
        if file_path.suffix in names.keys():
            names[file_path.suffix].add(file_path.stem)

    if dataset_name == "aml":
        add_remote_names(
            names,
            "default.csv",
            "default.h5ad",
            "short.h5ad",
        )
    elif dataset_name == "bmmc":
        add_remote_names(names, "default.csv", "default.h5ad")
    elif dataset_name == "poised":
        add_remote_names(names, "default.csv", "full.csv", "short.csv", "default.h5ad")
    elif dataset_name == "debarcoding":
        add_remote_names(
            names,
            "default.csv",
            "default.h5ad",
        )

    for kind, values in zip(["Data versions", "Tables", "UMAP reducers"], names.values()):
        if values:
            print(f"""    {kind}: '{"', '".join(list(values))}'""")


def _list(dataset_name: Optional[str] = None) -> None:
    """Show existing datasets and their different versions/table names.

    Args:
        dataset_name: Optional dataset name. If provided, only display the version names of the provided `dataset_name`, otherwise list all existing datasets.
    """
    data_path = get_data_path()

    if dataset_name is not None:
        dataset_path = data_path / dataset_name
        log.info(f"Listing versions inside {dataset_path}:")
        list_path(dataset_path)
        return

    log.info(f"List of existing datasets inside {data_path}:")
    dataset_paths = set(data_path.iterdir())

    for public_dataset in ["poised", "aml", "bmmc", "debarcoding"]:
        dataset_path = Path(data_path / public_dataset)
        dataset_path.mkdir(parents=True, exist_ok=True)
        dataset_paths.add(dataset_path)

    for dataset_path in dataset_paths:
        if dataset_path.is_dir():
            list_path(dataset_path)


def remove(
    dataset_name: str,
    version: Optional[str] = None,
    table: Optional[str] = None,
    reducer: Optional[str] = None,
) -> None:
    """Remove file(s) from a dataset folder.

    Args:
        dataset_name: Name of the dataset. Use `scyan.data.list()` to see the possible values.
        version: Name of the `.h5ad` file to remove (don't provide the extension).
        table: Name of the `.csv` file to remove (don't provide the extension).
        reducer: Name of the `.umap` file to remove (don't provide the extension).
    """
    data_path = get_data_path()
    dataset_path = data_path / dataset_name

    for arg, ext in [(version, "h5ad"), (table, "csv"), (reducer, "umap")]:
        if arg is not None:
            filepath = dataset_path / f"{arg}.{ext}"
            filepath.unlink()
            log.info(f"Successfully removed {filepath}")


def load(
    dataset_name: str,
    version: Optional[str] = "default",
    table: Optional[str] = "default",
    reducer: Optional[str] = None,
) -> Tuple[AnnData, pd.DataFrame]:
    """Load a dataset, i.e. its `AnnData` object and its knowledge table. Public datasets available are `"poised"`, `"aml"`, `"bmmc"`, and `"debarcoding"`; note that, if the dataset was not loaded yet, it is automatically downloaded (requires internet connection). Existing dataset names and versions/tables can be listed using [scyan.data.list][].

    !!! note
        If you want to load your own dataset, you first have to [create it](../../advanced/data).
    !!! note
        The data is saved by default inside `<home_path>/.scyan_data`. Optionally, if `scyan` repository was cloned, you can create `<scyan_repository_path>/data` and use it instead of the default data folder.

    Args:
        dataset_name: Name of the dataset. Either one of your dataset, or one public dataset among `"poised"`, `"aml"`, `"bmmc"`, and `"debarcoding"`.
        version: Name of the `anndata` file (.h5ad) that should be loaded. The available versions can be listed with `scyan.data.list()`. If `None`, don't return an `adata` object.
        table: Name of the knowledge table that should be loaded. If `None`, don't return the `table` dataframe.
        reducer: Name of the umap reducer that should be loaded. If `None`, don't return the `UMAP` reducer.

    Returns:
        Tuple containing the requested data, i.e. by default a tuple `(adata, table)` is returned (the adata instance and the knowledge table). But, for instance, if `version is None` and `reducer` is provided, then it returns a tuple `(table, reducer)`.
    """
    assert any(
        arg is not None for arg in [version, table, reducer]
    ), "Provide at least one argument that is not `None` among 'version', 'table', and 'reducer'."

    data_path = get_data_path()

    dataset_path = data_path / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    return tuple(
        get_local_file(dataset_path, dataset_name, arg, ext)
        for arg, ext in [(version, "h5ad"), (table, "csv"), (reducer, "umap")]
        if arg is not None
    )
