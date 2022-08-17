import logging
import warnings
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import flowio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from pandas.api.types import is_numeric_dtype
from torch import Tensor

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Transforming to str index")


def _root_path() -> Path:
    """Get the library root path

    Returns:
        `scyan` library root path
    """
    return Path(__file__).parent.parent


def _wandb_plt_image(fun: Callable, figsize: Tuple[int, int] = [7, 5]):
    """Transform a matplotlib figure into a wandb Image

    Args:
        fun: Function that makes the plot - do not plt.show().
        figsize: Matplotlib figure size.

    Returns:
        The wandb Image to be logged.
    """

    import io

    import wandb
    from PIL import Image

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    fun()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    return wandb.Image(Image.open(img_buf))


def read_fcs(
    path: str, names_selection: Optional[List[str]] = None, log_names: bool = True
) -> AnnData:
    """Read a FCS file and return an AnnData instance

    Args:
        path: Path to the FCS file that has to be read.
        names_selection: If `None`, automatically detect if a channel has to be loaded in `obs` (e.g. Time) or if it is a marker (e.g. CD4). Else, you can provide a list of channels to select as variables.
        log_names: If `True` and if `names_selection` is not `None`, then it logs all the names detected from the FCS file. It can be useful to set `names_selection` properly.

    Returns:
        `AnnData` instance containing the FCS data.
    """
    fcs_data = flowio.FlowData(str(path))
    data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))

    names = np.array(
        [value.get("PnS", value["PnN"]) for value in fcs_data.channels.values()]
    )

    if names_selection is None:
        is_marker = np.array(["PnS" in value for value in fcs_data.channels.values()])
    else:
        if log_names:
            log.info(
                f"Found {len(names)} names: {', '.join(names)}. Set log_names=False to disable this log."
            )

        is_marker = np.array([name in names_selection for name in names])

    X = data[:, is_marker]
    var = pd.DataFrame(index=names[is_marker])
    obs = pd.DataFrame(
        data=data[:, ~is_marker],
        columns=names[~is_marker],
        index=range(data.shape[0]),
    )

    return AnnData(X=X, var=var, obs=obs)


def write_fcs(adata: AnnData, path: str) -> None:
    """Write a FCS file based on a `AnnData` object.

    Args:
        adata: `AnnData` object to save.
        path: Path to write the file.
    """
    X = adata.X
    channel_names = list(adata.var_names)

    for column in adata.obs.columns:
        if is_numeric_dtype(adata.obs[column].dtype):
            X = np.c_[X, adata.obs[column].values]
            channel_names.append(column)

    for key in adata.obsm:
        X = np.concatenate((X, adata.obsm[key]), axis=1)
        channel_names += [f"{key}{i+1}" for i in range(adata.obsm[key].shape[1])]

    log.info(f"Found {len(channel_names)} channels: {', '.join(channel_names)}")

    with open(path, "wb") as f:
        flowio.create_fcs(X.flatten(), channel_names, f)


def _subset(indices: List[str], max_obs: int):
    if len(indices) < max_obs:
        return indices
    return indices[np.random.choice(len(indices), max_obs, replace=False)]


def _markers_to_indices(model, markers: List[str]) -> Tensor:
    return torch.tensor(
        [model.var_names.get_loc(marker) for marker in markers],
        dtype=int,
    )


def _pop_to_index(model, pop: str):
    assert pop in model.pop_names, f"Found invalid population name '{pop}'"
    return model.pop_names.get_loc(pop)


def _pops_to_indices(model, pops: List[str]) -> Tensor:
    return torch.tensor([_pop_to_index(model, pop) for pop in pops], dtype=int)


def _get_subset_indices(adata, n_cells: Union[int, None]):
    if n_cells is None or n_cells >= adata.n_obs:
        return np.arange(adata.n_obs)
    return np.random.choice(adata.n_obs, size=n_cells, replace=False)


def _process_pop_sample(model, pop: Union[str, List[str], int, Tensor, None] = None):
    if isinstance(pop, str):
        return _pop_to_index(model, pop)
    if isinstance(pop, list):
        return _pops_to_indices(model, pop)
    else:
        return pop


def _requires_fit(f: Callable) -> Callable:
    """Make sure the model has been trained"""

    @wraps(f)
    def wrapper(model, *args, **kwargs):
        assert (
            model._is_fitted
        ), "The model have to be trained first, consider running 'model.fit()'"
        return f(model, *args, **kwargs)

    return wrapper


def _validate_inputs(adata: AnnData, df: pd.DataFrame):
    assert isinstance(
        adata, AnnData
    ), f"The provided adata has to be an AnnData object (https://anndata.readthedocs.io/en/latest/), found {type(adata)}."

    assert isinstance(
        df, pd.DataFrame
    ), f"The marker-population matrix has to be a pandas DataFrame, found {type(df)}"

    not_found_columns = [c for c in df.columns if c not in adata.var_names]

    assert (
        not not_found_columns
    ), f"All column names from the marker-population table have to be a known marker from adata.var_names. Missing {not_found_columns}."

    if not df.dtypes.apply(is_numeric_dtype).all():
        log.warn(
            "Some columns of the marker-population table are not numeric / NaN. Every non-numeric value will be transformed into NaN."
        )
        df = df.apply(pd.to_numeric, errors="coerce")

    X = adata[:, df.columns].X

    assert (
        np.abs(X).max() < 1e3
    ), "The provided values are very high: have you run preprocessing first? E.g., consider 'scyan.tools.asinh_transform' or 'scyan.tools.auto_logicle_transform'"

    if np.abs(X.mean(axis=0)).max() > 0.2 or np.abs(X.std(axis=0) - 1).max() > 0.2:
        log.warn(
            "It seems that the data is not standardised. We advise using scaling (scyan.tools.scale) before initializing the model."
        )

    duplicates = df.duplicated()
    if duplicates.any():
        log.warn(
            f"Found duplicate populations in the knowledge matrix. We advise updating or removing the following rows: {', '.join(duplicates[duplicates].index)}"
        )

    return adata, df
