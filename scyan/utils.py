import logging
import warnings
from functools import wraps
from pathlib import Path
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from pandas.api.types import is_numeric_dtype
from torch import Tensor

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Transforming to str index")
warnings.filterwarnings("ignore", message=r".*Trying to modify attribute[\s\S]*")
warnings.filterwarnings("ignore", message=r".*No data for colormapping provided[\s\S]*")
warnings.filterwarnings("ignore", message=r".*does not have many workers[\s\S]*")


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


def _has_umap(adata: AnnData) -> np.ndarray:
    """Returns an ndarray telling on which cells the UMAP coordinates have been computed."""
    assert (
        "X_umap" in adata.obsm_keys()
    ), "Before plotting a UMAP, its coordinates need to be computed using 'scyan.tools.umap(...)' (see https://mics-lab.github.io/scyan/api/representation/#scyan.tools.umap)"
    return adata.obsm["X_umap"].sum(1) != 0


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


def _get_subset_indices(n_obs, n_cells: Union[int, None]):
    if n_cells is None or n_cells >= n_obs:
        return np.arange(n_obs)
    return np.random.choice(n_obs, size=n_cells, replace=False)


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
        ), "The model has to be trained first, consider running 'model.fit()'"
        return f(model, *args, **kwargs)

    return wrapper


def _add_level_predictions(model, obs_key: str) -> None:
    mpm: pd.DataFrame = model.table
    adata: AnnData = model.adata

    level_names = mpm.index.names[1:]
    obs_keys = [f"{obs_key}_{name}" for name in level_names]

    for i, new_obs_key in enumerate(obs_keys):
        pop_dict = {pop: levels_pops[i] for pop, *levels_pops in mpm.index}
        adata.obs[new_obs_key] = adata.obs[obs_key].map(pop_dict).astype("category")


def _get_pop_index(pop: str, table: pd.DataFrame):
    for i in range(table.index.nlevels - 1, -1, -1):
        if pop in table.index.get_level_values(i):
            return i
    raise BaseException(f"Population {pop} not found.")


def _check_is_processed(X: np.ndarray) -> None:
    assert (
        np.abs(X).max() < 1e3
    ), "The provided values are very high: have you run preprocessing first? E.g., consider running 'scyan.preprocess.asinh_transform' or 'scyan.preprocess.auto_logicle_transform' (see our tutorial: https://mics-lab.github.io/scyan/tutorials/preprocessing/)"


def _check_batch_arg(adata, batch_key, batch_ref):
    assert (
        batch_key is not None
    ), "Scyan model was trained with no batch_key, thus not correcting batch effect"

    batches = adata.obs[batch_key]

    if batch_ref is None:
        batch_ref = batches.value_counts().index[0]
        log.info(f"No batch_ref was provided, using {batch_ref} as reference.")
        return batch_ref

    possible_batches = set(batches)
    assert (
        batch_ref in possible_batches
    ), f"Batch reference '{batch_ref}' is not an existing batch. Choose one among: {', '.join(list(possible_batches))}."

    return batch_ref


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

    ratio_nan = df.isna().values.mean()
    if ratio_nan > 0.7:
        log.warn(
            f"Found {ratio_nan:.1%} of NA in the table, which is very high. If this is intended, just ignore the warning."
        )

    X = adata[:, df.columns].X

    _check_is_processed(X)

    if np.abs(X.std(axis=0) - 1).max() > 0.2 or X.min(0).max() >= 0:
        log.warn(
            "It seems that the data is not standardised. We advise using scaling (scyan.preprocess.scale) before initializing the model."
        )

    duplicates = df.duplicated()
    if duplicates.any():
        duplicates_names = duplicates[duplicates].index.get_level_values(0)
        log.warn(
            f"Found duplicate populations in the knowledge matrix. We advise updating or removing the following rows: {', '.join(map(str, duplicates_names))}"
        )

    if isinstance(df.index, pd.MultiIndex):
        assert (
            not df.index.to_frame().isna().any().any()
        ), "One or multiple population name(s) are NaN, you have to name them."

    ratio_non_standard = 1 - ((df**2 == 1) | df.isna()).values.mean()
    if ratio_non_standard > 0.15:
        log.warn(
            f"Found a significant proportion ({ratio_non_standard:.1%}) of non-standard values in the knowledge table. Scyan expects to find mostly -1/1/NA in the table, even though any other numerical value is accepted. If this is intended, just ignore the warning, else correct the table using mainly -1, 1 and NA (to denote negative expressions, positive expressions, or not-applicable respectively)."
        )

    return adata, df
