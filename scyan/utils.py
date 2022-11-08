import logging
import warnings
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import flowio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
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


def _to_df(adata: AnnData, layer: Optional[str] = None) -> pd.DataFrame:
    df = pd.concat([adata.to_df(layer), adata.obs], axis=1)

    for key in adata.obsm:
        names = [f"{key}{i+1}" for i in range(adata.obsm[key].shape[1])]
        df[names] = adata.obsm[key]

    return df


def write_fcs(
    adata: AnnData,
    path: str,
    layer: Optional[str] = None,
    columns_to_numeric: Optional[List] = None,
) -> Union[None, Dict]:
    """Based on a `AnnData` object, it writes a FCS file that contains (i) all the markers intensities, (ii) every numeric column of `adata.obs`, and (iii) all `adata.obsm` variables.

    !!! note
        As the FCS format doesn't support strings, some observations will not be kept in the FCS file.

    Args:
        adata: `AnnData` object to save.
        path: Path to write the file.
        layer: Name of the `adata` layer from which intensities will be extracted. If `None`, uses `adata.X`.
        columns_to_numeric: List of **non-numerical** column names from `adata.obs` that should be kept, by transforming them into integers. Note that you don't need to list the numerical columns, that are written inside the FCS by default.

    Returns:
        If `columns_to_numeric` is `None`, returns nothing. Else, return a dict whose keys are the observation column names being transformed, and the values are ordered lists of the label encoded classes. E.g., `{"batch": ["b1", "b2"]}` means that the batch `"b1"` was encoded by 0, and `"b2"` by 1.
    """
    df = _to_df(adata, layer)
    dict_classes = {}
    columns_removed = []

    for column in df.columns:
        if is_numeric_dtype(df[column].dtype):
            continue
        try:
            df[column] = pd.to_numeric(df[column].values)
            continue
        except:
            if columns_to_numeric is not None and column in columns_to_numeric:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].values)
                dict_classes[column] = list(le.classes_)
            else:
                del df[column]
                columns_removed.append(column)

    log.info(f"Found {len(df.columns)} features: {', '.join(df.columns)}.")
    if columns_removed:
        log.info(
            f"FCS does not support strings, so the following columns where removed: {', '.join(columns_removed)}.\nIf you want to keep these str observations, use the 'columns_to_numeric' argument to encod them."
        )

    with open(path, "wb") as f:
        flowio.create_fcs(f, df.values.flatten(), df.columns)

    if columns_to_numeric is not None:
        return dict_classes


def write_csv(
    adata: AnnData,
    path: str,
    layer: Optional[str] = None,
) -> Union[None, Dict]:
    """Based on a `AnnData` object, it writes a CSV file that contains (i) all the markers intensities, (ii) every numeric column of `adata.obs`, and (iii) all `adata.obsm` variables.

    Args:
        adata: `AnnData` object to save.
        path: Path to write the file.
        layer: Name of the `adata` layer from which intensities will be extracted. If `None`, uses `adata.X`.
    """
    _to_df(adata, layer).to_csv(path)


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
        ), "The model has to be trained first, consider running 'model.fit()'"
        return f(model, *args, **kwargs)

    return wrapper


def _add_level_predictions(model, obs_key: str) -> None:
    mpm: pd.DataFrame = model.marker_pop_matrix
    adata: AnnData = model.adata

    level_names = mpm.index.names[1:]
    obs_keys = [f"{obs_key}_{name}" for name in level_names]

    pop_dict = {pop: levels_pops for pop, *levels_pops in mpm.index}
    preds = np.vstack(adata.obs[obs_key].astype(str).apply(pop_dict.get))
    adata.obs[obs_keys] = pd.DataFrame(preds, dtype="category", index=adata.obs.index)


def _get_pop_index(pop: str, marker_pop_matrix: pd.DataFrame):
    for i in range(marker_pop_matrix.index.nlevels - 1, -1, -1):
        if pop in marker_pop_matrix.index.get_level_values(i):
            return i
    raise BaseException(f"Population {pop} not found.")


def _check_is_processed(X: np.ndarray) -> None:
    assert (
        np.abs(X).max() < 1e3
    ), "The provided values are very high: have you run preprocessing first? E.g., consider running 'scyan.tools.asinh_transform' or 'scyan.tools.auto_logicle_transform' (see our tutorial: https://mics-lab.github.io/scyan/tutorials/preprocessing/)"


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

    if np.abs(X.mean(axis=0)).max() > 0.2 or np.abs(X.std(axis=0) - 1).max() > 0.2:
        log.warn(
            "It seems that the data is not standardised. We advise using scaling (scyan.tools.scale) before initializing the model."
        )

    duplicates = df.duplicated()
    if duplicates.any():
        duplicates_names = duplicates[duplicates].index.get_level_values(0)
        log.warn(
            f"Found duplicate populations in the knowledge matrix. We advise updating or removing the following rows: {', '.join(duplicates_names)}"
        )

    ratio_non_standard = 1 - ((df**2 == 1) | df.isna()).values.mean()
    if ratio_non_standard > 0.15:
        log.warn(
            f"Found a significant proportion ({ratio_non_standard:.1%}) of non-standard values in the knowledge table. Scyan expects to find mostly -1/1/NA in the table, even though any other numerical value is accepted. If this is intended, just ignore the warning, else correct the table using mainly -1, 1 and NA (to denote negative expressions, positive expressions, or not-applicable respectively)."
        )

    return adata, df
