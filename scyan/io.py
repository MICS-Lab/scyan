import logging
from typing import Dict, List, Optional, Union

import fcsparser
import fcswrite
import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.api.types import is_numeric_dtype

log = logging.getLogger(__name__)


def read_fcs(
    path: str, obs_names: Optional[List[str]] = None, channel_suffix: Optional[str] = "S"
) -> AnnData:
    """Read a FCS file and return an AnnData instance.

    Args:
        path: Path to the FCS file that has to be read.
        obs_names: Optional list of channel names that has to be considered as an observation (i.e., inside `adata.obs`). By default, choose it automatically.
        channel_suffix: Suffix for the channel naming convention, i.e. `"S"` for "PnS", or `"N"` for "PnN". If `None`, keep the raw names.

    Returns:
        `AnnData` instance containing the FCS data.
    """
    meta, data = fcsparser.parse(path)

    if channel_suffix is not None:
        data.columns = [
            meta.get(f"$P{i + 1}{channel_suffix}", c) for i, c in enumerate(data.columns)
        ]

    if obs_names is None:
        obs_names = [
            c
            for i, c in enumerate(data.columns)
            if not f"$P{i + 1}{channel_suffix}" in meta
        ]
    elif not all(c in data.columns for c in obs_names):
        log.warn(
            f"The following observations were not found: {', '.join([c for c in obs_names if not c in data.columns])}"
        )
        obs_names = [c for c in obs_names if c in data.columns]

    var_names = [c for c in data.columns if not c in obs_names]

    return AnnData(
        X=data[var_names].values,
        var=pd.DataFrame(index=var_names),
        obs=data[obs_names],
        dtype=np.float32,
    )


def _to_df(adata: AnnData, layer: Optional[str] = None) -> pd.DataFrame:
    df = pd.concat([adata.to_df(layer), adata.obs], axis=1)

    for key in adata.obsm:
        names = [f"{key}{i+1}" for i in range(adata.obsm[key].shape[1])]
        df[names] = np.array(adata.obsm[key])

    return df


def write_fcs(
    adata: AnnData,
    path: str,
    layer: Optional[str] = None,
    columns_to_numeric: Optional[List] = None,
    **fcswrite_kwargs: int,
) -> Union[None, Dict]:
    """Based on a `AnnData` object, it writes a FCS file that contains (i) all the markers intensities, (ii) every numeric column of `adata.obs`, and (iii) all `adata.obsm` variables.

    !!! note
        As the FCS format doesn't support strings, some observations will not be kept in the FCS file.

    Args:
        adata: `AnnData` object to save.
        path: Path to write the file.
        layer: Name of the `adata` layer from which intensities will be extracted. If `None`, uses `adata.X`.
        columns_to_numeric: List of **non-numerical** column names from `adata.obs` that should be kept, by transforming them into integers. Note that you don't need to list the numerical columns, that are written inside the FCS by default.
        **fcswrite_kwargs: Optional kwargs provided to `fcswrite.write_fcs`.

    Returns:
        If `columns_to_numeric` is `None`, returns nothing. Else, return a dict whose keys are the observation column names being transformed, and the values are ordered lists of the label encoded classes. E.g., `{"batch": ["b1", "b2"]}` means that the batch `"b1"` was encoded by 0, and `"b2"` by 1.
    """
    df = _to_df(adata, layer)
    dict_classes = {}
    columns_removed = []

    for column in df.columns:
        if df[column].dtype == "bool":
            df[column] = df[column].astype(int).values
            continue
        if is_numeric_dtype(df[column].dtype):
            continue
        try:
            df[column] = pd.to_numeric(df[column].values)
            continue
        except:
            from sklearn.preprocessing import LabelEncoder

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

    fcswrite.write_fcs(str(path), list(df.columns), df.values, **fcswrite_kwargs)

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
