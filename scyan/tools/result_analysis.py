from typing import List, Optional, Union

import pandas as pd
from anndata import AnnData


def count_cell_populations(
    adata: AnnData, groupby: str, normalize: bool = False, obs_key: str = "scyan_pop"
) -> pd.DataFrame:
    """Count for each patient (or group) the number of cells for each population.

    Args:
        adata: An `AnnData` object.
        groupby: Key of `adata.obs` used to create groups (e.g. the patient ID).
        normalize: If `True`, returns percentage instead of counts.
        obs_key: Key of `adata.obs` containing the population names (or the values to count).

    Returns:
        A DataFrame of counts (one row per group, one column per population).
    """
    grouped = adata.obs.groupby(groupby)
    counts = grouped[obs_key].apply(lambda s: s.value_counts(normalize))
    return counts.unstack(level=-1)


def mean_intensities(
    adata: AnnData,
    groupby: Union[str, List[str]] = "scyan_pop",
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
    obsm_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute the Mean Metal Intensity (MMI) or Mean Fluorescence Intensity (MFI) per population.

    Args:
        adata: An `AnnData` object.
        groupby: Key(s) of `adata.obs` used to create groups. E.g., `scyan_pop` creates group of populations. It can be one string, or a list of strings: for instance, use `["id", "scyan_pop"]` computes MMI per population for each ID.
        layer: In which `adata.layers` we get fluorescence intensities. By default, it uses `adata.X`.
        obsm: In which `adata.obsm` we get fluorescence intensities. By default, it uses `adata.X`. If not `None` then `obsm_names` is required too.
        obsm_names: Ordered list of names in `adata.obsm[obsm]` if `obsm` was provided.

    Returns:
        A DataFrame of MFI. If `groupby` was a list, it is a multi-index dataframe.
    """
    if isinstance(groupby, str):
        groupby = [groupby]

    if obsm is not None:
        assert (
            layer is None
        ), "You must choose between 'obsm' and 'layer', do not use both."

        df = pd.DataFrame(data=adata.obsm[obsm], columns=obsm_names)
    else:
        df = adata.to_df(layer)

    for group in groupby:
        df[group] = adata.obs[group].values

    return df.groupby(groupby).mean()
