from typing import List, Optional, Union

import pandas as pd
from anndata import AnnData


def _get_counts(adata: AnnData, groupby, obs_key, normalize) -> pd.DataFrame:
    if groupby is None:
        return adata.obs[[obs_key]].apply(lambda s: s.value_counts(normalize)).T

    grouped = adata.obs.groupby(groupby)
    return grouped[obs_key].apply(lambda s: s.value_counts(normalize)).unstack(level=-1)


def count_cell_populations(
    adata: AnnData,
    groupby: Union[str, List[str], None] = None,
    normalize: bool = False,
    obs_key: str = "scyan_pop",
    among: str = None,
) -> pd.DataFrame:
    """Count for each patient (or group) the number of cells for each population.

    Args:
        adata: An `AnnData` object.
        groupby: Key(s) of `adata.obs` used to create groups (e.g. the patient ID).
        normalize: If `True`, returns percentage instead of counts.
        obs_key: Key of `adata.obs` containing the population names (or the values to count).
        among: Key of `adata.obs` containing the parent population name. For example, if 'T CD4 RM' is found in `adata.obs[obs_key]`, then we may find something like 'T cell' in `adata.obs[among]`. Typically, if using hierarchical populations, you can provide `'scyan_pop_level'` with your level name.

    Returns:
        A DataFrame of counts (one row per group, one column per population).
    """
    normalize = among is not None or normalize
    column_suffix = "percentage" if normalize else "count"

    counts = _get_counts(adata, groupby, obs_key, normalize)

    if among is None:
        counts.columns = [f"{name} {column_suffix}" for name in counts.columns]
        return counts

    parents_count = _get_counts(adata, groupby, among, normalize)

    df_parent = (
        adata.obs.groupby(among)[obs_key].apply(lambda s: s.value_counts()).unstack()
    )
    assert (
        (df_parent > 0).sum(0) == 1
    ).all(), f"Each population from adata.obs['{obs_key}'] should have one and only one parent population in adata.obs['{among}']"
    to_parent_dict = dict(df_parent.idxmax())

    counts /= parents_count[[to_parent_dict[pop] for pop in counts.columns]].values
    counts.columns = [
        f"{pop} {column_suffix} among {to_parent_dict[pop]}" for pop in counts.columns
    ]
    return counts


def mean_intensities(
    adata: AnnData,
    groupby: Union[str, List[str], None] = None,
    obs_key: str = "scyan_pop",
    unstack_join: Optional[str] = " mean intensity on ",
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
    obsm_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute the Mean Metal Intensity (MMI) or Mean Fluorescence Intensity (MFI) per population. If needed, mean intensities can be computed per group (e.g., per patient) by providing the `groupby` argument.

    Args:
        adata: An `AnnData` object.
        groupby: Key(s) of `adata.obs` used to create groups. For instance, `"id"` computes MMI per population for each ID. You can also provide something like `["group", "id"]` to get MMI per group, and per patient inside each group.
        obs_key: Key of `adata.obs` containing the population names.
        unstack_join: If `None`, keep the information grouped. Else, flattens the biomarkers into one series (or one row per group if `groupby` is a list) and uses `unstack_join` to join the names of the multi-level columns. For instance, `' expression on '` can be a good choice.
        layer: In which `adata.layers` we get fluorescence intensities. By default, it uses `adata.X`.
        obsm: In which `adata.obsm` we get fluorescence intensities. By default, it uses `adata.X`. If not `None` then `obsm_names` is required too.
        obsm_names: Ordered list of names in `adata.obsm[obsm]` if `obsm` was provided.

    Returns:
        A DataFrame of MFI. If `groupby` was a list, it is a multi-index dataframe.
    """
    if groupby is None:
        groupby = [obs_key]
    elif isinstance(groupby, str):
        groupby = [groupby, obs_key]
    else:
        groupby = list(groupby) + [obs_key]

    if obsm is not None:
        assert (
            layer is None
        ), "You must choose between 'obsm' and 'layer', do not use both."

        df = pd.DataFrame(data=adata.obsm[obsm], columns=obsm_names)
    else:
        df = adata.to_df(layer)

    for group in groupby:
        df[group] = adata.obs[group].values

    res = df.groupby(groupby).mean()

    if unstack_join is None:
        return res

    res = res.unstack(level=-1)
    if isinstance(res, pd.Series):
        res.index = [unstack_join.join(row).strip() for row in res.index.values]
    else:
        res.columns = [unstack_join.join(col).strip() for col in res.columns.values]
    return res
