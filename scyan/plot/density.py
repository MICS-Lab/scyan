from typing import List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData

from ..utils import _get_subset_indices
from .utils import check_population, get_palette_others, plot_decorator, select_markers


@plot_decorator(adata=True)
@check_population(return_list=True)
def kde(
    adata: AnnData,
    population: Union[str, List[str], None],
    markers: Optional[List[str]] = None,
    obs_key: str = "scyan_pop",
    n_markers: Optional[int] = 3,
    n_cells: Optional[int] = 100_000,
    ncols: int = 2,
    var_name: str = "Marker",
    value_name: str = "Expression",
    show: bool = True,
):
    """Plot Kernel-Density-Estimation for each provided population and for multiple markers.

    Args:
        adata: An `anndata` object.
        population: One population, or a list of population to be analyzed, or `None`. If not `None`, the population name(s) has to be in `adata.obs[obs_key]`.
        markers: List of markers to plot. If `None`, the list is chosen automatically.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        n_markers: Number of markers to choose automatically if `markers is None`.
        n_cells: Number of cells to be considered for the heatmap (to accelerate it when $N$ is very high). If `None`, consider all cells.
        ncols: Number of figures per row.
        var_name: Name displayed on the graphs.
        value_name: Name displayed on the graphs.
        show: Whether or not to display the figure.
    """
    indices = _get_subset_indices(adata.n_obs, n_cells)
    adata = adata[indices]

    markers = select_markers(adata, markers, n_markers, obs_key, population, 1)

    df = adata.to_df()

    if population is None:
        df = pd.melt(
            df,
            value_vars=markers,
            var_name=var_name,
            value_name=value_name,
        )

        sns.displot(
            df,
            x=value_name,
            col=var_name,
            col_wrap=ncols,
            kind="kde",
            common_norm=False,
            facet_kws=dict(sharey=False),
        )
        return

    keys = adata.obs[obs_key]
    df[obs_key] = np.where(~np.isin(keys, population), "Others", keys)

    df = pd.melt(
        df,
        id_vars=[obs_key],
        value_vars=markers,
        var_name=var_name,
        value_name=value_name,
    )

    sns.displot(
        df,
        x=value_name,
        col=var_name,
        hue=obs_key,
        col_wrap=ncols,
        kind="kde",
        common_norm=False,
        facet_kws=dict(sharey=False),
        palette=get_palette_others(df, obs_key),
        hue_order=sorted(df[obs_key].unique(), key="Others".__eq__),
    )
