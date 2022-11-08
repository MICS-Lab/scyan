from typing import List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns

from .. import Scyan
from ..utils import _get_subset_indices
from .utils import check_population, get_palette_others, plot_decorator, select_markers


@plot_decorator
@check_population(return_list=True)
def kde_per_population(
    model: Scyan,
    population: Union[str, List[str]],
    markers: Optional[List[str]] = None,
    n_markers: Optional[int] = 3,
    n_cells: Optional[int] = 100000,
    ncols: int = 2,
    obs_key: str = "scyan_pop",
    var_name: str = "Marker",
    value_name: str = "Expression",
    show: bool = True,
):
    """Plot Kernel-Density-Estimation for each provided population and for multiple markers.

    Args:
        model: Scyan model.
        population: One population or a list of population to interpret. To be valid, a population name have to be in `adata.obs[obs_key]`.
        markers: List of markers to plot. If `None`, the list is chosen automatically.
        n_markers: Number of markers to choose automatically if `markers is None`.
        n_cells: Number of cells to be considered for the heatmap (to accelerate it when $N$ is very high). If `None`, consider all cells.
        ncols: Number of figures per row.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        var_name: Name displayed on the graphs.
        value_name: Name displayed on the graphs.
        show: Whether or not to display the figure.
    """
    indices = _get_subset_indices(model.adata, n_cells)
    adata = model.adata[indices]

    markers = select_markers(model, markers, n_markers, obs_key, population, 1)

    df = adata.to_df()

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
