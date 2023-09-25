from typing import List, Optional, Union

import matplotlib.pyplot as plt
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
    key: str = "scyan_pop",
    n_markers: Optional[int] = 3,
    n_cells: Optional[int] = 100_000,
    ncols: int = 2,
    var_name: str = "Marker",
    value_name: str = "Expression",
    show: bool = True,
):
    """Plot Kernel-Density-Estimation for each provided population and for multiple markers.

    Args:
        adata: An `AnnData` object.
        population: One population, or a list of population to be analyzed, or `None`. If not `None`, the population name(s) has to be in `adata.obs[key]`.
        markers: List of markers to plot. If `None`, the list is chosen automatically.
        key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        n_markers: Number of markers to choose automatically if `markers is None`.
        n_cells: Number of cells to be considered for the heatmap (to accelerate it when $N$ is very high). If `None`, consider all cells.
        ncols: Number of figures per row.
        var_name: Name displayed on the graphs.
        value_name: Name displayed on the graphs.
        show: Whether or not to display the figure.
    """
    indices = _get_subset_indices(adata.n_obs, n_cells)
    adata = adata[indices]

    markers = select_markers(adata, markers, n_markers, key, population, 1)

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

    keys = adata.obs[key]
    df[key] = np.where(~np.isin(keys, population), "Others", keys)

    df = pd.melt(
        df,
        id_vars=[key],
        value_vars=markers,
        var_name=var_name,
        value_name=value_name,
    )

    sns.displot(
        df,
        x=value_name,
        col=var_name,
        hue=key,
        col_wrap=ncols,
        kind="kde",
        common_norm=False,
        facet_kws=dict(sharey=False),
        palette=get_palette_others(df, key),
        hue_order=sorted(df[key].unique(), key="Others".__eq__),
    )


@plot_decorator(adata=True)
def log_prob_threshold(adata: AnnData, show: bool = True):
    """Plot the number of cells annotated depending on the log probability threshold (below which cells are left non-classified). It can be helpful to determine the best threshold value, i.e. before a significative decrease in term of number of cells annotated.

    !!! note
        To use this function, you first need to fit a `scyan.Scyan` model and use the `model.predict()` method.

    Args:
        adata: The `AnnData` object used during the model training.
        show: Whether or not to display the figure.
    """
    assert (
        "scyan_log_probs" in adata.obs
    ), f"Cannot find 'scyan_log_probs' in adata.obs. Have you run model.predict()?"

    x = np.sort(adata.obs["scyan_log_probs"])
    y = 1 - np.arange(len(x)) / float(len(x))

    plt.plot(x, y)
    plt.xlim(-100, x.max())
    sns.despine(offset=10, trim=True)
    plt.ylabel("Ratio of predicted cells")
    plt.xlabel("Log density threshold")
