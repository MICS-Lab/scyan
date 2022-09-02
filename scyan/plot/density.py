from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import norm

from .. import Scyan
from ..utils import _get_subset_indices, _requires_fit
from .utils import check_population, get_palette_others, optional_show, select_markers


@optional_show
@check_population(return_list=True)
def kde_per_population(
    model: Scyan,
    populations: Union[str, List[str]],
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
        populations: One population or a list of population to interpret. To be valid, a population name have to be in `adata.obs[obs_key]`.
        markers: List of markers to plot. If `None`, the list is chosen automatically.
        n_markers: Number of markers to choose automatically if `markers is None`.
        n_cells: Number of cells to be considered for the heatmap (to accelerate it when $N$ is very high). If `None`, consider all cells.
        ncols: Number of figures per row.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        var_name: Name displayed on the graphs.
        value_name: Name displayed on the graphs.
        show: Whether or not to display the plot.
    """
    indices = _get_subset_indices(model.adata, n_cells)
    adata = model.adata[indices]

    markers = select_markers(model, markers, n_markers, obs_key, populations, 1)

    df = adata.to_df()

    keys = adata.obs[obs_key]
    df[obs_key] = np.where(~np.isin(keys, populations), "Others", keys)

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
        hue_order=list(populations) + ["Others"],
    )


@torch.no_grad()
@_requires_fit
@optional_show
@check_population(one=True)
def latent_expressions(
    model: Scyan,
    population: str,
    obs_key: str = "scyan_pop",
    max_value: float = 1.5,
    num_pieces: int = 100,
    radius: float = 0.05,
    show: bool = True,
):
    """Plot latent expressions of a group of cells (every marker in one plot).

    Args:
        model: Scyan model.
        population: Name of one population to interpret. To be valid, the population name has to be in `adata.obs[obs_key]`.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        max_value: Maximum absolute latent value.
        num_pieces: Number of pieces to display the colorbar.
        radius: Radius used to chunk the colorbar. Increase this value if multiple names overlap.
        show: Whether or not to display the plot.
    """
    condition = model.adata.obs[obs_key] == population
    u_mean = model(condition).mean(dim=0)
    values = u_mean.cpu().numpy().clip(-max_value, max_value)

    y = np.linspace(-max_value, max_value, num_pieces + 1)
    cmap = plt.get_cmap("RdBu")
    y_cmap = norm.pdf(np.abs(y) - 1, scale=model.hparams.prior_std)
    y_cmap = y_cmap - y_cmap.min()
    y_cmap = 0.5 - np.sign(y) * (y_cmap / y_cmap.max() / 2)
    colors = cmap(y_cmap).clip(0, 0.8)

    plt.figure(figsize=(2, 6), dpi=100)
    plt.vlines(np.zeros(num_pieces), y[:-1], y[1:], colors=colors, linewidth=5)
    plt.annotate("Pos", (-0.7, 1), fontsize=15)
    plt.annotate("Neg", (-0.7, -1), fontsize=15)

    for v in np.arange(-max_value, max_value, 2 * radius):
        labels = [
            label
            for value, label in zip(values, model.var_names)
            if abs(v - value) < radius
        ]
        if labels:
            plt.plot([0, 0.1], [v, v], "k")
            plt.annotate(", ".join(labels), (0.2, v - 0.03))

    plt.xlim([-1, 1])
    plt.axis("off")
