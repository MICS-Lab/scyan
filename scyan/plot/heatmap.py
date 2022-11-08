from itertools import groupby
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .. import Scyan
from ..utils import _get_subset_indices, _requires_fit
from .utils import check_population, plot_decorator


@torch.no_grad()
@plot_decorator
@_requires_fit
@check_population(one=True)
def probs_per_marker(
    model: Scyan,
    population: str,
    obs_key: str = "scyan_pop",
    prob_name: str = "Prob",
    vmin_threshold: int = -100,
    show: bool = True,
):
    """Interpretability tool: get a group of cells and plot a heatmap of marker probabilities for each population.

    Args:
        model: Scyan model.
        population: Name of one population to interpret. To be valid, the population name has to be in `adata.obs[obs_key]`.
        obs_key: Key to look for population in `adata.obs`. By default, uses the model predictions.
        prob_name: Name to display on the plot.
        vmin_threshold: Minimum threshold for the heatmap colorbar.
        show: Whether or not to display the figure.
    """
    u = model(model.adata.obs[obs_key] == population)

    log_probs = model.module.prior.log_prob_per_marker(u)
    mean_log_probs = log_probs.mean(dim=0).cpu().numpy()

    df_probs = pd.DataFrame(
        mean_log_probs,
        columns=model.var_names,
        index=model.pop_names,
    )
    df_probs = df_probs.reindex(
        df_probs.mean().sort_values(ascending=False).index, axis=1
    )
    means = df_probs.mean(axis=1)
    means = means / means.min() * df_probs.values.min()
    df_probs.insert(0, prob_name, means)
    df_probs.insert(1, " ", np.nan)
    df_probs.sort_values(by=prob_name, inplace=True, ascending=False)
    sns.heatmap(df_probs, cmap="magma", vmin=max(vmin_threshold, mean_log_probs.min()))
    plt.title("Log probabilities per marker for each population")


@torch.no_grad()
@plot_decorator
@_requires_fit
def subcluster(
    model: Scyan,
    n_cells: Optional[int] = 200000,
    latent: bool = True,
    obs_key: str = "scyan_pop",
    subcluster_key: str = "subcluster_index",
    figsize: Tuple[int, int] = (10, 5),
    vmax: float = 1.2,
    vmin: float = -1.2,
    cmap: Optional[str] = None,
    show: bool = True,
):
    """Display Scyan latent space for each population sub-clusters.
    !!! warning
        To run this plot function, you have to run [scyan.tools.subcluster][] first.

    !!! note
        If using the latent space, it will only show the marker you provided to Scyan. Else, it shows every marker of the panel.

    Args:
        model: Scyan model.
        n_cells: Number of cells to be considered for the heatmap (to accelerate it when $N$ is very high). If `None`, consider all cells.
        latent: If `True`, displays Scyan's latent expressions, else just the standardized expressions.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        subcluster_key: Key created by `scyan.tools.subcluster`.
        figsize: Matplotlib figure size. Increase it if you have display issues.
        vmax: Maximum value on the heatmap.
        vmax: Minimum value on the heatmap.
        cmap: Colormap name. By default, uses `"coolwarm"` if `latent`, else `"viridis"`.
        show: Whether or not to display the figure.
    """
    assert (
        subcluster_key in model.adata.obs.columns
    ), "Subcluster key '{subcluster_key}' was not found in adata.obs - have you run scyan.tools.subcluster before?"

    plt.figure(figsize=figsize)

    indices = _get_subset_indices(model.adata, n_cells)
    adata = model.adata[indices]

    x = model(indices).cpu().numpy() if latent else adata.X
    columns = model.var_names if latent else adata.var_names

    df = pd.DataFrame(x, columns=columns)
    df[obs_key] = adata.obs[obs_key].values
    df[subcluster_key] = adata.obs[subcluster_key].values

    df = df.groupby([obs_key, subcluster_key]).mean().dropna()
    pops = df.index.get_level_values(obs_key)
    df.index = list(df.index.get_level_values(subcluster_key))

    if cmap is None:
        cmap = "coolwarm" if latent else "viridis"

    ax = sns.heatmap(df, vmax=vmax, vmin=vmin, cmap=cmap)
    trans = ax.get_xaxis_transform()

    x0 = -1
    delta = 0.2
    n_pops = len(df.index)
    scale = 1 / n_pops
    groups = [(label, len(list(g))) for label, g in groupby(pops)]

    pos = 0
    for i, (label, k) in enumerate(groups):
        y1, y2 = pos, pos + k
        y_line1, y_line2 = (n_pops - y1 - 0.5) * scale, (n_pops - y2 + 0.5) * scale

        ax.plot(
            [x0 + delta, x0, x0, x0 + delta],
            [y_line1, y_line1, y_line2, y_line2],
            color="k",
            transform=trans,
            clip_on=False,
        )
        if i:
            plt.plot([0, len(model.var_names)], [y1, y1], color="black")
        plt.text(x0 - delta, (y1 + y2) / 2, label, ha="right", va="center")

        pos += k

    plt.yticks(rotation=0)
