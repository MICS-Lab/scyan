from itertools import groupby
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .. import Scyan
from ..utils import _requires_fit
from .utils import check_population, optional_show


@torch.no_grad()
@_requires_fit
@optional_show
@check_population()
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
        show: Whether or not to display the plot.
    """
    where = model.adata.obs[obs_key] == population
    u = model.module(model.x[where], model.covariates[where])[0]

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
@_requires_fit
@optional_show
def latent_heatmap(model: Scyan, obs_key: str = "scyan_pop", show: bool = True):
    """Show Scyan latent space for each population.

    Args:
        model: Scyan model.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        show: Whether or not to display the plot.
    """
    u = model()

    df = pd.DataFrame(u.cpu().numpy(), columns=model.var_names)
    df["Population"] = model.adata.obs[obs_key].values

    sns.heatmap(df.groupby("Population").mean(), vmax=1.2, vmin=-1.2, cmap="coolwarm")


@torch.no_grad()
@_requires_fit
@optional_show
def subclusters(
    model: Scyan,
    obs_key: str = "scyan_pop",
    subcluster_key: str = "subcluster_index",
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
):
    """Display Scyan latent space for each population sub-clusters.
    !!! warning
        To run this plot function, you have to run [scyan.preprocess.subcluster][scyan.preprocess.subcluster] first.

    Args:
        model: Scyan model.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        subcluster_key: Key created by `scyan.preprocess.subcluster`.
        figsize: Matplotlib figure size. Increase it if you have display issues.
        show: Whether or not to display the plot.
    """
    assert (
        subcluster_key in model.adata.obs.columns
    ), "Subcluster key '{subcluster_key}' was not found in adata.obs - have you run scyan.preprocess.subcluster before?"

    plt.figure(figsize=figsize)

    u = model()

    df = pd.DataFrame(u.cpu().numpy(), columns=model.var_names)
    df[obs_key] = model.adata.obs[obs_key].values
    df[subcluster_key] = model.adata.obs[subcluster_key].values

    df = df.groupby([obs_key, subcluster_key]).mean().dropna()
    pops = df.index.get_level_values(obs_key)
    df.index = list(df.index.get_level_values(subcluster_key))

    ax = sns.heatmap(df, vmax=1.2, vmin=-1.2, cmap="coolwarm")
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
