from typing import Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import groupby

from .. import Scyan
from .utils import optional_show, check_population


@torch.no_grad()
@optional_show
def marker_matrix_reconstruction(
    model: Scyan, show_diff: bool = False, show: bool = True
):
    """Reconstructs the marker matrix based on predictions

    Args:
        model (Scyan): Scyan model
        show_diff (bool, optional): Whether do show the difference with the actual marker-population matrix. Defaults to False.
        show (bool, optional): Whether to plt.show() or not. Defaults to True.
    """
    predictions = model.predict(key_added=None)
    predictions.name = "Population"
    h = model().cpu().numpy()
    df = pd.concat(
        [predictions, pd.DataFrame(h, columns=model.marker_pop_matrix.columns)], axis=1
    )
    df = df.groupby("Population").median()
    heatmap = df.loc[model.marker_pop_matrix.index]
    if show_diff:
        heatmap -= model.marker_pop_matrix
    sns.heatmap(heatmap, cmap="coolwarm", center=0)
    plt.title("Marker matrix approximation by the model embedding space")


@torch.no_grad()
@optional_show
@check_population()
def probs_per_marker(
    model: Scyan,
    population: str,
    obs_key: str = "scyan_pop",
    prob_name: str = "Prob",
    show: bool = True,
    vmin_threshold: int = -100,
):
    """Plots a heatmap of marker probabilities for each population

    Args:
        model (Scyan): Scyan model
        where (ArrayLike): Array where cells have to be considered
        prob_name (str, optional): Name displayed on the plot. Defaults to "Prob".
        show (bool, optional): Whether to plt.show() or not. Defaults to True.
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
@optional_show
def latent_heatmap(model: Scyan, obs_key: str = "scyan_pop", show: bool = True):
    u = model()

    df = pd.DataFrame(u.cpu().numpy(), columns=model.var_names)
    df["Population"] = model.adata.obs[obs_key].values

    sns.heatmap(df.groupby("Population").mean(), center=0, cmap="coolwarm")


@torch.no_grad()
@optional_show
def subclusters(
    model: Scyan,
    obs_key: str = "scyan_pop",
    subcluster_key: str = "leiden_subcluster",
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
):
    plt.figure(figsize=figsize)

    u = model()

    df = pd.DataFrame(u.cpu().numpy(), columns=model.var_names)
    df[obs_key] = model.adata.obs[obs_key].values
    df[subcluster_key] = model.adata.obs[subcluster_key].values

    df = df.groupby([obs_key, subcluster_key]).mean().dropna()
    pops = df.index.get_level_values(obs_key)
    df.index = list(df.index.get_level_values(subcluster_key))

    ax = sns.heatmap(df, center=0, cmap="coolwarm")
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
