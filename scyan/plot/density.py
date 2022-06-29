import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union
import numpy as np
from scipy.stats import norm
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import matplotlib
from scipy import stats
import torch

from .. import Scyan
from .utils import optional_show, check_population, get_palette_others, select_markers


@optional_show
@check_population(return_list=True)
def kde_per_population(
    model: Scyan,
    populations: Union[str, List[str]],
    obs_key: str = "scyan_pop",
    markers: Union[List[str], None] = None,
    n_markers: Union[int, None] = 3,
    ncols: int = 4,
    var_name: str = "Marker",
    value_name: str = "Expression",
    show: bool = True,
):
    """Plots a KDE for each population for a given marker

    Args:
        model (Scyan): Scyan model
        where (ArrayLike): Array where cells have to be considered.
        cell_type_key (str): Key that gets the cell_type, e.g. 'scyan_pop'
        markers (Union[List[str], None], optional): List of markers to consider. None means all markers being considered. Defaults to None.
        ncols (int, optional): Number of columns to be displayed. Defaults to 4.
        hue_name (str, optional): Hue name. Defaults to "Population".
        var_name (str, optional): Var name. Defaults to "Marker".
        value_name (str, optional): Value name. Defaults to "Expression".
        show (bool, optional): Whether to plt.show() or not. Defaults to True.
    """
    markers = select_markers(model, markers, n_markers, obs_key, populations)

    df = model.adata.to_df()

    keys = model.adata.obs[obs_key]
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
        hue_order=populations + ["Others"],
    )


@torch.no_grad()
@optional_show
@check_population()
def latent_expressions(
    model: Scyan,
    population: str,
    obs_key: str = "scyan_pop",
    max_value: float = 1.5,
    num_pieces: int = 100,
    radius: float = 0.05,
    show: bool = True,
):
    condition = model.adata.obs[obs_key] == population
    u_mean = model.module(model.x[condition], model.covariates[condition])[0].mean(dim=0)
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
