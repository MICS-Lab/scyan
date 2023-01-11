from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .. import Scyan
from .utils import check_population, plot_decorator


@torch.no_grad()
@plot_decorator()
@check_population(one=True)
def probs_per_marker(
    model: Scyan,
    population: str,
    obs_key: str = "scyan_pop",
    prob_name: str = "Prob",
    vmin_threshold: int = -100,
    figsize: Tuple[float] = (10, 6),
    show: bool = True,
):
    """Interpretability tool: get a group of cells and plot a heatmap of marker probabilities for each population.

    Args:
        model: Scyan model.
        population: Name of one population to interpret. To be valid, the population name has to be in `adata.obs[obs_key]`.
        obs_key: Key to look for population in `adata.obs`. By default, uses the model predictions.
        prob_name: Name to display on the plot.
        vmin_threshold: Minimum threshold for the heatmap colorbar.
        figsize: Pair `(width, height)` indicating the size of the figure.
        show: Whether or not to display the figure.
    """
    u = model(model.adata.obs[obs_key] == population)

    log_probs = model.module.prior.log_prob_per_marker(u)
    mean_log_probs = log_probs.mean(dim=0).numpy(force=True)

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

    plt.figure(figsize=figsize)
    sns.heatmap(df_probs, cmap="magma", vmin=max(vmin_threshold, mean_log_probs.min()))
    plt.title("Log probabilities per marker for each population")
