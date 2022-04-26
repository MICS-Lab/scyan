import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from .. import Scyan
from ..utils import _optional_show


@_optional_show
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
    h = model().detach().numpy()
    df = pd.concat(
        [predictions, pd.DataFrame(h, columns=model.marker_pop_matrix.columns)], axis=1
    )
    df = df.groupby("Population").median()
    heatmap = df.loc[model.marker_pop_matrix.index]
    if show_diff:
        heatmap -= model.marker_pop_matrix
    sns.heatmap(heatmap, cmap="coolwarm", center=0)
    plt.title("Marker matrix approximation by the model embedding space")


@_optional_show
def probs_per_marker(
    model: Scyan,
    population: str,
    obs_key: str = "scyan_pop",
    prob_name: str = "Prob",
    show: bool = True,
    vmin_threshold: int = -100,
):
    """Plots a heatmap of marker pronbabilities for each population

    Args:
        model (Scyan): Scyan model
        where (ArrayLike): Array where cells have to be considered
        prob_name (str, optional): Name displayed on the plot. Defaults to "Prob".
        show (bool, optional): Whether to plt.show() or not. Defaults to True.
    """
    where = model.adata.obs[obs_key] == population
    u = model.module(model.x[where], model.covariates[where])[0]

    log_probs = model.module.prior.log_prob_per_marker(u)
    mean_log_probs = log_probs.mean(dim=0).detach().numpy()

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
