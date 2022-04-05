import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from typing import Callable, List, Union
from anndata import AnnData
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import matplotlib
from scipy import stats

from . import Scyan


def _optional_show(f: Callable) -> Callable:
    """Decorator that shows a matplotlib figure if the provided 'show' argument is True"""

    def wrapper(*args, **kwargs):
        f(*args, **kwargs)
        if kwargs.get("show", True):
            plt.show()

    return wrapper


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
def kde_per_population(
    model: Scyan,
    where: ArrayLike,
    cell_type_key: str,
    markers: Union[List[str], None] = None,
    ncols: int = 4,
    hue_name: str = "Population",
    var_name: str = "Marker",
    value_name: str = "Expression",
    show: bool = True,
):
    """Plots a KDE for each population for a given marker

    Args:
        model (Scyan): Scyan model
        where (ArrayLike): Array where cells have to be considered.
        cell_type_key (str): Key that gets the cell_type, e.g. 'scyan_knn_pop'
        markers (Union[List[str], None], optional): List of markers to consider. None means all markers being considered. Defaults to None.
        ncols (int, optional): Number of columns to be displayed. Defaults to 4.
        hue_name (str, optional): Hue name. Defaults to "Population".
        var_name (str, optional): Var name. Defaults to "Marker".
        value_name (str, optional): Value name. Defaults to "Expression".
        show (bool, optional): Whether to plt.show() or not. Defaults to True.
    """
    adata = model.adata[where]
    markers = adata.var_names if markers is None else markers

    df = adata.to_df()
    df[hue_name] = adata.obs[cell_type_key]
    df = pd.melt(
        df,
        id_vars=[hue_name],
        value_vars=markers,
        var_name=var_name,
        value_name=value_name,
    )

    grid = sns.FacetGrid(
        df, col=var_name, hue=hue_name, col_wrap=ncols, sharex=False, sharey=False
    )
    grid.map(sns.histplot, value_name, kde=True)
    grid.add_legend()


@_optional_show
def probs_per_marker(
    model: Scyan, where: ArrayLike, prob_name: str = "Prob", show: bool = True
):
    """Plots a heatmap of marker pronbabilities for each population

    Args:
        model (Scyan): Scyan model
        where (ArrayLike): Array where cells have to be considered
        prob_name (str, optional): Name displayed on the plot. Defaults to "Prob".
        show (bool, optional): Whether to plt.show() or not. Defaults to True.
    """
    u = model.module(model.x[where], model.covariates[where])[0]
    normal = torch.distributions.normal.Normal(0, model.hparams.prior_std)

    h = model.module.difference_to_modes(u)

    _probs_per_marker = normal.log_prob(h).mean(dim=0).detach().numpy()

    df_probs = pd.DataFrame(
        _probs_per_marker,
        columns=model.adata.var_names,
        index=model.marker_pop_matrix.index,
    )
    df_probs = df_probs.reindex(
        df_probs.mean().sort_values(ascending=False).index, axis=1
    )
    means = df_probs.mean(axis=1)
    means = means / means.min() * df_probs.values.min()
    df_probs.insert(0, prob_name, means)
    df_probs.insert(1, " ", np.nan)
    df_probs.sort_values(by=prob_name, inplace=True, ascending=False)
    sns.heatmap(df_probs, cmap="magma", vmin=max(-100, _probs_per_marker.min()))
    plt.title("Log probabilities per marker for each population")


@_optional_show
def latent_expressions(model: Scyan, where: ArrayLike, show: bool = True):
    """Plots all markers in one graph

    Args:
        model (Scyan): Scyan model
        where (ArrayLike): Array where cells have to be considered
        show (bool, optional): Whether to plt.show() or not. Defaults to True.
    """
    h_mean = model.module(model.x[where], model.covariates[where])[0].mean(dim=0)

    labels = model.marker_pop_matrix.columns
    values = h_mean.detach().numpy()

    x_pdf = np.linspace(min(-1.5, values.min()), max(values.max(), 1.5), 200)
    y_pdf = 0.5 * (
        norm.pdf(x_pdf, loc=1, scale=model.hparams.prior_std)
        + norm.pdf(x_pdf, loc=-1, scale=model.hparams.prior_std)
    )

    plt.figure(figsize=(10, 5))
    plt.plot(x_pdf, y_pdf, "--", color="grey")
    plt.xticks([-1, 0, 1])
    plt.yticks([0, round(y_pdf.max(), 1)])

    cmap = matplotlib.cm.get_cmap("viridis")

    for value, label in sorted(zip(values, labels)):
        plt.scatter([value], [0], c=[cmap(value)], label=label)
        plt.annotate(
            label,
            (value - 0.02, 0.02),
            rotation=90,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )
    plt.legend()


@_optional_show
def pop_weighted_kde(
    model: Scyan,
    pop: str,
    n_samples: int = 5000,
    alpha: float = 0.2,
    thresh: float = 0.05,
    ref: Union[str, None] = None,
    show: bool = True,
):
    """Plots a Kernel Density Estimation on a scatterplot

    Args:
        model (Scyan): Scyan model
        pop (str): Population considered
        n_samples (int, optional): Number of dots to be plot. Defaults to 5000.
        alpha (float, optional): Scatter plot transparancy. Defaults to 0.2.
        thresh (float, optional): KDE threshold. Defaults to 0.05.
        ref (Union[str, None], optional): Population reference. None means no population reference. Defaults to None.
        show (bool, optional): Whether to plt.show() or not. Defaults to True.
    """
    adata1 = model.adata[model.adata.obs.cell_type == pop]
    if ref is None:
        adata2 = model.adata[model.adata.obs.cell_type != pop]
    else:
        adata2 = model.adata[model.adata.obs.cell_type == ref]

    markers_statistics = [
        (
            stats.kstest(
                adata1[:, marker].X.flatten(), adata2[:, marker].X.flatten()
            ).statistic,
            marker,
        )
        for marker in model.marker_pop_matrix.columns
    ]
    markers = [marker for _, marker in sorted(markers_statistics, reverse=True)]

    df = model.adata.to_df()
    df["proba"] = model.predict_proba()[pop].values
    if ref is not None:
        df["proba_ref"] = model.predict_proba()[ref].values
    df = df.sample(n=n_samples, random_state=0)

    if ref is not None:
        sns.kdeplot(
            data=df,
            x=markers[0],
            y=markers[1],
            weights="proba_ref",
            fill=True,
            thresh=thresh,
            color="C1",
        )
        plt.legend(
            handles=[
                mlines.Line2D([], [], color="C1", marker="s", ls="", label=ref),
                mlines.Line2D([], [], color="C0", marker="s", ls="", label=pop),
            ]
        )
    sns.kdeplot(
        data=df,
        x=markers[0],
        y=markers[1],
        weights="proba",
        fill=True,
        thresh=thresh,
        color="C0",
    )
    sns.scatterplot(data=df, x=markers[0], y=markers[1], alpha=alpha, color=".0", s=5)
    plt.title(
        f"KDE of cells weighted by the probability of {pop}{'' if ref is None else f' and ref {ref}'}"
    )
