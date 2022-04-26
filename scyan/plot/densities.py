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

from .. import Scyan
from ..utils import _optional_show


@_optional_show
def kde_per_population(
    model: Scyan,
    population: str,
    obs_key: str = "scyan_pop",
    markers: Union[List[str], None] = None,
    ncols: int = 4,
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
    hue_name = f"{obs_key} is '{population}'"
    markers = model.adata.var_names if markers is None else markers

    df = model.adata.to_df()
    df[hue_name] = pd.Categorical(model.adata.obs[obs_key] == population)
    df = pd.melt(
        df,
        id_vars=[hue_name],
        value_vars=markers,
        var_name=var_name,
        value_name=value_name,
    )

    sns.displot(
        df,
        x=value_name,
        col=var_name,
        hue=hue_name,
        col_wrap=ncols,
        kind="kde",
        common_norm=False,
        sharey=False,
    )


@_optional_show
def latent_expressions(
    model: Scyan, population: str, obs_key: str = "scyan_pop", show: bool = True
):
    """Plots all markers in one graph

    Args:
        model (Scyan): Scyan model
        where (ArrayLike): Array where cells have to be considered
        show (bool, optional): Whether to plt.show() or not. Defaults to True.
    """
    where = model.adata.obs[obs_key] == population
    u_mean = model.module(model.x[where], model.covariates[where])[0].mean(dim=0)

    labels = model.marker_pop_matrix.columns
    values = u_mean.detach().numpy()

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
    population: str,
    obs_key: str = "scyan_pop",
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
    adata1 = model.adata[model.adata.obs[obs_key] == population]
    if ref is None:
        adata2 = model.adata[model.adata.obs[obs_key] != population]
    else:
        adata2 = model.adata[model.adata.obs[obs_key] == ref]

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
    df["proba"] = model.predict_proba()[population].values
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
                mlines.Line2D([], [], color="C0", marker="s", ls="", label=population),
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
        f"KDE of cells weighted by the probability of {population}{'' if ref is None else f' and ref {ref}'}"
    )
