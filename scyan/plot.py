import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from typing import List, Union
from anndata import AnnData
import numpy as np
from scipy.stats import norm
import matplotlib.patheffects as pe
import matplotlib

from . import Scyan


def _optional_show(f):
    """Decorator that shows a matplotlib figure if the provided 'show' argument is True"""

    def wrapper(*args, **kwargs):
        f(*args, **kwargs)
        if kwargs.get("show", True):
            plt.show()

    return wrapper


@_optional_show
def marker_matrix_reconstruction(
    scyan: Scyan, show_diff: bool = False, show: bool = True
):
    predictions = scyan.predict(key_added=None)
    predictions.name = "Population"
    h = scyan().detach().numpy()
    df = pd.concat(
        [predictions, pd.DataFrame(h, columns=scyan.marker_pop_matrix.columns)], axis=1
    )
    df = df.groupby("Population").median()
    heatmap = df.loc[scyan.marker_pop_matrix.index]
    if show_diff:
        heatmap -= scyan.marker_pop_matrix
    sns.heatmap(heatmap, cmap="coolwarm", center=0)
    plt.title("Marker matrix approximation by the model embedding space")


@_optional_show
def kde_per_population(
    adata: AnnData,
    cell_type_key: str,
    markers: Union[List[str], None] = None,
    ncols: int = 4,
    hue_name: str = "Population",
    var_name: str = "Marker",
    value_name: str = "Expression",
    show: bool = True,
):
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
    grid.map(sns.kdeplot, value_name)
    grid.add_legend()


@_optional_show
def probs_per_marker(scyan: Scyan, where, prob_name: str = "Prob", show: bool = True):
    h = scyan.module(scyan.x[where], scyan.covariates[where])[0]
    normal = torch.distributions.normal.Normal(0, 1)

    _probs_per_marker = normal.log_prob(
        (h[:, None, :] - scyan.module.rho_inferred[None, ...]) / scyan.module.std_diags
    ) - torch.log(scyan.module.std_diags)

    df_probs = pd.DataFrame(
        _probs_per_marker.mean(dim=0).detach().numpy(),
        columns=scyan.adata.var_names,
        index=scyan.marker_pop_matrix.index,
    )
    df_probs = df_probs.reindex(
        df_probs.mean().sort_values(ascending=False).index, axis=1
    )
    means = df_probs.mean(axis=1)
    means = means / means.min() * df_probs.values.min()
    df_probs.insert(0, prob_name, means)
    df_probs.insert(1, " ", np.nan)
    df_probs.sort_values(by=prob_name, inplace=True, ascending=False)
    sns.heatmap(df_probs, cmap="magma")
    plt.title("Log probabilities per marker for each population")


@_optional_show
def latent_expressions(model, where, show=True):
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
