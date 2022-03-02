import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from typing import List, Union
from anndata import AnnData

from . import Scyan


def _optional_show(f):
    """plt.show() only if show == True"""

    def wrapper(*args, **kwargs):
        f(*args, **kwargs)
        if kwargs.get("show", True):
            plt.show()

    return wrapper


@_optional_show
def marker_matrix_reconstruction(scyan: Scyan, show: bool = True, show_diff=False):
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


def kde_per_population(
    adata: AnnData,
    cell_type_key: str,
    markers: Union[List[str], None] = None,
    show: bool = True,
):
    df = adata.to_df()
    df["pop"] = adata.obs[cell_type_key]

    markers = adata.var_names if markers is None else markers
    for marker in markers:
        sns.displot(df, x=marker, hue="pop", kind="kde")
        plt.title(f"KDE of marker {marker} per population")
        plt.show()


@_optional_show
def probs_per_marker(scyan: Scyan, where, show: bool = True):
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
    df_probs["Mean"] = df_probs.mean(axis=1)
    sns.heatmap(df_probs, cmap="magma")
    plt.title("Log probabilities per marker for each population")
