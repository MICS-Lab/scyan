from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import norm

from .. import Scyan
from ..utils import _get_subset_indices, _requires_fit
from .utils import check_population, plot_decorator


@torch.no_grad()
@plot_decorator
@_requires_fit
def pops_expressions(
    model: Scyan,
    latent: bool = True,
    obs_key: str = "scyan_pop",
    n_cells: Optional[int] = 200000,
    vmax: float = 1.2,
    vmin: float = -1.2,
    cmap: Optional[str] = None,
    show: bool = True,
):
    """Heatmap that shows (latent or standardized) cell expressions for all populations.

    !!! note
        If using the latent space, it will only show the marker you provided to Scyan. Else, it shows every marker of the panel.

    Args:
        model: Scyan model.
        latent: If `True`, displays Scyan's latent expressions, else just the standardized expressions.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        n_cells: Number of cells to be considered for the heatmap (to accelerate it when $N$ is very high). If `None`, consider all cells.
        vmax: Maximum value on the heatmap.
        vmax: Minimum value on the heatmap.
        cmap: Colormap name. By default, uses `"coolwarm"` if `latent`, else `"viridis"`.
        show: Whether or not to display the figure.
    """
    indices = _get_subset_indices(model.adata, n_cells)

    x = model(indices).cpu().numpy() if latent else model.adata[indices].X
    columns = model.var_names if latent else model.adata.var_names

    df = pd.DataFrame(x, columns=columns)
    df["Population"] = model.adata[indices].obs[obs_key].values

    if cmap is None:
        cmap = "coolwarm" if latent else "viridis"

    sns.heatmap(df.groupby("Population").mean(), vmax=vmax, vmin=vmin, cmap=cmap)
    plt.title(
        f"{'Latent' if latent else 'Standardized'} expressions grouped by {obs_key}"
    )


@torch.no_grad()
@plot_decorator
@_requires_fit
def boxplot_expressions(
    model: Scyan,
    marker: str,
    latent: bool = False,
    obs_key: str = "scyan_pop",
    ascending: bool = False,
    figsize: tuple = (10, 5),
    fliersize: float = 0.5,
):
    """Boxplot of expressions per population (for one marker).

    Args:
        model: Scyan model.
        marker: Name of the marker whose expressions are to be displayed.
        latent: If `True`, displays Scyan's latent expressions, else just the standardized expressions.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        ascending: Whether to sort populations by ascending mean expressions.
        figsize: Figure size. Defaults to (10, 5).
        fliersize: Outlier dot size (see seaborn boxplot).
    """
    if latent:
        assert (
            marker in model.var_names
        ), f"Marker {marker} was not used during the model training, impossible to use 'latent=True'."

        u = model()
        y = u[:, model.var_names.get_loc(marker)].cpu().numpy()
    else:
        y = model.adata.obs_vector(marker)

    df = pd.DataFrame({obs_key: model.adata.obs[obs_key], "y": y})
    order = df.groupby(obs_key).mean().sort_values("y", ascending=ascending).index

    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=obs_key, y="y", fliersize=fliersize, order=order)
    plt.xticks(rotation=90)
    plt.ylabel(f"{marker} {'latent' if latent else 'standardized'} expression")


@torch.no_grad()
@plot_decorator
@_requires_fit
@check_population(one=True)
def pop_expressions(
    model: Scyan,
    population: str,
    obs_key: str = "scyan_pop",
    max_value: float = 1.5,
    num_pieces: int = 100,
    radius: float = 0.05,
    show: bool = True,
):
    """Plot latent cell expressions for one population. Contrary to `scyan.plot.pops_expressions`, in displays expressions on a vertical bar, from `Neg` to `Pos`.

    Args:
        model: Scyan model.
        population: Name of one population to interpret. To be valid, the population name has to be in `adata.obs[obs_key]`.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        max_value: Maximum absolute latent value.
        num_pieces: Number of pieces to display the colorbar.
        radius: Radius used to chunk the colorbar. Increase this value if multiple names overlap.
        show: Whether or not to display the figure.
    """
    condition = model.adata.obs[obs_key] == population
    u_mean = model(condition).mean(dim=0)
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
