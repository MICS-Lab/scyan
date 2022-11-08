from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData

from .. import Scyan
from ..utils import _subset
from .utils import check_population, get_palette_others, plot_decorator, select_markers


@plot_decorator
@check_population(return_list=True)
def scatter(
    model: Scyan,
    population: Union[str, List[str]],
    markers: Optional[List[str]] = None,
    n_markers: Optional[int] = 3,
    obs_key: str = "scyan_pop",
    max_obs: int = 2000,
    s: float = 1.0,
    show: bool = True,
) -> None:
    """Display marker expressions on 2D scatter plots with colors per population.
    One scatter plot is displayed for each pair of markers.

    Args:
        model: Scyan model
        population: One population or a list of population to be colored. To be valid, a population name have to be in `adata.obs[obs_key]`.
        markers: List of markers to plot. If `None`, the list is chosen automatically.
        n_markers: Number of markers to choose automatically if `markers is None`.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        max_obs: Maximum number of cells per population to be displayed.
        s: Dot marker size.
        show: Whether or not to display the figure.
    """
    adata = model.adata
    markers = select_markers(model, markers, n_markers, obs_key, population)

    data = adata[:, markers].to_df()
    keys = adata.obs[obs_key].astype(str)
    data["Population"] = np.where(~np.isin(keys, population), "Others", keys)

    pops = list(population) + ["Others"]
    if max_obs is not None:
        groups = data.groupby("Population").groups
        data = data.loc[[i for pop in pops[::-1] for i in _subset(groups[pop], max_obs)]]

    palette = get_palette_others(data, "Population")

    g = sns.PairGrid(data, hue="Population", corner=True, palette=palette, hue_order=pops)
    g.map_offdiag(sns.scatterplot, s=s)
    g.map_diag(sns.histplot)
    g.add_legend()


def umap(
    adata: AnnData,
    color: Union[str, List[str]],
    vmax: Union[str, float] = "p95",
    vmin: Union[str, float] = "p05",
    **scanpy_kwargs: int,
):
    """Plot a UMAP using scanpy.

    !!! note
        If you trained your UMAP with [scyan.tools.umap][] on a subset of cells, it will only display the desired subset of cells.

    Args:
        adata: An `anndata` object.
        color: Marker(s) or `obs` name(s) to color. It can be either just one string, or a list (it will plot one UMAP per element in the list).
        vmax: `scanpy.pl.umap` vmax argument.
        vmin: `scanpy.pl.umap` vmin argument.
        **scanpy_kwargs: Optional kwargs provided to `scanpy.pl.umap`.
    """
    assert isinstance(
        adata, AnnData
    ), f"umap first argument has to be an AnnData object. Received type {type(adata)}."

    assert (
        "X_umap" in adata.obsm_keys()
    ), "Before plotting data, UMAP coordinates need to be computed using 'scyan.tools.umap(...)' (see https://mics-lab.github.io/scyan/api/representation/#scyan.tools.umap)"

    if "has_umap" in adata.obs:
        adata = adata[adata.obs.has_umap]

    sc.pl.umap(adata, color=color, vmax=vmax, vmin=vmin, **scanpy_kwargs)


def pop_level(
    model: Scyan,
    group_name: str,
    level_name: str = "level",
    obs_key: str = "scyan_pop",
    **scanpy_kwargs: int,
) -> None:
    """Plot all subpopulations of a group at a certain level on a UMAP (according to the populations levels provided in the knowledge table).

    Args:
        model: Scyan model.
        group_name: The group to look at among the populations of the selected level.
        level_name: Name of the column of the knowledge table containing the names of the grouped populations.
        obs_key: Key of `adata.obs` to access the model predictions.
    """
    adata = model.adata
    mpm = model.marker_pop_matrix

    assert isinstance(
        mpm.index, pd.MultiIndex
    ), "To use this function, you need a MultiIndex DataFrame, see: https://mics-lab.github.io/scyan/tutorials/advanced/#hierarchical-population-display"

    level_names = mpm.index.names[1:]
    assert (
        level_name in level_names
    ), f"Level '{level_name}' unknown. Choose one of: {level_names}"

    base_pops = mpm.index.get_level_values(0)
    group_pops = mpm.index.get_level_values(level_name)
    assert (
        group_name in group_pops
    ), f"Invalid group name '{group_name}'. It has to be one of: {', '.join(group_pops)}."

    valid_populations = [
        pop for pop, group in zip(base_pops, group_pops) if group == group_name
    ]
    key_name = f"{obs_key}_one_level"
    adata.obs[key_name] = pd.Categorical(
        [pop if pop in valid_populations else np.nan for pop in adata.obs[obs_key]]
    )
    umap(
        adata,
        color=key_name,
        title=f"Among {group_name}",
        na_in_legend=False,
        **scanpy_kwargs,
    )
