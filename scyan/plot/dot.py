from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData

from .. import Scyan
from ..utils import _get_subset_indices, _has_umap, _subset
from .utils import check_population, get_palette_others, plot_decorator, select_markers


@plot_decorator(adata=True)
@check_population(return_list=True)
def scatter(
    adata: AnnData,
    population: Union[str, List[str], None],
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
        adata: An `anndata` object.
        population: One population, or a list of population to be colored, or `None`. If not `None`, the population name(s) has to be in `adata.obs[obs_key]`.
        markers: List of markers to plot. If `None`, the list is chosen automatically.
        n_markers: Number of markers to choose automatically if `markers is None`.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        max_obs: Maximum number of cells per population to be displayed. If population is None, then this number is multiplied by 10.
        s: Dot marker size.
        show: Whether or not to display the figure.
    """
    markers = select_markers(adata, markers, n_markers, obs_key, population)

    if population is None:
        indices = _get_subset_indices(adata.n_obs, max_obs * 10)
        data = adata[indices, markers].to_df()
        g = sns.PairGrid(data, corner=True)
        g.map_offdiag(sns.scatterplot, s=s)
        g.map_diag(sns.histplot)
        g.add_legend()
        return

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
    color: Union[str, List[str]] = None,
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

    has_umap = _has_umap(adata)
    if not has_umap.all():
        adata = adata[has_umap]

    if color is None:
        return sc.pl.umap(adata, **scanpy_kwargs)

    return sc.pl.umap(adata, color=color, vmax=vmax, vmin=vmin, **scanpy_kwargs)


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
    table = model.table

    assert isinstance(
        table.index, pd.MultiIndex
    ), "To use this function, you need a MultiIndex DataFrame, see: https://mics-lab.github.io/scyan/tutorials/advanced/#hierarchical-population-display"

    level_names = table.index.names[1:]
    assert (
        level_name in level_names
    ), f"Level '{level_name}' unknown. Choose one of: {level_names}"

    base_pops = table.index.get_level_values(0)
    group_pops = table.index.get_level_values(level_name)
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
