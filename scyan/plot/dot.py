from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData

from .. import Scyan
from ..utils import _subset
from .utils import check_population, get_palette_others, optional_show, select_markers


@optional_show
@check_population(return_list=True)
def scatter(
    model: Scyan,
    populations: Union[str, List[str]],
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
        populations: One population or a list of population to be colored. To be valid, a population name have to be in `adata.obs[obs_key]`.
        markers: List of markers to plot. If `None`, the list is chosen automatically.
        n_markers: Number of markers to choose automatically if `markers is None`.
        obs_key: Key to look for populations in `adata.obs`. By default, uses the model predictions.
        max_obs: Maximum number of cells per population to be displayed.
        s: Dot marker size.
        show: Whether or not to display the plot.
    """
    adata = model.adata
    markers = select_markers(model, markers, n_markers, obs_key, populations)

    data = adata[:, markers].to_df()
    keys = adata.obs[obs_key].astype(str)
    data["Population"] = np.where(~np.isin(keys, populations), "Others", keys)

    pops = list(populations) + ["Others"]
    if max_obs is not None:
        groups = data.groupby("Population").groups
        data = data.loc[[i for pop in pops[::-1] for i in _subset(groups[pop], max_obs)]]

    g = sns.PairGrid(data, hue="Population", corner=True)

    palette = get_palette_others(data, "Population")

    g.map_offdiag(sns.scatterplot, s=s, palette=palette, hue_order=pops)
    g.map_diag(sns.histplot, palette=palette)
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
    if "has_umap" in adata.obs:
        adata = adata[adata.obs.has_umap]

    sc.pl.umap(adata, color=color, vmax=vmax, vmin=vmin, **scanpy_kwargs)


def _get_pop_to_group(model: Scyan) -> dict:
    df = model.marker_pop_matrix

    assert isinstance(
        df.index, pd.MultiIndex
    ), "To plot main populations, you need to provide a 'Group name' to your CSV file (the second column)."

    return {pop: group for pop, group in df.index}


def all_groups(model: Scyan, obs_key: str = "scyan_pop", **scanpy_kwargs: int) -> None:
    """Plot all main groups on a UMAP (according to the populations groups provided in the knowledge table).

    Args:
        model: Scyan model.
        obs_key: Key of `adata.obs` to access the model predictions.
    """
    adata = model.adata

    assert (
        "scyan_pop" in adata.obs.columns
    ), "Found no model predictions. Have you run 'model.predict()' first?"

    pop_to_group = _get_pop_to_group(model)

    adata.obs["scyan_group"] = pd.Categorical(
        [pop_to_group[pop] for pop in adata.obs[obs_key]]
    )

    umap(adata, color="scyan_group", **scanpy_kwargs)


def one_group(
    model: Scyan, group_name: str, obs_key: str = "scyan_pop", **scanpy_kwargs: int
) -> None:
    """Plot all subpopulations of a group on a UMAP (according to the populations groups provided in the knowledge table).

    Args:
        model: Scyan model.
        group_name: The group to look at.
        obs_key: Key of `adata.obs` to access the model predictions.
    """
    adata = model.adata

    pop_to_group = _get_pop_to_group(model)

    assert (
        group_name in pop_to_group.values()
    ), f"Invalid group name '{group_name}'. It has to be one of: {', '.join(pop_to_group.values())}."

    adata.obs["scyan_one_group"] = pd.Categorical(
        [
            pop if pop_to_group[pop] == group_name else "Others"
            for pop in adata.obs[obs_key]
        ]
    )
    palette = get_palette_others(adata.obs, "scyan_one_group")
    umap(
        adata,
        color="scyan_one_group",
        palette=palette,
        title=f"Among {group_name}",
        **scanpy_kwargs,
    )
