from typing import List, Optional, Union

import numpy as np
import seaborn as sns

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

    pops = populations + ["Others"]
    if max_obs is not None:
        groups = data.groupby("Population").groups
        data = data.loc[[i for pop in pops[::-1] for i in _subset(groups[pop], max_obs)]]

    g = sns.PairGrid(data, hue="Population", corner=True)

    palette = get_palette_others(data, "Population")

    g.map_offdiag(sns.scatterplot, s=s, palette=palette, hue_order=pops)
    g.map_diag(sns.histplot, palette=palette)
    g.add_legend()
