import seaborn as sns
from typing import List, Union
import numpy as np

from .. import Scyan
from ..utils import _subset
from .utils import optional_show, check_population, get_palette_others, select_markers


@optional_show
@check_population(return_list=True)
def scatter(
    model: Scyan,
    populations: Union[str, List[str]],
    markers: Union[List[str], None] = None,
    n_markers: Union[int, None] = 3,
    obs_key: str = "scyan_pop",
    max_obs: int = 2000,
    s: float = 1.0,
    show: bool = True,
):
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
