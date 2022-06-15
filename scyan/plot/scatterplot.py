import seaborn as sns
from typing import List, Union
import scanpy as sc
import numpy as np

from .. import Scyan
from .utils import optional_show, check_population, get_palette_others


@optional_show
@check_population(return_list=True)
def scatter(
    model: Scyan,
    populations: Union[str, List[str]],
    markers: List[str],
    obs_key: str = "scyan_pop",
    n_obs: int = 5000,
    s: float = 1.0,
    show: bool = True,
):
    if n_obs is not None:
        adata = sc.pp.subsample(model.adata, n_obs=n_obs, copy=True)
    else:
        adata = model.adata

    data = adata[:, markers].to_df()
    keys = adata.obs[obs_key].astype(str)
    data["Population"] = np.where(~np.isin(keys, populations), "Others", keys)

    g = sns.PairGrid(data, hue="Population", corner=True)

    palette = get_palette_others(data, "Population")

    g.map_offdiag(sns.scatterplot, s=s, palette=palette)
    g.map_diag(sns.histplot, palette=palette)
    g.add_legend()
