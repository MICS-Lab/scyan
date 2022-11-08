import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from umap import UMAP

if TYPE_CHECKING:
    from . import Scyan

from ..utils import _check_is_processed, _get_subset_indices

log = logging.getLogger(__name__)


def subcluster(
    model: "Scyan",
    resolution: float = 1,
    size_ratio_th: float = 0.02,
    min_cells_th: int = 200,
    population: Optional[str] = None,
    obs_key: str = "scyan_pop",
    subcluster_key: str = "subcluster_index",
    umap_display_key: str = "leiden_subcluster",
) -> None:
    """Create sub-clusters among the populations predicted by Scyan. Some population may not be divided.
    !!! info
        After having run this method, you can analyze the results with:
        ```python
        import scanpy as sc
        scyan.plot.umap(adata, color="leiden_subcluster") # Visualize the sub clusters
        scyan.plot.subcluster(model) # Scyan latent space on each sub cluster
        ```

    Args:
        model: Scyan model
        resolution: Resolution used for leiden clustering. Higher resolution leads to more clusters.
        size_ratio_th: (Only used if `population` is `None`): Minimum ratio of cells to be considered as a significant cluster (compared to the parent cluster).
        min_cells_th: (Only used if `population` is `None`): Minimum number of cells to be considered as a significant cluster.
        population: Name of the population to target. By default, run population discovery on all populations.
        obs_key: Key to look for population in `adata.obs`. By default, uses the model predictions, but you can also choose a population level (if any).
        subcluster_key: Key added to `adata.obs` to indicate the index of the subcluster.
        umap_display_key: Key added to `adata.obs` to plot the sub-clusters on a UMAP.
    """
    adata = model.adata
    key_added = f"scyan_leiden_{resolution}"
    if population is not None:
        key_added = f"{key_added}_{population}"

    if key_added in adata.obs:
        log.info(
            "Found leiden labels with the same resolution. Skipping leiden clustering."
        )
    elif population is None:
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added)
    else:
        condition = adata.obs[obs_key] == population
        adata_sub = adata[condition].copy()
        sc.pp.neighbors(adata_sub)
        sc.tl.leiden(adata_sub, resolution=resolution)
        adata.obs[key_added] = np.nan
        adata.obs.loc[condition, key_added] = adata_sub.obs.leiden
        adata.obs[subcluster_key] = adata.obs[key_added]

    if population is None:
        adata.obs[subcluster_key] = ""
        for pop in adata.obs[obs_key].cat.categories:
            condition = adata.obs[obs_key] == pop

            labels = adata[condition].obs[key_added]
            ratios = labels.value_counts(normalize=True)
            ratios = ratios[labels.value_counts() > min_cells_th]

            if (ratios > size_ratio_th).sum() < 2:
                adata.obs.loc[condition, subcluster_key] = np.nan
                continue

            rename_dict = {
                k: i for i, (k, v) in enumerate(ratios.items()) if v > size_ratio_th
            }
            adata.obs.loc[condition, subcluster_key] = [
                rename_dict.get(l, np.nan) for l in labels
            ]

    series = adata.obs[subcluster_key]
    adata.obs[umap_display_key] = pd.Categorical(
        np.where(
            series.isna(),
            np.nan,
            adata.obs[obs_key].astype(str) + " -> " + series.astype(str),
        )
    )


def umap(
    adata: AnnData,
    markers: Optional[List[str]] = None,
    obsm: Optional[str] = None,
    n_cells: Optional[int] = 500000,
    min_dist: float = 0.5,
    obsm_key: str = "X_umap",
    filter: Optional[Tuple] = None,
    **umap_kwargs: int,
) -> UMAP:
    """Run a [UMAP](https://umap-learn.readthedocs.io/en/latest/) on a specific set of markers (or all markers by default). It can be useful to show differences that are due to some markers of interest, instead of using the whole panel.

    !!! info

        This function returns a UMAP reducer. You can reuse it with `reducer.transform(...)` or save it with [scyan.data.add][].

    !!! note

        To actually plot the UMAP, use [scyan.plot.umap][].

    Args:
        adata: An `anndata` object.
        markers: List marker names. By default, use all the panel markers, i.e., `adata.var_names`.
        obsm: Name of the obsm to consider to train the UMAP. By default, uses `adata.X`.
        n_cells: Number of cells to be considered for the UMAP (to accelerate it when $N$ is very high). If `None`, consider all cells.
        min_dist: Min dist UMAP parameter.
        obsm_key: Key for `adata.obsm` to add the embedding.
        filter: Optional tuple `(obs_key, value)` used to train the UMAP on a set of cells that satisfies a constraint. `obs_key` is the key of `adata.obs` to consider, and `value` the value the cells need to have.
        **umap_kwargs: Optional kwargs to provide to the `UMAP` initialization.

    Returns:
        UMAP reducer.
    """
    reducer = UMAP(min_dist=min_dist, **umap_kwargs)

    if markers is None:
        markers = adata.var_names

    adata.obsm[obsm_key] = np.zeros((adata.n_obs, 2))
    indices = _get_subset_indices(adata, n_cells)
    adata_view = adata[indices, markers]
    X = adata_view.X if obsm is None else adata_view.obsm[obsm]

    _check_is_processed(X)

    if n_cells is not None:
        adata.obs["has_umap"] = np.in1d(np.arange(adata.n_obs), indices)

    log.info("Fitting UMAP...")
    if filter is None:
        embedding = reducer.fit_transform(X)
    else:
        obs_key, value = filter
        reducer.fit(X[adata[indices].obs[obs_key] == value])
        log.info("Transforming...")
        embedding = reducer.transform(X)

    adata.obsm[obsm_key][indices] = embedding

    return reducer
