import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from umap import UMAP

from ..utils import _check_is_processed, _get_subset_indices, _has_umap

log = logging.getLogger(__name__)


def leiden(
    adata: AnnData,
    resolution: float = 1,
    key_added: str = "leiden",
    n_neighbors: int = 15,
) -> None:
    """Leiden clustering

    Args:
        adata: AnnData object.
        resolution: Resolution of the clustering.
        key_added: Name of the key of adata.obs where clusters will be saved.
        n_neighbors: Number of neighbors.
    """
    try:
        import leidenalg
    except:
        raise ImportError(
            """To run leiden, you need to have 'leidenalg' installed. You can install the population discovery extra with "pip install 'scyan[discovery]'", or directly install leidenalg with "conda install -c conda-forge leidenalg"."""
        )

    import igraph as ig
    from sklearn.neighbors import kneighbors_graph

    if not "knn_graph" in adata.obsp:
        adata.obsp["knn_graph"] = kneighbors_graph(
            adata.X, n_neighbors=n_neighbors, metric="euclidean", include_self=False
        )

    # TODO (improvement): add weights according to euclidean distance
    graph = ig.Graph.Weighted_Adjacency(adata.obsp["knn_graph"], mode="DIRECTED")

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
    )
    adata.obs[key_added] = pd.Categorical([str(x) for x in partition.membership])


def subcluster(
    adata: AnnData,
    population: str,
    markers: Optional[List[str]] = None,
    key: str = "scyan_pop",
    resolution: float = 0.2,
    size_ratio_th: float = 0.02,
    min_cells_th: int = 200,
    n_cells: int = 100_000,
) -> None:
    """Create sub-clusters among a given populations, and filters small clusters according to (i) a minimum number of cells and (ii) a minimum ratio of cells.
    !!! info
        After having run this method, you can analyze the results with [scyan.plot.umap][] and [scyan.plot.pops_expressions][].

    Args:
        adata: An `AnnData` object.
        population: Name of the population to target (one of `adata.obs[key]`).
        markers: Optional list of markers used to create subclusters. By default, uses the complete panel.
        key: Key to look for population in `adata.obs`. By default, uses the model predictions, but you can also choose a population level (if any), or other observations.
        resolution: Resolution used for leiden clustering. Higher resolution leads to more clusters.
        size_ratio_th: (Only used if `population` is `None`): Minimum ratio of cells to be considered as a significant cluster (compared to the parent cluster).
        min_cells_th: (Only used if `population` is `None`): Minimum number of cells to be considered as a significant cluster.
        n_cells: Number of cells to be considered for the subclustering (to accelerate it when $N$ is very high). If `None`, consider all cells.
    """
    leiden_key = f"leiden_{resolution}_{population}"
    subcluster_key = f"scyan_subcluster_{population}"
    condition = adata.obs[key] == population
    markers = list(adata.var_names if markers is None else markers)

    if leiden_key in adata.obs and adata.uns.get(leiden_key, []) == markers:
        log.info(
            "Found leiden labels with the same resolution. Skipping leiden clustering."
        )
        indices = np.where(~adata.obs[leiden_key].isna())[0]
        adata_sub = adata[indices, markers].copy()
    else:
        has_umap = _has_umap(adata)
        if has_umap.all() or condition.sum() <= n_cells:
            indices = _get_subset_indices(condition.sum(), n_cells)
            indices = np.where(condition)[0][indices]
        else:
            indices = _get_subset_indices((condition & has_umap).sum(), n_cells)
            indices = np.where(condition & has_umap)[0][indices]

            k = len(indices)
            if k < n_cells:
                indices2 = _get_subset_indices((condition & ~has_umap).sum(), n_cells - k)
                indices2 = np.where(condition & ~has_umap)[0][indices2]
                indices = np.concatenate([indices, indices2])

        adata_sub = adata[indices, markers].copy()

        leiden(adata_sub, resolution, leiden_key)

    series = pd.Series(index=np.arange(adata.n_obs), dtype=str)
    series[indices] = adata_sub.obs[leiden_key].values
    adata.obs[leiden_key] = series.values
    adata.obs[leiden_key] = adata.obs[leiden_key].astype("category")

    counts = adata_sub.obs[leiden_key].value_counts()
    remove = counts < max(counts.sum() * size_ratio_th, min_cells_th)

    assert (
        not remove.all()
    ), "All subclusters where filtered. Consider updating size_ratio_th and/or min_cells_th."

    adata_sub.obs.loc[
        np.isin(adata_sub.obs[leiden_key], remove[remove].index), leiden_key
    ] = np.nan

    series = pd.Series(index=np.arange(adata.n_obs), dtype=str)
    series[indices] = adata_sub.obs[leiden_key].values
    adata.obs[subcluster_key] = series.values
    adata.obs[subcluster_key] = adata.obs[subcluster_key].astype("category")

    adata.uns[leiden_key] = markers
    log.info(
        f"Subclusters created, you can now use:\n   - scyan.plot.umap(adata, color='{subcluster_key}') to show the clusters\n   - scyan.plot.pops_expressions(model, key='{subcluster_key}') to plot their expressions"
    )


def umap(
    adata: AnnData,
    markers: Optional[List[str]] = None,
    obsm: Optional[str] = None,
    n_cells: Optional[int] = 200_000,
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
        adata: An `AnnData` object.
        markers: List marker names. By default, use all the panel markers, i.e., `adata.var_names`.
        obsm: Name of the obsm to consider to train the UMAP. By default, uses `adata.X`.
        n_cells: Number of cells to be considered for the UMAP (to accelerate it when $N$ is very high). If `None`, consider all cells.
        min_dist: Min dist UMAP parameter.
        obsm_key: Key for `adata.obsm` to add the embedding.
        filter: Optional tuple `(key, value)` used to train the UMAP on a set of cells that satisfies a constraint. `key` is the key of `adata.obs` to consider, and `value` the value the cells need to have.
        **umap_kwargs: Optional kwargs to provide to the `UMAP` initialization.

    Returns:
        UMAP reducer.
    """
    reducer = UMAP(min_dist=min_dist, **umap_kwargs)

    if markers is None:
        markers = adata.var_names

    adata.obsm[obsm_key] = np.zeros((adata.n_obs, 2))
    indices = _get_subset_indices(adata.n_obs, n_cells)
    adata_view = adata[indices, markers]
    X = adata_view.X if obsm is None else adata_view.obsm[obsm]

    _check_is_processed(X)

    log.info("Fitting UMAP...")
    if filter is None:
        embedding = reducer.fit_transform(X)
    else:
        key, value = filter
        reducer.fit(X[adata[indices].obs[key] == value])
        log.info("Transforming...")
        embedding = reducer.transform(X)

    adata.obsm[obsm_key][indices] = embedding

    return reducer
