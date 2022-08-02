import logging
from typing import TYPE_CHECKING, List, Optional

import flowutils
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from anndata import AnnData
from umap import UMAP

if TYPE_CHECKING:
    from . import Scyan

log = logging.getLogger(__name__)


def auto_logicle_transform(adata: AnnData, q: float = 0.05, m: float = 4.5) -> None:
    """[Logicle transformation](https://pubmed.ncbi.nlm.nih.gov/16604519/), implementation from Charles-Antoine Dutertre.
    We recommend it for flow cytometry or spectral flow cytometry data.

    Args:
        adata: An `anndata` object.
        q: See logicle article. Defaults to 0.05.
        m: See logicle article. Defaults to 4.5.
    """
    for marker in adata.var_names:
        column = adata[:, marker].X.toarray().flatten()

        w = 0
        t = column.max()
        negative_values = column[column < 0]

        if negative_values.size:
            threshold = np.quantile(negative_values, 0.25) - 1.5 * scipy.stats.iqr(
                negative_values
            )
            negative_values = negative_values[negative_values >= threshold]

            if negative_values.size:
                r = 1e-8 + np.quantile(negative_values, q)
                if 10**m * abs(r) > t:
                    w = (m - np.log10(t / abs(r))) / 2

        if not w or w > 2:
            log.warning(
                f"Auto logicle transformation failed for {marker}. Using default logicle."
            )
            w, t = 1, 5e5

        column = flowutils.transforms.logicle(column, None, t=t, m=m, w=w)
        adata[:, marker] = column.clip(np.quantile(column, 1e-5))


def asinh_transform(adata: AnnData, translation: float = 1, cofactor: float = 5):
    """Asinh transformation for cell-expressions: $asinh((x - translation)/cofactor)$.

    Args:
        adata: An `anndata` object.
        translation: Constant substracted to the marker expression before division by the cofactor.
        cofactor: Scaling factor before computing the asinh.
    """
    adata.X = np.arcsinh((adata.X - translation) / cofactor)


def scale(adata: AnnData, max_value: float = 10, **kwargs: int):
    """Standardise data using scanpy.

    Args:
        adata: An `anndata` object.
        max_value: Clip to this value after scaling. Defaults to 10.
        **kwargs: Optional `sc.pp.scale` kwargs.
    """
    sc.pp.scale(adata, max_value=max_value, **kwargs)


def subcluster(
    model: "Scyan",
    resolution: float = 1,
    size_ratio_th: float = 0.02,
    min_cells_th: int = 200,
    obs_key: str = "scyan_pop",
    subcluster_key: str = "subcluster_index",
    umap_display_key: str = "leiden_subcluster",
) -> None:
    """Create sub-clusters among the populations predicted by Scyan. Some population may not be divided.
    !!! info
        After having run this method, you can analyze the results with:
        ```python
        import scanpy as sc
        sc.pl.umap(adata, color="leiden_subcluster") # Visualize the sub clusters
        scyan.plot.subclusters(model) # Scyan latent space on each sub cluster
        ```

    Args:
        model: Scyan model
        resolution: Resolution used for leiden clustering. Higher resolution leads to more clusters.
        size_ratio_th: Minimum ratio of cells to be considered as a significant cluster (compared to the parent cluster).
        min_cells_th: Minimum number of cells to be considered as a significant cluster.
        obs_key: Key to look for population in `adata.obs`. By default, uses the model predictions.
        subcluster_key: Key added to `adata.obs` to indicate the index of the subcluster.
        umap_display_key: Key added to `adata.obs` to plot the sub-clusters on a UMAP.
    """
    adata = model.adata

    if (
        "leiden" in adata.obs
        and adata.uns.get("leiden", {}).get("params", {}).get("resolution") == resolution
    ):
        log.info(
            "Found leiden labels with the same resolution. Skipping leiden clustering."
        )
    else:
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=resolution)

    adata.obs[subcluster_key] = ""
    for pop in adata.obs[obs_key].cat.categories:
        condition = adata.obs[obs_key] == pop

        labels = adata[condition].obs.leiden
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
    markers: Optional[List[str]],
    obsm_key: str = "X_umap",
    min_dist: float = 0.5,
    **umap_kwargs: int,
) -> UMAP:
    """Run a [UMAP](https://umap-learn.readthedocs.io/en/latest/) on a specific set of markers (or all markers by default). It can be useful to show differences that are due to some markers of interest, instead of using the whole panel.

    !!! info

        This function returns a UMAP reducer. You can reuse it with `reducer.transform(...)` or save it with [`scyan.data.add`](../add).

    !!! note

        To actually plot the UMAP, use [`sc.pl.umap`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pl.umap.html).

    Args:
        adata: An `anndata` object.
        markers: List marker names. By default, use all the panel markers.
        obsm_key: Key for `adata.obsm` to add the embedding. Defaults to "X_umap", i.e. you will be able to display the UMAP with `scanpy`.
        min_dist: Min dist UMAP parameter. Defaults to 0.5.
        **umap_kwargs: Optional kwargs to provide to the `UMAP` initialization.

    Returns:
        UMAP reducer.
    """
    reducer = UMAP(min_dist=min_dist, **umap_kwargs)

    if markers is None:
        markers = adata.var_names

    embedding = reducer.fit_transform(adata[:, markers].X)
    adata.obsm[obsm_key] = embedding

    return reducer
