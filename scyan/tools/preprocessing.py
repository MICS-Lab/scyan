import logging
from typing import Optional

import flowutils
import numpy as np
import scipy
from anndata import AnnData

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


def asinh_transform(adata: AnnData, translation: float = 1, cofactor: float = 5) -> None:
    """Asinh transformation for cell-expressions: $asinh((x - translation)/cofactor)$.

    Args:
        adata: An `anndata` object.
        translation: Constant substracted to the marker expression before division by the cofactor.
        cofactor: Scaling factor before computing the asinh.
    """
    adata.X = np.arcsinh((adata.X - translation) / cofactor)


def scale(adata: AnnData, max_value: float = 10) -> None:
    """Standardise data.

    Args:
        adata: An `anndata` object.
        max_value: Clip to this value after scaling.
    """
    means = adata.X.mean(axis=0)
    adata.X = adata.X - means

    stds = adata.X.std(axis=0)
    adata.X = (adata.X / stds).clip(-max_value, max_value)

    adata.uns["scyan_scaling"] = {"means": means, "stds": stds}


def unscale(adata: AnnData, layer: Optional[str] = None) -> np.ndarray:
    """Reverse standardisation. It requires to have run [scyan.tools.scale](../scale) before.

    Args:
        adata: An `anndata` object.
        layer: Name of the anndata layer to consider. By default, use `adata.X`.

    Returns:
        Unscaled numpy array of shape $(N, M)$.
    """
    assert (
        "scyan_scaling" in adata.uns
    ), "It seems you haven't run 'scyan.tools.scale' before."

    X = adata.X if layer is None else adata.layers[layer]

    return adata.uns["scyan_scaling"]["means"] + X * adata.uns["scyan_scaling"]["stds"]
