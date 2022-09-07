import logging
from typing import List, Optional

import flowutils
import numpy as np
import scipy
from anndata import AnnData

log = logging.getLogger(__name__)


def auto_logicle_transform(
    adata: AnnData, q: float = 0.05, m: float = 4.5, quantile_clip: Optional[float] = 1e-5
) -> None:
    """[Logicle transformation](https://pubmed.ncbi.nlm.nih.gov/16604519/), implementation from Charles-Antoine Dutertre.
    We recommend it for flow cytometry or spectral flow cytometry data.

    Args:
        adata: An `anndata` object.
        q: See logicle article. Defaults to 0.05.
        m: See logicle article. Defaults to 4.5.
    """
    adata.uns["scyan_logicle"] = {}

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
        adata.uns["scyan_logicle"][marker] = [t, m, w]

        if quantile_clip is None:
            adata[:, marker] = column
        else:
            adata[:, marker] = column.clip(np.quantile(column, quantile_clip))


def _logicle_inverse_one(adata: AnnData, obsm: Optional[str], marker: str) -> np.ndarray:
    column = adata[:, marker].X if obsm is None else adata[:, marker].obsm[obsm]
    column = column.flatten()
    return flowutils.transforms.logicle_inverse(
        column, None, *adata.uns["scyan_logicle"][marker]
    )


def asinh_transform(adata: AnnData, translation: float = 1, cofactor: float = 5) -> None:
    """Asinh transformation for cell-expressions: $asinh((x - translation)/cofactor)$.

    Args:
        adata: An `anndata` object.
        translation: Constant substracted to the marker expression before division by the cofactor.
        cofactor: Scaling factor before computing the asinh.
    """
    adata.uns["scyan_asinh"] = [translation, cofactor]
    adata.X = np.arcsinh((adata.X - translation) / cofactor)


def inverse_transform(
    adata: AnnData,
    obsm: Optional[str] = None,
    obsm_names: Optional[List[str]] = None,
    transformation: Optional[str] = None,
) -> np.ndarray:
    """Inverse the transformation function, i.e. either [scyan.tools.auto_logicle_transform][] or [scyan.tools.asinh_transform][]. It requires to have run have of these before.

    !!! note
        If you scaled your data, the complete inverse consists in running [scyan.tools.unscale][] first, and then this function.

    Args:
        adata: An `anndata` object.
        obsm: Name of the anndata obsm to consider. If `None`, use `adata.X`.
        obsm_names: Names of the ordered markers from obsm. It is required if obsm is not `None`, if there are less markers than in `adata.X`, and if the transformation to reverse is `logicle`. Usually, it corresponds to `model.var_names`.
        transformation: Name of the transformation to inverse: one of `['logicle', 'asinh', None]`. By default, it chooses automatically depending on which transformation was previously run.

    Returns:
        Inverse transformed expressions array of shape $(N, M)$.
    """
    if transformation is None:
        transformation = "asinh" if "scyan_asinh" in adata.uns else None
        transformation = "logicle" if "scyan_logicle" in adata.uns else transformation
        if transformation is None:
            raise ValueError(
                "No transformation to inverse: you need to run 'asinh_transform' or 'auto_logicle_transform' before to inverse it."
            )

    if transformation == "logicle":
        log.info("Performing inverse logicle transform")
        assert (
            "scyan_logicle" in adata.uns
        ), "You need to run 'auto_logicle_transform' before to inverse it."

        if obsm is None:
            obsm_names = adata.var_names
        elif obsm_names is None:
            assert (
                adata.obsm[obsm].shape[1] != adata.n_vars
            ), f"When the number of var in adata.obsm['{obsm}'] is not `adata.n_vars`, use `obs_names`"
            obsm_names = adata.var_names

        return np.stack(
            [_logicle_inverse_one(adata, obsm, marker) for marker in obsm_names],
            axis=1,
        )

    if transformation == "asinh":
        log.info("Performing inverse asinh transform")
        assert (
            "scyan_asinh" in adata.uns
        ), "You need to run 'asinh_transform' before to inverse it."

        X = adata.X if obsm is None else adata.obsm[obsm]
        translation, cofactor = adata.uns["scyan_asinh"]

        return np.sinh(X) * cofactor + translation

    raise NameError(
        f"Parameter 'transformation' has to be 'logicle' or 'asinh'. Found {transformation}."
    )


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


def unscale(
    adata: AnnData, obsm: Optional[str] = None, obsm_names: Optional[List[str]] = None
) -> np.ndarray:
    """Reverse standardisation. It requires to have run [scyan.tools.scale][] before.

    Args:
        adata: An `anndata` object.
        obsm: Name of the adata obsm to consider. If `None`, use `adata.X`.
        obsm_names: Names of the ordered markers from obsm. It is required if obsm is not `None`, and if there are less markers than in `adata.X`. Usually, it corresponds to `model.var_names`.

    Returns:
        Unscaled numpy array of shape $(N, M)$.
    """
    assert (
        "scyan_scaling" in adata.uns
    ), "It seems you haven't run 'scyan.tools.scale' before."

    X = adata.X if obsm is None else adata.obsm[obsm]
    means, stds = adata.uns["scyan_scaling"]["means"], adata.uns["scyan_scaling"]["stds"]

    if obsm is not None and X.shape[1] != adata.n_vars:
        assert (
            obsm_names is not None
        ), f"Found {X.shape[1]} markers in adata.obsm['{obsm}'], but 'adata' has {adata.n_vars} vars. Please use the 'obsm_names' argument to provide the ordered names of the markers used in adata.obsm['{obsm}']."

        indices = [adata.var_names.get_loc(marker) for marker in obsm_names]
        means, stds = means[indices], stds[indices]

    return means + X * stds
