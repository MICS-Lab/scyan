import numpy as np
import pytest
from anndata import AnnData

import scyan


@pytest.fixture
def raw_adata() -> AnnData:
    adata, _ = scyan.data.load("aml", size="short")
    return adata.raw.to_adata()


def test_asinh(raw_adata: AnnData):
    scyan.tools.asinh_transform(raw_adata)


@pytest.fixture
def test_autologicle(raw_adata: AnnData) -> AnnData:
    scyan.tools.auto_logicle_transform(raw_adata)
    return raw_adata


def is_zero(a: np.ndarray) -> bool:
    return np.isclose(a, 0, atol=1e-5).all()


def test_scale_unscale(test_autologicle: AnnData) -> AnnData:
    adata = test_autologicle
    adata.raw = adata

    scyan.tools.scale(adata, max_value=np.inf)

    assert is_zero(adata.X.mean(axis=0))
    assert is_zero(adata.X.std(axis=0) - 1)

    unscaled = scyan.tools.unscale(adata)

    assert is_zero(adata.raw.X - unscaled)
