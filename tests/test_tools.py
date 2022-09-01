import numpy as np
import pytest
from anndata import AnnData

import scyan


def assert_is_zero(a: np.ndarray, atol=1e-5) -> bool:
    assert np.isclose(a, 0, atol=atol).all()


@pytest.fixture
def raw_adata() -> AnnData:
    adata, _ = scyan.data.load("aml", version="short")
    adata = adata.raw.to_adata()
    adata.raw = adata
    return adata


@pytest.fixture
def test_asinh(raw_adata: AnnData):
    scyan.tools.asinh_transform(raw_adata)
    return raw_adata


@pytest.fixture
def test_autologicle(raw_adata: AnnData) -> AnnData:
    scyan.tools.auto_logicle_transform(raw_adata)
    return raw_adata


def test_inverse_asinh(test_asinh: AnnData):
    assert "scyan_asinh" in test_asinh.uns
    inversed = scyan.tools.inverse_transform(test_asinh)
    assert_is_zero(inversed - test_asinh.raw.X, 1e-3)


def test_inverse_logicle(test_autologicle: AnnData):
    assert "scyan_logicle" in test_autologicle.uns
    inversed = scyan.tools.inverse_transform(test_autologicle)
    assert_is_zero(inversed - test_autologicle.raw.X, 1e-2)


def test_scale_unscale(test_autologicle: AnnData) -> AnnData:
    adata = test_autologicle
    adata.raw = adata

    scyan.tools.scale(adata, max_value=np.inf)

    assert_is_zero(adata.X.mean(axis=0))
    assert_is_zero(adata.X.std(axis=0) - 1)

    unscaled = scyan.tools.unscale(adata)

    assert_is_zero(adata.raw.X - unscaled)
