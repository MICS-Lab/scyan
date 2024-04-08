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
    adata.raw = adata.copy()
    return adata


@pytest.fixture
def raw_adata_cytof() -> AnnData:
    adata, _ = scyan.data.load("aml", version="short")
    adata = adata.raw.to_adata()
    adata.X = adata.X.clip(0)
    adata.raw = adata
    return adata


@pytest.fixture
def test_asinh(raw_adata_cytof: AnnData):
    scyan.preprocess.asinh_transform(raw_adata_cytof)
    return raw_adata_cytof


@pytest.fixture
def test_autologicle(raw_adata: AnnData) -> AnnData:
    scyan.preprocess.auto_logicle_transform(raw_adata, quantile_clip=None)
    return raw_adata


def test_inverse_asinh(test_asinh: AnnData):
    assert "scyan_asinh" in test_asinh.uns
    inversed = scyan.preprocess.inverse_transform(test_asinh)
    assert_is_zero(inversed - test_asinh.raw.X, 1e-3)


def test_inverse_logicle(test_autologicle: AnnData):
    assert "scyan_logicle" in test_autologicle.uns
    inversed = scyan.preprocess.inverse_transform(test_autologicle)
    assert_is_zero(inversed - test_autologicle.raw.X, 1e-2)


def test_scale_cytof(test_asinh: AnnData) -> AnnData:
    scyan.preprocess.scale(test_asinh)

    assert test_asinh.X.min() == -1


def test_scale_unscale(test_autologicle: AnnData) -> AnnData:
    adata = test_autologicle
    adata.raw = adata

    scyan.preprocess.scale(adata, max_value=np.inf)

    assert_is_zero(adata.X.std(axis=0) - 1)

    unscaled = scyan.preprocess.unscale(adata)

    assert_is_zero(adata.raw.X - unscaled)
