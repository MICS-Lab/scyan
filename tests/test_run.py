import pytest
import torch

import scyan
from scyan import Scyan


@pytest.mark.parametrize("dataset", ["aml", "bmmc"])
def test_init_model(dataset):
    adata, marker_pop_matrix = scyan.data.load(dataset)
    Scyan(adata, marker_pop_matrix)


@pytest.fixture
def short_model():
    adata, marker_pop_matrix = scyan.data.load("aml", size="short")
    return Scyan(adata, marker_pop_matrix)


def test_inverse(short_model: Scyan):
    u = short_model()
    x = short_model.module.inverse(u, short_model.covariates)

    assert torch.isclose(x, short_model.x, atol=1e-6).all()


@pytest.fixture
def test_run_model(short_model: Scyan):
    short_model.fit(max_epochs=2)
    return short_model


@pytest.fixture
def test_predict_model(test_run_model: Scyan):
    test_run_model.predict()
    return test_run_model


def test_sample(test_run_model: Scyan):
    test_run_model.sample(123)
