import pytest

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


@pytest.fixture
def test_run_model(short_model: Scyan):
    short_model.fit(max_epochs=2)
    return short_model


@pytest.fixture
def test_predict_model(test_run_model: Scyan):
    test_run_model.predict()
    return test_run_model


def test_knn_predict(test_predict_model: Scyan):
    test_predict_model.knn_predict(n_neighbors=4)


def test_sample(test_run_model: Scyan):
    test_run_model.sample(123)
