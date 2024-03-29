import pytest
import torch

import scyan
from scyan import Scyan


@pytest.fixture
def init_data():
    return scyan.data.load("aml", version="short")


@pytest.fixture
def short_model(init_data):
    return Scyan(*init_data)


@pytest.fixture
def short_model_batch_effect(init_data):
    return Scyan(*init_data, batch_key="subject")


@pytest.fixture
def test_run_model(short_model: Scyan):
    short_model.fit(max_epochs=2)
    return short_model


def test_inverse(test_run_model: Scyan):
    u = test_run_model()
    x = test_run_model.module.inverse(u, test_run_model.covariates)

    assert torch.isclose(x, test_run_model.x, atol=1e-6).all()


def test_run_model_batch_effect(short_model_batch_effect: Scyan):
    short_model_batch_effect.fit(max_epochs=1)
    return short_model_batch_effect


def test_predict_model(test_run_model: Scyan):
    test_run_model.predict()
    return test_run_model


def test_sample1(test_run_model: Scyan):
    test_run_model.sample(123)


def test_sample2(test_run_model: Scyan):
    test_run_model.sample(123, pop="CD4 T cells")


def test_sample3(test_run_model: Scyan):
    test_run_model.sample(123, pop=3)


def test_sample4(test_run_model: Scyan):
    test_run_model.sample(2, pop=["CD4 T cells", "CD8 T cells"])


def test_sample5(test_run_model: Scyan):
    test_run_model.sample(3, pop=torch.Tensor([1, 3, 7]))
