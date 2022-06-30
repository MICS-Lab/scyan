import pytest
import torch

import scyan
from scyan import Scyan


@pytest.fixture
def init_data():
    return scyan.data.load("aml", size="short")


@pytest.fixture
def short_model(init_data):
    return Scyan(*init_data)


@pytest.fixture
def short_model_batch_effect(init_data):
    return Scyan(*init_data, batch_key="subject", batch_ref="H1")


def test_inverse(short_model: Scyan):
    u = short_model()
    x = short_model.module.inverse(u, short_model.covariates)

    assert torch.isclose(x, short_model.x, atol=1e-6).all()


@pytest.fixture
def test_run_model(short_model: Scyan):
    short_model.fit(max_epochs=2)
    return short_model


def test_run_model_batch_effect(short_model_batch_effect: Scyan):
    short_model_batch_effect.fit(max_epochs=1)
    return short_model_batch_effect


def test_predict_model(test_run_model: Scyan):
    test_run_model.predict()
    return test_run_model


def test_sample(test_run_model: Scyan):
    test_run_model.sample(123)
