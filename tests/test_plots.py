import pytest
from typing import List

import scyan
from scyan import Scyan


@pytest.fixture
def pop() -> str:
    return "CD8 T cells"


@pytest.fixture
def ref() -> str:
    return "CD4 T cells"


@pytest.fixture
def model() -> Scyan:
    adata, marker_pop_matrix = scyan.data.load("aml", size="short")
    model = Scyan(adata, marker_pop_matrix)
    model.predict()
    return model


def test_kde_per_population(model: Scyan, pop: str, ref: str):
    scyan.plot.kde_per_population(model, pop, show=False)


def test_latent_expressions(model: Scyan, pop: str, ref: str):
    scyan.plot.latent_expressions(model, pop, show=False)


def test_pop_weighted_kde(model: Scyan, pop: str, ref: str):
    scyan.plot.pop_weighted_kde(model, pop, n_samples=model.adata.n_obs, show=False)


def test_pop_weighted_kde_with_ref(model: Scyan, pop: str, ref: str):
    scyan.plot.pop_weighted_kde(
        model, pop, n_samples=model.adata.n_obs, ref=ref, show=False
    )


def test_probs_per_marker(model: Scyan, pop: str, ref: str):
    scyan.plot.probs_per_marker(model, pop, show=False)


def test_latent_heatmap(model: Scyan):
    scyan.plot.latent_heatmap(model, show=False)


def test_scatter(model: Scyan, pop: str, ref: str):
    scyan.plot.scatter(model, [pop, ref], max_obs=100, show=False)
