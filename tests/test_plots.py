import pytest

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
    adata, table = scyan.data.load("aml", version="short")
    model = Scyan(adata, table)
    model.fit(max_epochs=1)
    model.predict()
    return model


def test_kde_per_population(model: Scyan, pop: str, ref: str):
    scyan.plot.kde_per_population(model.adata, pop, show=False)


def test_pop_expressions(model: Scyan, pop: str, ref: str):
    scyan.plot.pop_expressions(model, pop, show=False)


def test_probs_per_marker(model: Scyan, pop: str, ref: str):
    scyan.plot.probs_per_marker(model, pop, show=False)


def test_pops_expressions(model: Scyan):
    scyan.plot.pops_expressions(model, show=False)


def test_scatter(model: Scyan, pop: str, ref: str):
    scyan.plot.scatter(model.adata, [pop, ref], max_obs=100, show=False)
