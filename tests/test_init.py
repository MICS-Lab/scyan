import pytest

import scyan

@pytest.mark.parametrize("dataset", ["aml", "bmmc"])
def test_init(dataset):
    adata, marker_pop_matrix = scyan.data.load(dataset)
    model = scyan.Scyan(adata, marker_pop_matrix)
    