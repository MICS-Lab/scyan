import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scyan

pop_names = ["a1", "a2", "b", "c"]


@pytest.fixture
def adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 2, 1, 2],
            "scyan_pop": ["a1", "a2", "a2", "a2", "b", "b", "c", "c"],
            "scyan_pop_level": ["A", "A", "A", "A", "B", "B", "C", "C"],
        }
    )
    values = np.array([[1, 2], [0, 0], [1, 4], [0, 0], [1, 0], [0, 0], [0, 0], [2, 1]])
    return AnnData(values, obs=obs, dtype=np.float32)


def test_count_cell_types(adata: AnnData):
    df = scyan.tools.cell_type_ratios(adata, normalize=False)

    assert all(
        df[f"{pop} count"][0] == count for pop, count in zip(pop_names, [1, 3, 2, 2])
    )


def test_normalize_cell_populations(adata: AnnData):
    df = scyan.tools.cell_type_ratios(adata)

    assert all(
        df[f"{pop} percentage"][0] == count / 8
        for pop, count in zip(pop_names, [1, 3, 2, 2])
    )


def test_group_cell_populations(adata: AnnData):
    df = scyan.tools.cell_type_ratios(adata, groupby="id", normalize=False)

    assert df.loc[1, "c count"] == 1
    assert df.loc[1, "a2 count"] == 3
    assert np.isnan(df.loc[2, "a1 count"])


def test_cell_populations_among(adata: AnnData):
    df = scyan.tools.cell_type_ratios(adata, groupby="id", among="scyan_pop_level")

    assert df.loc[1, "c percentage among C"] == 1
    assert df.loc[1, "a2 percentage among A"] == 0.75
    assert np.isnan(df.loc[2, "a2 percentage among A"])


def test_mean_intensities(adata: AnnData):
    series = scyan.tools.mean_intensities(adata)

    assert series["0 mean intensity on a1"] == 1
    assert series["0 mean intensity on b"] == 0.5
    assert series["1 mean intensity on c"] == 0.5


def test_mean_intensities(adata: AnnData):
    df = scyan.tools.mean_intensities(adata, groupby="id")

    assert df.loc[1, "0 mean intensity on a1"] == 1
    assert df.loc[1, "0 mean intensity on b"] == 1
    assert df.loc[2, "1 mean intensity on c"] == 1
    assert np.isnan(df.loc[2, "1 mean intensity on a1"])
