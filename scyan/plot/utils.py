from collections import defaultdict
from functools import wraps
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .. import Scyan
from ..utils import _get_subset_indices


def plot_decorator(f: Callable) -> Callable:
    """Decorator that shows a matplotlib figure if the provided 'show' argument is True"""

    @wraps(f)
    def wrapper(model, *args, **kwargs):
        assert isinstance(
            model, Scyan
        ), f"{f.__name__} first argument has to be scyan.Scyan model. Received type {type(model)}."

        res = f(model, *args, **kwargs)
        if kwargs.get("show", True):
            plt.show()
        return res

    return wrapper


def check_population(return_list: bool = False, one: bool = False):
    """Decorator that checks if the provided population (or populations) exists"""

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(
            model: Scyan,
            population: Union[str, List[str]],
            *args,
            obs_key="scyan_pop",
            **kwargs,
        ):
            if model.adata.obs[obs_key].dtype == "category":
                populations = model.adata.obs[obs_key].cat.categories.values
            else:
                populations = set(model.adata.obs[obs_key].values)
            if isinstance(population, str):
                if population not in populations:
                    raise NameError(
                        f"Invalid input population. '{population}' has to be one of {populations}."
                    )
                if return_list:
                    population = [population]
            else:
                if one:
                    raise ValueError(
                        f"Argument 'population' has to be a string. Found {population}."
                    )
                not_found_names = [p for p in population if p not in populations]
                if not_found_names:
                    raise NameError(
                        f"Invalid input population list. {not_found_names} has to be inside {populations}."
                    )
            return f(model, population, *args, obs_key=obs_key, **kwargs)

        return wrapper

    return decorator


def get_palette_others(
    data: pd.DataFrame,
    key: str,
    palette: str = None,
    others: str = "Others",
    value: float = 0.5,
):
    pops = data[key].unique()

    colors = sns.color_palette(palette or "Set1", len(pops))
    colors = dict(zip(pops, colors))
    if others in colors.keys():
        colors[others] = (value, value, value)

    return colors


def ks_statistics(
    model: Scyan, obs_key: str, populations: List[str], max_obs: int = 5000
):
    adata = model.adata
    statistics = defaultdict(float)

    for pop in populations:
        adata1 = adata[adata.obs[obs_key] == pop]

        if len(populations) > 1:
            other_pops = [p for p in populations if p != pop]
            adata2 = adata[np.isin(adata.obs[obs_key], other_pops)]
        else:
            adata2 = adata[adata.obs[obs_key] != pop]

        adata1 = adata1[_get_subset_indices(adata1, max_obs)]
        adata2 = adata2[_get_subset_indices(adata2, max_obs)]

        for marker in model.var_names:
            statistics[marker] += stats.kstest(
                adata1[:, marker].X.flatten(), adata2[:, marker].X.flatten()
            ).statistic

    return statistics


def select_markers(
    model: Scyan,
    markers: Optional[List[str]],
    n_markers: Optional[int],
    obs_key: str,
    populations: List[str],
    min_markers: int = 2,
):
    MIN_MARKERS_ERROR = f"Provide at least {min_markers} marker(s) to plot or use 'scyan.plot.kde_per_population'"

    if markers is None:
        assert (
            n_markers is not None
        ), "You need to provide a list of markers or a number of markers to be chosen automatically"
        assert n_markers >= min_markers, MIN_MARKERS_ERROR

        statistics = ks_statistics(model, obs_key, populations)
        statistics = sorted(statistics.items(), key=lambda x: x[1], reverse=True)
        markers = [m for m, _ in statistics[:n_markers]]

    assert len(markers) >= min_markers, MIN_MARKERS_ERROR
    return markers
