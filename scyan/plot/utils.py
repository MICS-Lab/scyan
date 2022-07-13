from collections import defaultdict
from functools import wraps
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from .. import Scyan


def optional_show(f: Callable) -> Callable:
    """Decorator that shows a matplotlib figure if the provided 'show' argument is True"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        if kwargs.get("show", True):
            plt.show()
        return res

    return wrapper


def check_population(return_list: bool = False):
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
            if isinstance(population, (list, tuple, np.ndarray)):
                not_found_names = [p for p in population if p not in populations]
                if not_found_names:
                    raise NameError(
                        f"Invalid input population list. {not_found_names} has to be inside {populations}."
                    )
            else:
                if population not in populations:
                    raise NameError(
                        f"Invalid input population. {population} has to be one of {populations}."
                    )
                if return_list:
                    population = [population]
            return f(model, population, *args, obs_key=obs_key, **kwargs)

        return wrapper

    return decorator


def get_palette_others(data, key, default="Set1", others="Others", value=0.5):
    pops = data[key].unique()
    colors = sns.color_palette(default, len(pops))
    colors = dict(zip(pops, colors))
    colors[others] = (value, value, value)
    return colors


def ks_statistics(model: Scyan, obs_key: str, populations: List[str]):
    adata = model.adata
    statistics = defaultdict(float)

    for pop in populations:
        adata1 = adata[adata.obs[obs_key] == pop]

        if len(populations) > 1:
            other_pops = [p for p in populations if p != pop]
            adata2 = adata[np.isin(adata.obs[obs_key], other_pops)]
        else:
            adata2 = adata[adata.obs[obs_key] != pop]

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
