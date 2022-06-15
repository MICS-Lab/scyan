from typing import Callable, List, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .. import Scyan


def optional_show(f: Callable) -> Callable:
    """Decorator that shows a matplotlib figure if the provided 'show' argument is True"""

    def wrapper(*args, **kwargs):
        f(*args, **kwargs)
        if kwargs.get("show", True):
            plt.show()

    return wrapper


def check_population(return_list: bool = False):
    """Decorator that checks if the provided population (or populations) exists"""

    def decorator(f: Callable) -> Callable:
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
            f(model, population, *args, obs_key=obs_key, **kwargs)

        return wrapper

    return decorator


def get_palette_others(data, key, default="Set1", others="Others"):
    pops = data[key].unique()
    colors = sns.color_palette(default, len(pops))
    colors = dict(zip(pops, colors))
    colors[others] = (0.5, 0.5, 0.5)
    return colors
