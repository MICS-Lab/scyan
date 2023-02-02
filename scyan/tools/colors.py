import colorsys
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


class GroupPalette:
    def __init__(self, alpha_l: float, step_l: float, alpha_s: float, step_s: float):
        self.alpha_l = alpha_l
        self.alpha_s = alpha_s
        self.step_l = step_l
        self.step_s = step_s

        self.k_l = 1 + int((1 - 2 * alpha_l) // step_l)
        self.k_s = 1 + int((1 - 2 * alpha_s) // step_s)
        self.max_size = self.k_l * self.k_s

    def get_ls(self, i: int, j: int):
        l = 1 - self.alpha_l - i * self.step_l
        s = 1 - self.alpha_s - j * self.step_s
        return l, s

    def get_block(self, h: float, size: int) -> List[Tuple[float]]:
        assert (
            size <= self.max_size
        ), f"Too many categories ({size}): can define at most {self.max_size} colors per group. Consider lowering these arguments: step_l, step_s, alpha_l, alpha_s."

        indices = [(i % self.k_l, i // self.k_l) for i in range(size)]
        indices = sorted(indices)

        return [colorsys.hls_to_rgb(h, *self.get_ls(i, j)) for i, j in indices]

    def __call__(self, sizes: List[int], hue_shift: float) -> List:
        hues = sorted(range(len(sizes)), key=lambda x: [x % 3, x])
        hues = np.array(hues) / len(sizes) + hue_shift

        return [self.get_block(h, size) for h, size in zip(hues, sizes)]


def palette_level(
    table: pd.DataFrame,
    population_index: Union[int, str] = 0,
    level_index: Union[int, str] = 1,
    hue_shift: float = 0.4,
    alpha_l: float = 0.25,
    step_l: float = 0.15,
    alpha_s: float = 0.3,
    step_s: float = 0.4,
) -> Dict[str, Tuple[float]]:
    """Computes a color palette that in grouped by the hierarchical main populations. It improves the UMAP readability when many populations are defined.

    !!! info
        Once such a color palette is defined, you can use it for plotting. For instance, try `scyan.plot.umap(adata, color="scyan_pop", palette=palette)`, where `palette` is the one you created with this function.

    Args:
        table: Knowledge table provided to Scyan. It must be a multi-index DataFrame.
        population_index: Index or name of the level in `table.index` storing the low-level/children population names.
        level_index: Index or name of the level in `table.index` storing the main population names.
        hue_shift: Shift the hue values. The value must be a float in `[0, 1]`.
        alpha_l: Lower it to have a larger lightness range of colors.
        step_l: Increase it to have more distinct colors (in term of lightness).
        alpha_s: Lower it to have a larger saturation range of colors.
        step_s: Increase it to have more distinct colors (in term of saturation).

    Returns:
        A dictionnary whose keys are population names and values are RGB colors.
    """
    assert isinstance(
        table.index, pd.MultiIndex
    ), f"The provided table has no multi-index. To work with hierarchical populations, consider reading https://mics-lab.github.io/scyan/tutorials/usage/#working-with-hierarchical-populations"

    pops = table.index.get_level_values(population_index).values
    level = table.index.get_level_values(level_index)
    level_counts = level.value_counts()

    group_palette = GroupPalette(alpha_l, step_l, alpha_s, step_s)
    color_groups = group_palette(level_counts.values, hue_shift)

    block_indices = [level_counts.index.get_loc(pop) for pop in level]
    s = pd.Series(level)
    inner_block_indices = s.groupby(s).cumcount().values

    return {
        pop: list(color_groups[block_index][inner_index])
        for pop, block_index, inner_index in zip(pops, block_indices, inner_block_indices)
    }
