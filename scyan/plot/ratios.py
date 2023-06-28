import logging
from math import ceil
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib.lines import Line2D

from ..tools import cell_type_ratios
from .utils import plot_decorator

log = logging.getLogger(__name__)


@plot_decorator(adata=True)
def pop_percentage(
    adata: AnnData,
    groupby: Union[str, List[str], None] = None,
    key: str = "scyan_pop",
    show: bool = True,
):
    """Show populations percentages. Depending on `groupby`, this is either done globally, or as a stacked bar plot (one bar for each group).

    Args:
        adata: An `AnnData` object.
        groupby: Key(s) of `adata.obs` used to create groups (e.g. the patient ID).
        key: Key of `adata.obs` containing the population names (or the values) for which percentage will be displayed.
        show: Whether or not to display the figure.
    """
    if groupby is None:
        adata.obs[key].value_counts(normalize=True).mul(100).plot.bar()
    else:
        adata.obs.groupby(groupby)[key].value_counts(normalize=True).mul(
            100
        ).unstack().plot.bar(stacked=True)
        plt.legend(
            bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, frameon=False
        )

    plt.ylabel(f"{key} percentage")
    sns.despine(offset=10, trim=True)
    plt.xticks(rotation=90)


@plot_decorator(adata=True)
def pop_dynamics(
    adata: AnnData,
    time_key: str,
    groupby: Union[str, List[str], None] = None,
    key: str = "scyan_pop",
    among: str = None,
    n_cols: int = 4,
    size_mul: Optional[float] = None,
    figsize: tuple[float, float] = None,
    show: bool = True,
):
    """Show populations percentages dynamics for different timepoints. Depending on `groupby`, this is either done globally, or for each group.

    Args:
        adata: An `AnnData` object.
        time_key: Key of `adata.obs` containing the timepoints. We recommend to use a categorical series (to use the right timepoint order).
        groupby: Key(s) of `adata.obs` used to create groups (e.g. the patient ID).
        key: Key of `adata.obs` containing the population names (or the values) for which dynamics will be displayed.
        among: Key of `adata.obs` containing the parent population name. See [scyan.tools.cell_type_ratios][].
        n_cols: Number of figures per row.
        size_mul: Dot size multiplication factor. By default, it is computed using the population counts.
        figsize: matplotlib figure size.
        show: Whether or not to display the figure.
    """
    if not adata.obs[time_key].dtype.name == "category":
        log.info(f"Converting adata.obs['{time_key}'] to categorical")
        adata.obs[time_key] = adata.obs[time_key].astype("category")

    if groupby is None:
        groupby = [time_key]
    else:
        groupby = ([groupby] if isinstance(groupby, str) else groupby) + [time_key]

    df = cell_type_ratios(adata, groupby=groupby, key=key, normalize="%", among=among)
    df_log_count = np.log(
        1 + cell_type_ratios(adata, groupby=groupby, key=key, normalize=False)
    )

    n_pops = df.shape[1]
    n_rows = ceil(n_pops / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (12, 3 * n_rows))

    if size_mul is None:
        size_mul = 40 / df_log_count.mean().mean()

    axes = axes.flatten()

    if len(groupby) == 1:
        for i, pop in enumerate(df.columns):
            axes[i].plot(df.index, df.iloc[:, i])
            axes[i].scatter(df.index, df.iloc[:, i], s=df_log_count.iloc[:, i] * size_mul)
    else:
        drop_levels = list(range(len(groupby) - 1))

        for group, group_df in df.groupby(level=drop_levels):
            label = " ".join(map(str, group)) if isinstance(group, tuple) else group
            group_df = group_df.droplevel(drop_levels)
            group_df_log_count = df_log_count.loc[group]

            for i, pop in enumerate(group_df.columns):
                axes[i].plot(group_df.index, group_df.iloc[:, i], label=label)
                axes[i].scatter(
                    group_df.index,
                    group_df.iloc[:, i],
                    s=(group_df_log_count.iloc[:, i] * size_mul).clip(0, 100),
                )

        fig.legend(
            *axes[0].get_legend_handles_labels(),
            bbox_to_anchor=(1.04, 0.55),
            loc="lower left",
            borderaxespad=0,
            frameon=False,
        )

    timepoints = adata.obs[time_key].cat.categories
    for i, pop in enumerate(df.columns):
        axes[i].set_ylabel(pop)
        axes[i].set_xlabel(time_key)
        axes[i].set_xticks(range(len(timepoints)), timepoints)

    sizes = [1, 10, 25, 40, 60]
    legend_markers = [
        Line2D([0], [0], linewidth=0, marker="o", markersize=np.sqrt(s)) for s in sizes
    ]
    legend2 = fig.legend(
        legend_markers,
        [f" {ceil(np.exp(s / size_mul) - 1):,} cells" for s in sizes],
        bbox_to_anchor=(1.04, 0.45),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
    )
    fig.add_artist(legend2)

    for ax in axes[n_pops:]:
        ax.set_axis_off()

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
