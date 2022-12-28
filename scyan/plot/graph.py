from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

from .. import Scyan
from .utils import plot_decorator


@plot_decorator()
def pops_hierarchy(model: Scyan, figsize: tuple = (18, 5), show: bool = True) -> None:
    """Plot populations as a tree, where each level corresponds to more detailed populations. To run this function, your knowledge table need to contain at least one population 'level' (see [this tutorial](../../tutorials/advanced/#hierarchical-population-display)), and you need to install `graphviz`.

    Args:
        model: Scyan model.
        figsize: Matplotlib figure size.
        show: Whether or not to display the figure.
    """
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout

    table = model.table

    assert isinstance(
        table.index, pd.MultiIndex
    ), "To plot population hierarchy, you need a MultiIndex DataFrame. See the documentation for more details."

    G = nx.DiGraph()
    G.add_node(" ")

    def add_nodes(table, indices, level, parent=" "):
        if level == -1:
            return

        index = table.index.get_level_values(level)
        dict_indices = defaultdict(list)
        for i in indices:
            dict_indices[index[i]].append(i)

        for name, indices in dict_indices.items():
            if not name == parent:
                G.add_node(name)
                G.add_edge(parent, name)
            add_nodes(table, indices, level - 1, name)

    add_nodes(
        table,
        range(len(table)),
        table.index.nlevels - 1,
    )

    plt.figure(figsize=figsize)
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True, arrows=False, node_size=0)
