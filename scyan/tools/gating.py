import logging

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from ..plot.utils import check_has_umap

log = logging.getLogger(__name__)


class _SelectFromCollection:
    """From https://matplotlib.org/stable/gallery/widgets/polygon_selector_demo.html"""

    def __init__(self, ax, collection, alpha_other=0.3):
        from matplotlib.widgets import PolygonSelector

        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.n_obs = len(self.xys)

        self.fc = collection.get_facecolors()
        self.fc = np.tile(self.fc, (self.n_obs, 1))

        self.poly = PolygonSelector(ax, self.onselect, draw_bounding_box=True)
        self.ind = []

    def onselect(self, verts):
        from matplotlib.path import Path

        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


class PolygonGating:
    """Prompts the user to make a polygon gate on a UMAP plot to select cells.

    !!! note

        We recommend using it on Jupyter Notebooks. To be able to select the cells, you should first run `%matplotlib tk` on a blank jupyter cell. After the selection, you can run `%matplotlib inline` to retrieve the default behavior.

    ```py
    # Usage example (to be run on a jupyter notebook)
    >>> %matplotlib tk
    >>> polygon_gating = scyan.tools.PolygonGating(adata)
    >>> polygon_gating.select() # select the cells
    >>> polygon_gating.save_selection() # save the selected cells in adata.obs
    ```
    """

    def __init__(self, adata: AnnData) -> None:
        """
        Args:
            adata: An `anndata` object.
        """
        check_has_umap(lambda adata: 0)
        self.adata = adata
        self.x_umap = self.adata.obsm["X_umap"]
        self.has_umap = self.x_umap.sum(1) != 0

    def select(self, s: float = 0.05) -> None:
        """Open a UMAP plot on which you can draw a polygon to select cells.

        Args:
            s: Size of the cells on the plot.
        """
        _, ax = plt.subplots()

        pts = ax.scatter(
            self.x_umap[self.has_umap, 0],
            self.x_umap[self.has_umap, 1],
            marker=".",
            rasterized=True,
            s=s,
        )

        self.selector = _SelectFromCollection(ax, pts)

        print(
            f"Enclose cells within a polygon. Helper:\n    - Click on the plot to add a polygon vertex\n    - Press the 'esc' key to start a new polygon\n    - Try holding the 'ctrl' key to move a single vertex"
        )
        plt.show()

    def save_selection(self, key_added: str = "scyan_selected"):
        """Save the selected cells in `adata.obs[key_added]`.

        Args:
            key_added: Column name used to save the selected cells in `adata.obs`.
        """
        self.adata.obs[key_added] = False
        col_index = self.adata.obs.columns.get_loc(key_added)
        self.adata.obs.iloc[
            np.where(self.has_umap)[0][self.selector.ind], col_index
        ] = True
        print(f"Cell selection saved in adata.obs['{key_added}']")
