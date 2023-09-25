import logging

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from ..utils import _has_umap

log = logging.getLogger(__name__)


class _SelectFromCollection:
    """Updated from https://matplotlib.org/stable/gallery/widgets/polygon_selector_demo.html"""

    def __init__(self, ax, collection, xy: np.ndarray, alpha_other: float = 0.3):
        from matplotlib.widgets import PolygonSelector

        self.canvas = ax.figure.canvas
        self.collection = collection
        self.xy = xy
        self.alpha_other = alpha_other

        self.n_obs = len(self.xy)

        self.fc = collection.get_facecolors()
        self.fc = np.tile(self.fc, (self.n_obs, 1))

        self.poly = PolygonSelector(ax, self.onselect, draw_bounding_box=True)
        self.ind = []

    def onselect(self, verts):
        from matplotlib.path import Path

        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xy))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()


class PolygonGatingUMAP:
    """Class used to select cells on a UMAP using polygons.

    !!! note

        If used on a Jupyter Notebook, you should first run `%matplotlib tk`. After the selection, you can run `%matplotlib inline` to retrieve the default behavior.

    ```py
    # Usage example (`%matplotlib tk` is required for the cell selection on jupyter notebooks)
    >>> %matplotlib tk
    >>> selector = scyan.tools.PolygonGatingUMAP(adata)
    >>> selector.select()         # select the cells

    >>> sub_adata = selector.extract_adata() # on a notebook, this has to be on a new jupyter cell
    ```
    """

    def __init__(self, adata: AnnData) -> None:
        """
        Args:
            adata: An `AnnData` object.
        """
        self.adata = adata
        self.has_umap = _has_umap(adata)
        self.x_umap = self.adata.obsm["X_umap"]

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

        self.selector = _SelectFromCollection(ax, pts, self.x_umap[self.has_umap])

        log.info(
            f"Enclose cells within a polygon. Helper:\n    - Click on the plot to add a polygon vertex\n    - Press the 'esc' key to start a new polygon\n    - Try holding the 'ctrl' key to move a single vertex\n    - Once the polygon is finished and overlaid in red, you can close the window"
        )
        plt.show()

    def save_selection(self, key_added: str = "scyan_selected"):
        """Save the selected cells in `adata.obs[key_added]`.

        Args:
            key_added: Column name used to save the selected cells in `adata.obs`.
        """
        self.adata.obs[key_added] = "unselected"
        col_index = self.adata.obs.columns.get_loc(key_added)
        self.adata.obs.iloc[
            np.where(self.has_umap)[0][self.selector.ind], col_index
        ] = "selected"
        self.adata.obs[key_added] = self.adata.obs[key_added].astype("category")

        self.selector.disconnect()
        log.info(
            f"Selected {len(self.selector.ind)} cells and saved the selection in adata.obs['{key_added}']"
        )

    def extract_adata(self) -> AnnData:
        """Returns an anndata objects whose cells where inside the polygon"""
        log.info(f"Selected {len(self.selector.ind)} cells")
        self.selector.disconnect()

        return self.adata[np.where(self.has_umap)[0][self.selector.ind]]


class PolygonGatingScatter:
    """Class used to select cells on a scatterplot using polygons.

    !!! note

        If used on a Jupyter Notebook, you should first run `%matplotlib tk` on a blank jupyter cell. After the selection, you can run `%matplotlib inline` to retrieve the default behavior.

    ```py
    # Usage example (`%matplotlib tk` is required for the cell selection on jupyter notebooks)
    >>> %matplotlib tk
    >>> selector = scyan.tools.PolygonGatingScatter(adata)
    >>> selector.select()         # select the cells

    >>> sub_adata = selector.extract_adata() # on a notebook, this has to be on a new jupyter cell
    ```
    """

    def __init__(self, adata: AnnData) -> None:
        """
        Args:
            adata: An `AnnData` object.
        """
        self.adata = adata

    def select(
        self, x: str, y: str, s: float = 0.05, max_cells_display: int = 100_000
    ) -> None:
        """Open a scatter plot on which you can draw a polygon to select cells.

        Args:
            x: Column name of adata.obs used for the x-axis
            y: Column name of adata.obs used for the y-axis
            s: Size of the cells on the plot.
        """
        _, ax = plt.subplots()

        indices = np.arange(self.adata.n_obs)
        if max_cells_display is not None and max_cells_display < self.adata.n_obs:
            indices = np.random.choice(
                np.arange(self.adata.n_obs), size=max_cells_display, replace=False
            )

        x = self.adata.obs_vector(x)
        y = self.adata.obs_vector(y)
        xy = np.stack([x, y], axis=1)

        pts = ax.scatter(
            xy[indices, 0],
            xy[indices, 1],
            marker=".",
            rasterized=True,
            s=s,
        )

        self.selector = _SelectFromCollection(ax, pts, xy)

        log.info(
            f"Enclose cells within a polygon. Helper:\n    - Click on the plot to add a polygon vertex\n    - Press the 'esc' key to start a new polygon\n    - Try holding the 'ctrl' key to move a single vertex\n    - Once the polygon is finished and overlaid in red, you can close the window"
        )
        plt.show()

    def save_selection(self, key_added: str = "scyan_selected"):
        """Save the selected cells in `adata.obs[key_added]`.

        Args:
            key_added: Column name used to save the selected cells in `adata.obs`.
        """
        self.adata.obs[key_added] = "unselected"
        col_index = self.adata.obs.columns.get_loc(key_added)
        self.adata.obs.iloc[self.selector.ind, col_index] = "selected"
        self.adata.obs[key_added] = self.adata.obs[key_added].astype("category")

        self.selector.disconnect()
        log.info(
            f"Selected {len(self.selector.ind)} cells and saved the selection in adata.obs['{key_added}']"
        )

    def extract_adata(self) -> AnnData:
        """Returns an anndata objects whose cells where inside the polygon"""
        log.info(f"Selected {len(self.selector.ind)} cells")
        self.selector.disconnect()

        return self.adata[self.selector.ind]
