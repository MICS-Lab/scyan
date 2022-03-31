import matplotlib.pyplot as plt
from typing import Callable, Tuple
import scanpy as sc
from pathlib import Path
from anndata import AnnData
import numpy as np
import pandas as pd
import flowio


def root_path() -> Path:
    return Path(__file__).parent.parent


def wandb_plt_image(fun: Callable, figsize: Tuple[int, int] = [7, 5]):
    from PIL import Image
    import wandb
    import io

    """Transform a matplotlib figure into a wandb Image

    Args:
        fun: function that makes the plot - do not plt.show().
        figsize: Matplotlib figure size

    Returns:
        wandb.Image: the wandb Image to be logged
    """
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    fun()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    return wandb.Image(Image.open(img_buf))


def process_umap_latent(model, min_dist=0.05):
    model.adata.obsm["X_scyan"] = model().detach().numpy()
    sc.pp.neighbors(model.adata, use_rep="X_scyan")
    sc.tl.umap(model.adata, min_dist=min_dist)


def read_fcs(path: str) -> AnnData:
    fcs_data = flowio.FlowData(str(path))
    data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))

    names = np.array(
        [[value["PnN"], value.get("PnS", None)] for value in fcs_data.channels.values()]
    )
    is_marker = names[:, 1] != None

    X = data[:, is_marker]
    var = pd.DataFrame(index=names[is_marker, 1])
    obs = pd.DataFrame(
        data=data[:, ~is_marker],
        columns=names[~is_marker, 0],
        index=range(data.shape[0]),
    )

    return AnnData(X=X, var=var, obs=obs)
