import wandb
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import Callable, Tuple
import scanpy as sc


def wandb_plt_image(fun: Callable, figsize: Tuple[int, int] = [7, 5]) -> wandb.Image:
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


def process_umap_latent(scyan, min_dist=0.05):
    scyan.adata.obsm["X_scyan"] = scyan().detach().numpy()
    sc.pp.neighbors(scyan.adata, use_rep="X_scyan")
    sc.tl.umap(scyan.adata, min_dist=min_dist)
