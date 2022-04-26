import matplotlib.pyplot as plt
from typing import Callable, Tuple
import scanpy as sc
from pathlib import Path
from anndata import AnnData
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import flowio
from typing import Union, List
import torch
from torch import Tensor


def root_path() -> Path:
    """Gets the library root path

    Returns:
        Path: scyan library root path
    """
    return Path(__file__).parent.parent


def _wandb_plt_image(fun: Callable, figsize: Tuple[int, int] = [7, 5]):
    """Transforms a matplotlib figure into a wandb Image

    Args:
        fun (Callable): Function that makes the plot - do not plt.show().
        figsize (Tuple[int, int], optional): Matplotlib figure size. Defaults to [7, 5].

    Returns:
        wandb.Image: the wandb Image to be logged
    """

    from PIL import Image
    import wandb
    import io

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    fun()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    return wandb.Image(Image.open(img_buf))


def process_umap_latent(model, min_dist: float = 0.05):
    """Computes the UMAP based on Scyan's latent space

    Args:
        model (Scyan): Scyan model
        min_dist (float, optional): min_dist parameter for the UMAP. Defaults to 0.05.
    """
    model.adata.obsm["X_scyan"] = model().detach().numpy()
    sc.pp.neighbors(model.adata, use_rep="X_scyan")
    sc.tl.umap(model.adata, min_dist=min_dist)


def read_fcs(path: str) -> AnnData:
    """Reads a FCS file and returns an AnnData instance

    Args:
        path (str): Path to the FCS file

    Returns:
        AnnData: AnnData instance containing the FCS data
    """
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


def write_fcs(adata: AnnData, path: str) -> None:
    """Converts an adata instance into a FCS file

    Args:
        adata (AnnData): AnnData instance containing all the data
        path (str): Path where the FCS file will be saved
    """
    X = adata.X
    channel_names = list(adata.var_names)

    for column in adata.obs.columns:
        if is_numeric_dtype(adata.obs[column].dtype):
            X = np.c_[X, adata.obs[column].values]
            channel_names.append(column)

    for key in adata.obsm:
        X = np.concatenate((X, adata.obsm[key]), axis=1)
        channel_names += [f"{key}{i+1}" for i in range(adata.obsm[key].shape[1])]

    print(f"Found {len(channel_names)} channels: {', '.join(channel_names)}")

    with open(path, "wb") as f:
        flowio.create_fcs(X.flatten(), channel_names, f)


def _markers_to_indices(model, markers: List[str]) -> Tensor:
    """Transforms a list of markers into their corresponding indices in the marker-population matrix

    Args:
        model (Scyan): Scyan model
        markers (List[str]): List of marker names

    Returns:
        Tensor: Tensor of marker indices
    """
    return torch.tensor(
        [model.marker_pop_matrix.columns.get_loc(marker) for marker in markers],
        dtype=int,
    )


def _pops_to_indices(model, pops: List[str]) -> Tensor:
    """Transforms a list of populations into their corresponding indices in the marker-population matrix

    Args:
        model (Scyan): Scyan model
        pops (List[str]): List of population names

    Returns:
        Tensor: Tensor of population indices
    """
    return torch.tensor(
        [model.marker_pop_matrix.index.get_loc(pop) for pop in pops], dtype=int
    )


def _process_pop_sample(model, pop: Union[str, List[str], int, Tensor, None]):
    if isinstance(pop, str):
        return model.marker_pop_matrix.index.get_loc(pop)
    if isinstance(pop, list):
        return _pops_to_indices(model, pop)
    else:
        return pop


def _optional_show(f: Callable) -> Callable:
    """Decorator that shows a matplotlib figure if the provided 'show' argument is True"""

    def wrapper(*args, **kwargs):
        f(*args, **kwargs)
        if kwargs.get("show", True):
            plt.show()

    return wrapper
