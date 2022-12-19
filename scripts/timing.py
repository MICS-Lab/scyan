import time

import hydra
import pytorch_lightning as pl
import scanpy as sc
from omegaconf import DictConfig

import scyan

from . import utils


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    """Run scyan on a dataset specified by the config/config.yaml with different number of cells.
    NB: the only purpose of this file is to time the model on AML.

    Args:
        config: Hydra generated configuration (automatic).

    Returns:
        Metric chosen by the config to be optimized for hyperparameter search, e.g. the loss.
    """
    pl.seed_everything(config.seed)

    ### Instantiate everything
    adata, marker_pop_matrix = scyan.data.load(config.project.name)

    times, n_samples = [], []

    for n_obs in [4_000_000, 2_000_000, 1_000_000, 500_000, 250_000, 125_000]:
        print(f"Undersampling cells to N={n_obs}...")
        sc.pp.subsample(adata, n_obs=n_obs)

        start = time.perf_counter()

        utils.init_and_fit_model(adata, marker_pop_matrix, config)

        times.append(time.perf_counter() - start)
        n_samples.append(n_obs)

        print("Num samples:", n_samples)
        print("Times:", times)


if __name__ == "__main__":
    main()
