import time

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

import scyan

from . import utils


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    """Run scyan on a dataset specified by the config/config.yaml with different number of cells.
    NB: the only purpose of this file is to time the model on POISED.

    Args:
        config: Hydra generated configuration (automatic).
    """
    pl.seed_everything(config.seed)

    adata, table = scyan.data.load(config.project.name)
    n_obs = config.project.get("n_obs", None)

    if n_obs is not None:
        print(f"Undersampling cells to N={n_obs}...")
        scyan.preprocess.subsample(adata, n_obs=n_obs)

    start = time.perf_counter()
    utils.init_and_fit_model(adata, table, config)

    print(f"Run in {time.perf_counter() - start} seconds on {adata.n_obs} cells.")


if __name__ == "__main__":
    main()
