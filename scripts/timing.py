import time

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

import scyan

from . import utils


@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    """Runs scyan on a dataset specified by the config/config.yaml with different number of cells.
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

    correction_mode = config.project.get("batch_key") is not None

    for n in [adata.n_obs] + [200_000, 400_000, 800_000, 1_600_000, 3_200_000, 6_400_000]:
        if n > adata.n_obs:
            adata = utils.oversample(adata, n, correction_mode)

        start = time.perf_counter()

        utils.init_and_fit_model(adata, marker_pop_matrix, config)

        times.append(time.perf_counter() - start)
        n_samples.append(n)

        print("Num samples:", n_samples)
        print("Times:", times)


if __name__ == "__main__":
    main()
