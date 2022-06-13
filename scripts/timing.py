import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from anndata import AnnData
import time

import scyan

from .utils import init_and_fit_model


@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    """Runs scyan on a dataset specified by the config/config.yaml file.
    It can be used for optuna hyperparameter search together with Weight&Biases to monitor the model.
    Note that using this file is not optional, you can use the library as such.

    Args:
        config (DictConfig): Hydra generated configuration (automatic)

    Returns:
        float: metric chosen by the config to be optimized for hyperparameter search, e.g. the loss
    """
    pl.seed_everything(config.seed)

    ### Instantiate everything
    adata, marker_pop_matrix = scyan.data.load(config.project.name)

    times, n_samples = [], []

    for n in [adata.n_obs] + [200000, 400000, 800000, 1600000]:
        if n > adata.n_obs:
            sampling_strategy = dict(Counter(np.random.choice(adata.obs.cell_type, n)))
            sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X, cell_type = sm.fit_resample(adata.X, adata.obs.cell_type.values)

            adata = AnnData(X=X, var=adata.var)
            adata.obs["cell_type"] = cell_type

        start = time.perf_counter()

        init_and_fit_model(adata, marker_pop_matrix, config)

        times.append(time.perf_counter() - start)
        n_samples.append(n)

        print("n:", n_samples)
        print("times:", times)


if __name__ == "__main__":
    main()
