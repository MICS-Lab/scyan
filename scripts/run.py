from collections import defaultdict

import hydra
import numpy as np
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

import scyan
from scyan.model import Scyan

from . import utils


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> float:
    """Run scyan on a dataset specified by the config/config.yaml file.
    It can be used for optuna hyperparameter search together with Weight_&_Biases to monitor the model.
    NB: using this file is optional. If you don't need hyperoptimization and monitoring then use the library directly.

    Args:
        config: Hydra generated configuration (automatic).

    Returns:
        Metric chosen by the config to be optimized for hyperparameter search, e.g. the loss.
    """
    adata, table = scyan.data.load(
        config.project.name,
        version=config.project.get("version", "default"),
        table=config.project.get("table", "default"),
    )

    all_metrics = defaultdict(list)

    for i in range(config.n_run):
        pl.seed_everything(i)

        ### Init Weight & Biases (if config.wandb.mode="online")
        if config.wandb.mode != "disabled":
            wandb.init(
                project=config.project.wandb_project_name,
                mode=config.wandb.mode,
                config=OmegaConf.to_container(
                    config, resolve=True, throw_on_missing=True
                ),
                reinit=True,
            )
            wandb_logger = WandbLogger()
        else:
            wandb_logger = None

        model: Scyan = utils.init_and_fit_model(adata, table, config, wandb_logger)

        if config.save_predictions:
            adata.obs[["scyan_pop"]].reset_index().to_csv(
                f"pred_{config.project.name}_{i}.csv"
            )

        # Runs only when W&B is enabled and when save UMAP is True
        utils.compute_umap(model, config)

        metrics_dict = utils.compute_metrics(model, config)

        for name, value in metrics_dict.items():
            all_metrics[name].append(value)

        if config.wandb.mode != "disabled":
            wandb.finish()

    print(f"--- Finished {config.n_run} Run(s) ---\n")
    for name, values in all_metrics.items():
        values = np.array(values)
        print(f"{name}: {values.mean():.4f} ± {values.std():.4f}\n  {values}\n")

    return utils.metric_to_optimize(all_metrics, config)


if __name__ == "__main__":
    main()
