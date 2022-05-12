import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

import scyan
from scyan.model import Scyan

from .utils import init_and_fit_model, classification_metrics


@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig) -> float:
    """Runs scyan on a dataset specified by the config/config.yaml file.
    It can be used for optuna hyperparameter search together with Weight&Biases to monitor the model.
    Note that using this file is not optional, you can use the library as such.

    Args:
        config (DictConfig): Hydra generated configuration (automatic)

    Returns:
        float: metric chosen by the config to be optimized for hyperparameter search, e.g. the loss
    """
    adata, marker_pop_matrix = scyan.data.load(config.project.name)
    scores = []

    for i in range(config.n_run_testing):
        pl.seed_everything(i)

        ### Init Weight & Biases (if config.wandb.mode="online")
        wandb.init(
            project=config.project.wandb_project_name,
            mode=config.wandb.mode,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            reinit=True,
        )
        wandb_logger = WandbLogger()

        model: Scyan = init_and_fit_model(adata, marker_pop_matrix, config, wandb_logger)

        y_true = model.adata.obs[config.project.label]
        y_pred = model.adata.obs["scyan_pop"]

        accuracy, f1, kappa = classification_metrics(y_true, y_pred)

        print(f"\nClassification metrics at run {i}:")
        print(f"Accuracy: {accuracy:.4f}\nF1-score: {f1:.4f}\nKappa: {kappa:.4f}")
        wandb.run.summary["accuracy"] = accuracy
        wandb.run.summary["f1_score"] = f1
        wandb.run.summary["kappa"] = kappa

        scores.append([accuracy, f1, kappa])

        wandb.finish()

    print("\n--- FINISHED TESTING ---")
    scores = np.array(scores)
    print("Scores (acc, f1, kappa):\n", scores)
    print("\nMeans (acc, f1, kappa):\n", scores.mean(axis=0))
    print("\nStds (acc, f1, kappa):\n", scores.std(axis=0))


if __name__ == "__main__":
    main()
