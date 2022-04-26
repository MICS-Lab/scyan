import wandb
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List
from pytorch_lightning import Callback, Trainer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)
import numpy as np

import scyan
from scyan.model import Scyan


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

    ### Instantiate everything
    adata, marker_pop_matrix = scyan.data.load(config.project.name)

    scores = []
    i = 0
    while len(scores) < 20:
        ### Init Weight & Biases (if config.wandb.mode="online")
        wandb.init(
            project=config.project.wandb_project_name,
            mode=config.wandb.mode,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            reinit=True,
        )
        wandb_logger = WandbLogger()

        i += 1
        pl.seed_everything(i)

        model: Scyan = hydra.utils.instantiate(
            config.model,
            adata=adata,
            marker_pop_matrix=marker_pop_matrix,
            continuous_covariate_keys=config.project.get(
                "continuous_covariate_keys", []
            ),
            categorical_covariate_keys=config.project.get(
                "categorical_covariate_keys", []
            ),
            _convert_="partial",
        )

        callbacks: List[Callback] = (
            [hydra.utils.instantiate(cb_conf) for cb_conf in config.callbacks.values()]
            if "callbacks" in config
            else []
        )

        trainer: Trainer = hydra.utils.instantiate(
            config.trainer,
            logger=wandb_logger,
            callbacks=callbacks,
            _convert_="partial",
        )

        ### Training
        model.fit(trainer=trainer)
        model.predict()
        model.knn_predict()

        y_true = model.adata.obs[config.project.label]
        y_pred = model.adata.obs.scyan_knn_pop

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        kappa = cohen_kappa_score(y_true, y_pred)

        print("\nClassification metrics:")
        print(f"Accuracy: {accuracy:.4f}\nF1-score: {f1:.4f}\nKappa: {kappa:.4f}")
        wandb.run.summary["accuracy"] = accuracy
        wandb.run.summary["f1_score"] = f1
        wandb.run.summary["kappa"] = kappa

        labels = model.adata.obs.scyan_knn_pop

        if len(set(labels.values)) == len(model.marker_pop_matrix.index):
            scores.append([accuracy, f1, kappa])
            wandb.run.summary["success"] = True
        else:
            print(
                "Warning: not all populations are present. Setting classification metrics to 0."
            )
            wandb.run.summary["success"] = False

        wandb.finish()

    scores = np.array(scores)
    print("FINISH")
    print(scores)
    print(scores.mean(axis=0))
    print(scores.std(axis=0))


if __name__ == "__main__":
    main()
