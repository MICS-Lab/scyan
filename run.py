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
    silhouette_score,
    davies_bouldin_score,
)
import numpy as np

import scyan
from scyan.model import Scyan
from scyan.utils import _wandb_plt_image


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig) -> float:
    """Runs scyan on a dataset specified by the config/config.yaml file.
    It can be used for optuna hyperparameter search together with Weight&Biases to monitor the model.
    Note that using this file is not optional, you can use the library as such.

    Args:
        config (DictConfig): Hydra generated configuration (automatic)

    Returns:
        float: metric chosen by the config to be optimized for hyperparameter search, e.g. the loss
    """
    pl.seed_everything(config.seed)

    ### Init Weight & Biases (if config.wandb.mode="online")
    wandb.init(
        project=config.project.name,
        mode=config.wandb.mode,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )
    wandb_logger = WandbLogger()

    ### Instantiate everything
    adata, marker_pop_matrix = scyan.data.load(config.project.name)

    model: Scyan = hydra.utils.instantiate(
        config.model,
        adata=adata,
        marker_pop_matrix=marker_pop_matrix,
        continuous_covariate_keys=config.project.get("continuous_covariate_keys", []),
        categorical_covariate_keys=config.project.get("categorical_covariate_keys", []),
        _convert_="partial",
    )

    callbacks: List[Callback] = (
        [hydra.utils.instantiate(cb_conf) for cb_conf in config.callbacks.values()]
        if "callbacks" in config
        else []
    )

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=wandb_logger, callbacks=callbacks, _convert_="partial"
    )

    ### Training
    model.fit(trainer=trainer)
    model.predict()
    model.knn_predict()

    ### Compute UMAP after training (if wandb is enabled and if the save parameters are True)
    palette = adata.uns.get("palette", None)  # Get color palette if existing

    if config.wandb.mode != "disabled" and config.wandb.save_umap:
        wandb.log(
            {
                "umap": _wandb_plt_image(
                    lambda: sc.pl.umap(
                        model.adata, color="scyan_knn_pop", show=False, palette=palette
                    )
                )
            }
        )

    if config.wandb.mode != "disabled" and config.wandb.save_umap_latent_space:
        scyan.utils.process_umap_latent(model)
        wandb.log(
            {
                "umap_latent_space": _wandb_plt_image(
                    lambda: sc.pl.umap(
                        model.adata,
                        color=["scyan_knn_pop"]
                        + model.categorical_covariate_keys
                        + model.continuous_covariate_keys,
                        show=False,
                    )
                )
            }
        )

    ### Printing model accuracy and cohen's kappa (if there are some known labels)
    if config.project.get("label", None):
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

        X, labels = model.adata.X, model.adata.obs.scyan_knn_pop
        silhouette = silhouette_score(X, labels)
        dbs = davies_bouldin_score(X, labels)

        print(f"\nClustering metrics:")
        print(f"Silhouette score: {silhouette:.4f}\nDavies Bouldin Score: {dbs:.4f}")
        wandb.run.summary["silhouette_score"] = silhouette
        wandb.run.summary["dbs"] = dbs

    ### Finishing
    metric = trainer.logged_metrics.get(config.optimized_metric)
    wandb.finish()

    return metric


if __name__ == "__main__":
    main()
