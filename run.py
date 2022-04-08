import wandb
import pandas as pd
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List
from pytorch_lightning import Callback, Trainer
from anndata import AnnData
from sklearn.metrics import accuracy_score, cohen_kappa_score

import scyan
from scyan.model import Scyan
from scyan.utils import _wandb_plt_image, process_umap_latent


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
    trainer.fit(model)
    model.predict()

    ### Compute UMAP after training
    palette = adata.uns.get("palette", None)  # Get color palette if existing
    covariate_keys = model.categorical_covariate_keys + model.continuous_covariate_keys

    if config.wandb.mode != "disabled" and config.wandb.save_umap:
        wandb.log(
            {
                "umap": _wandb_plt_image(
                    lambda: sc.pl.umap(
                        model.adata, color="scyan_pop", show=False, palette=palette
                    )
                )
            }
        )

    if config.wandb.mode != "disabled" and config.wandb.save_umap_latent_space:
        process_umap_latent(model)
        wandb.log(
            {
                "umap_latent_space": _wandb_plt_image(
                    lambda: sc.pl.umap(
                        model.adata,
                        color=["scyan_pop"] + covariate_keys,
                        show=False,
                    )
                )
            }
        )

    ### Print model accuracy and cohen's kappa if there are some known labels
    if config.project.get("label", None):
        model.predict()
        model.knn_predict()

        print(
            f"\nModel accuracy: {accuracy_score(model.adata.obs[config.project.label], model.adata.obs.scyan_knn_pop):.4f}"
        )
        print(
            f"Model Cohen's kappa: {cohen_kappa_score(model.adata.obs[config.project.label], model.adata.obs.scyan_knn_pop):.4f}"
        )

    ### Finishing
    metric = trainer.logged_metrics.get(config.optimized_metric)
    wandb.finish()

    return metric


if __name__ == "__main__":
    main()
