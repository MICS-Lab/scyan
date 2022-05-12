import hydra
from typing import List
from pytorch_lightning import Callback, Trainer
import wandb
import logging
import numpy as np
import scanpy as sc
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    silhouette_score,
)

import scyan
from scyan.model import Scyan

log = logging.getLogger(__name__)


def init_and_fit_model(adata, marker_pop_matrix, config, wandb_logger=None):
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
        config.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
        _convert_="partial",
    )

    model.fit(trainer=trainer)
    model.predict()

    return model


def classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)

    return {"Accuracy": accuracy, "F1-score": f1, "Kappa": kappa}


def compute_metrics(model, config, scyan_pop_key="scyan_pop"):
    if config.project.get("label", None):
        y_true = model.adata.obs[config.project.label]
        y_pred = model.adata.obs[scyan_pop_key]
        metrics_dict = classification_metrics(y_true, y_pred)
    else:
        log.info("No label provided. The classification metrics are not computed.")
        metrics_dict = {}

    X, labels = model.adata.X, model.adata.obs[scyan_pop_key]
    labels_count = len(set(labels.values))
    metrics_dict["Labels count"] = labels_count

    if config.compute_silhouette_score:
        if config.force_all_populations and (
            labels_count < len(model.marker_pop_matrix.index)
        ):
            log.warning("Not all pops are present. Setting silhouette metric to 0.")
            silhouette = 0
        else:
            silhouette = silhouette_score(X, labels)

        metrics_dict["Penalized Silhouette"] = silhouette + labels_count

    print("\nComputed metrics:")
    for name, value in metrics_dict.items():
        print(f"{name}: {value:.4f}")
        wandb.run.summary[name] = value

    return metrics_dict


def metric_to_optimize(all_metrics, config):
    return np.array(all_metrics[config.optimized_metric]).mean()


def compute_umaps(model, config, scyan_pop_key="scyan_pop"):
    palette = model.adata.uns.get("palette", None)  # Get color palette if existing

    if config.wandb.mode != "disabled" and config.wandb.save_umap:
        wandb.log(
            {
                "umap": scyan.utils._wandb_plt_image(
                    lambda: sc.pl.umap(
                        model.adata, color=scyan_pop_key, show=False, palette=palette
                    )
                )
            }
        )

    if config.wandb.mode != "disabled" and config.wandb.save_umap_latent_space:
        scyan.utils.process_umap_latent(model)
        wandb.log(
            {
                "umap_latent_space": scyan.utils._wandb_plt_image(
                    lambda: sc.pl.umap(
                        model.adata,
                        color=[scyan_pop_key]
                        + model.categorical_covariate_keys
                        + model.continuous_covariate_keys,
                        show=False,
                    )
                )
            }
        )
