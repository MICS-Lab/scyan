import hydra
from typing import List
from pytorch_lightning import Callback, Trainer
import wandb
import logging
import numpy as np
import scanpy as sc
from sklearn import metrics

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
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average="macro")
    balanced_acc = metrics.balanced_accuracy_score(y_true, y_pred)

    return {"Accuracy": accuracy, "F1-Score": f1, "Balanced Accuracy": balanced_acc}


def compute_metrics(model, config, scyan_pop_key="scyan_pop"):
    if config.project.get("label", None):
        y_true = model.adata.obs[config.project.label]
        y_pred = model.adata.obs[scyan_pop_key]
        metrics_dict = classification_metrics(y_true, y_pred)
    else:
        log.info("No label provided. The classification metrics are not computed.")
        metrics_dict = {}

    X, labels = model.adata.X, model.adata.obs[scyan_pop_key]

    n_missing_pop = len(model.marker_pop_matrix.index) - len(set(labels.values))
    metrics_dict["Number of missing pop"] = n_missing_pop

    dbs = metrics.davies_bouldin_score(X, labels)
    metrics_dict["Davies-Bouldin score"] = dbs

    p = labels.value_counts(normalize=True).values
    neg_log_dir = -np.log(p).sum()
    metrics_dict["Neg log Dirichlet"] = neg_log_dir

    metrics_dict["Heuristic"] = (n_missing_pop + 1) * dbs * neg_log_dir

    print("\n-- Run metrics --")
    for name, value in metrics_dict.items():
        print(f"{name}: {value:.4f}")
        wandb.run.summary[name] = value
    print()

    return metrics_dict


def metric_to_optimize(all_metrics, config):
    if len(all_metrics[config.optimized_metric]):
        return np.array(all_metrics[config.optimized_metric]).mean()

    log.info(
        f"Metric used for hyperparamsearch ({config.optimized_metric}) was not computed. Returning 0 instead."
    )
    return 0


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
