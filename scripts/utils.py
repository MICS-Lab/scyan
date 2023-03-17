import logging
from typing import List, Optional

import hydra
import numpy as np
import numpy.typing as npt
import pandas as pd
import wandb
from anndata import AnnData
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn import metrics

import scyan
from scyan.model import Scyan

log = logging.getLogger(__name__)


def init_and_fit_model(
    adata: AnnData,
    table: pd,
    config: DictConfig,
    wandb_logger: Optional[WandbLogger] = None,
) -> Scyan:
    """Initialize Scyan with the Hydra config, fit the model and run predictions.
    NB: if not using Hydra, then do **not** use this function.

    Args:
        adata: `AnnData` object containing the FCS data.
        table: Dataframe representing the biological knowledge about markers and populations.
        config: Hydra generated configuration.
        wandb_logger: Weight & Biases logger.

    Returns:
        The model.
    """
    model: Scyan = hydra.utils.instantiate(
        config.model,
        adata=adata,
        table=table,
        continuous_covariates=config.project.get("continuous_covariates", []),
        categorical_covariates=config.project.get("categorical_covariates", []),
        batch_key=config.project.get("batch_key"),
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
    model.predict(log_prob_th=-np.inf)  # Prediction on all cells for hyperoptimisation

    return model


def classification_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict:
    """Compute classification metrics.

    Args:
        y_true: True annotations.
        y_pred: Predictions.

    Returns:
        A dict of metrics.
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average="macro")
    balanced_acc = metrics.balanced_accuracy_score(y_true, y_pred)

    return {"Accuracy": accuracy, "F1-Score": f1, "Balanced Accuracy": balanced_acc}


def compute_metrics(model: Scyan, config: DictConfig, key: str = "scyan_pop") -> dict:
    """Compute model metrics.

    Args:
        model: Scyan model.
        config: Hydra generated configuration.
        key: Key in `adata.obs` where predictions are saved.

    Returns:
        A dict of metrics.
    """
    if config.project.get("label", None):
        y_true = model.adata.obs[config.project.label]
        y_pred = model.adata.obs[key].fillna("")
        metrics_dict = classification_metrics(y_true, y_pred)
    else:
        log.info("No label provided. The classification metrics are not computed.")
        metrics_dict = {}

    X, labels = model.x.numpy(force=True), model.adata.obs[key]
    X, labels = X[~labels.isna()], labels[~labels.isna()]

    n_missing_pop = len(model.pop_names) - len(set(labels.values))
    metrics_dict["Number of missing pop"] = n_missing_pop

    dbs = metrics.davies_bouldin_score(X, labels)
    metrics_dict["Davies-Bouldin score"] = dbs

    p = labels.value_counts(normalize=True).values
    neg_log_dir = -np.log(p).sum()
    metrics_dict["Neg log Dirichlet"] = neg_log_dir

    if model._corr_mode:
        from .lisi import compute_lisi

        corr = model.batch_effect_correction()
        model.adata.obsm["scyan_corrected"] = corr.numpy(force=True)
        lisi = compute_lisi(model.adata, model.hparams.batch_key, "scyan_corrected")

        metrics_dict["iLISI"] = lisi
        metrics_dict["Heuristic"] = (n_missing_pop + 1) * dbs * neg_log_dir / lisi
    else:
        metrics_dict["Heuristic"] = (n_missing_pop + 1) * dbs * neg_log_dir

    print("\n-- Run metrics --")
    for name, value in metrics_dict.items():
        print(f"{name}: {value:.4f}")
        if config.wandb.mode != "disabled":
            wandb.run.summary[name] = value
    print()

    return metrics_dict


def metric_to_optimize(all_metrics: dict, config: DictConfig) -> float:
    """Find and average the metric that have to be hyperoptimized.

    Args:
        all_metrics: Dict of metrics.
        config: Hydra generated configuration.

    Returns:
        The averaged metric to optimize.
    """
    if len(all_metrics[config.optimized_metric]):
        return np.array(all_metrics[config.optimized_metric]).mean()

    log.info(
        f"Metric used for hyperparamsearch ({config.optimized_metric}) was not computed. Returning 0 instead."
    )
    return 0


def compute_umap(model: Scyan, config: DictConfig, key: str = "scyan_pop") -> None:
    """Log a UMAP with Weight & Biases.

    Args:
        model: Scyan model.
        config: Hydra generated configuration.
        key: Key in `adata.obs` where predictions are saved.
    """
    palette = model.adata.uns.get("palette", None)  # Get color palette if existing

    if config.wandb.mode != "disabled" and config.wandb.save_umap:
        wandb.log(
            {
                "umap": scyan.utils._wandb_plt_image(
                    lambda: scyan.plot.umap(
                        model.adata, color=key, show=False, palette=palette
                    )
                )
            }
        )
