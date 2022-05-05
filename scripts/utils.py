import hydra
from typing import List
from pytorch_lightning import Callback, Trainer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    silhouette_score,
    davies_bouldin_score,
)

from scyan.model import Scyan


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
    if config.run_knn:
        model.knn_predict()

    return model


def classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)

    return accuracy, f1, kappa


def clustering_metrics(X, labels):
    silhouette = silhouette_score(X, labels)
    dbs = davies_bouldin_score(X, labels)
    return silhouette, dbs
