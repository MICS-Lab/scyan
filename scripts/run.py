import wandb
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

import scyan
from scyan.model import Scyan
from scyan.utils import _wandb_plt_image

from .utils import init_and_fit_model, classification_metrics, clustering_metrics


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
    pl.seed_everything(config.seed)

    ### Init Weight & Biases (if config.wandb.mode="online")
    wandb.init(
        project=config.project.wandb_project_name,
        mode=config.wandb.mode,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )
    wandb_logger = WandbLogger()

    adata, marker_pop_matrix = scyan.data.load(config.project.name)
    model: Scyan = init_and_fit_model(adata, marker_pop_matrix, config, wandb_logger)

    ### Compute UMAP after training (if wandb is enabled and if the save parameters are True)
    palette = adata.uns.get("palette", None)  # Get color palette if existing
    scyan_pop_key = "scyan_knn_pop" if config.run_knn else "scyan_pop"

    if config.wandb.mode != "disabled" and config.wandb.save_umap:
        wandb.log(
            {
                "umap": _wandb_plt_image(
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
                "umap_latent_space": _wandb_plt_image(
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

    ### Printing model metrics (if we have known labels)
    if config.project.get("label", None):
        y_true = model.adata.obs[config.project.label]
        y_pred = model.adata.obs[scyan_pop_key]

        print("\nClassification metrics:")
        accuracy, f1, kappa = classification_metrics(y_true, y_pred)

        print(f"Accuracy: {accuracy:.4f}\nF1-score: {f1:.4f}\nKappa: {kappa:.4f}")
        wandb.run.summary["accuracy"] = accuracy
        wandb.run.summary["f1_score"] = f1
        wandb.run.summary["kappa"] = kappa

        print(f"\nClustering metrics:")
        X, labels = model.adata.X, model.adata.obs[scyan_pop_key]

        wandb.run.summary["n_labels"] = len(set(labels.values))
        if config.force_all_populations and (
            len(set(labels.values)) < len(model.marker_pop_matrix.index)
        ):
            print("Warning: not all pops are present. Setting clustering metrics to 0.")
            silhouette, dbs = 0, 0
        else:
            silhouette, dbs = clustering_metrics(X, labels)

        print(f"Silhouette score: {silhouette:.4f}\nDavies Bouldin Score: {dbs:.4f}")
        label_penalty = len(set(labels.values)) - model.n_pops
        wandb.run.summary["silhouette_score"] = silhouette + label_penalty
        wandb.run.summary["dbs"] = dbs + label_penalty

    ### Finishing
    metric = wandb.run.summary[config.optimized_metric]
    wandb.finish()

    return metric


if __name__ == "__main__":
    main()
