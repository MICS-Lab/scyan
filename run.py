import wandb
import pandas as pd
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import dotenv
from typing import List
from pytorch_lightning import Callback, Trainer
from anndata import AnnData

from scyan.model import Scyan
from scyan.utils import wandb_plt_image, process_umap_latent

dotenv.load_dotenv()


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    pl.seed_everything(config.seed)

    ### Init Weight & Biases (if config.wandb.mode="online")
    wandb.init(
        project=config.project.name,
        mode=config.wandb.mode,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )
    wandb_logger = WandbLogger()

    ### Instantiate everything
    marker_pop_matrix: pd.DataFrame = pd.read_csv(config.marker_pop_path, index_col=0)
    adata: AnnData = sc.read_h5ad(config.data_path)

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
    palette = config.project.get("palette")
    covariate_keys = model.categorical_covariate_keys + model.continuous_covariate_keys

    if config.wandb.save_umap:
        wandb.log(
            {
                "umap": wandb_plt_image(
                    lambda: sc.pl.umap(
                        model.adata, color="scyan_pop", show=False, palette=palette
                    )
                )
            }
        )

    if config.wandb.save_umap_latent_space:
        process_umap_latent(model)
        wandb.log(
            {
                "umap_latent_space": wandb_plt_image(
                    lambda: sc.pl.umap(
                        model.adata,
                        color=["scyan_pop"] + covariate_keys,
                        show=False,
                    )
                )
            }
        )

    ### Finishing
    metric = trainer.logged_metrics.get(config.optimized_metric)
    wandb.finish()

    return metric


if __name__ == "__main__":
    main()
