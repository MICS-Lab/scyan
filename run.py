import wandb
import pandas as pd
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import dotenv
from typing import List
from pytorch_lightning import Callback

from scyan.utils import wandb_plt_image

dotenv.load_dotenv()


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    pl.seed_everything(config.seed)

    wandb.init(
        project=config.wandb.project,
        mode=config.wandb.mode,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )
    wandb_logger = WandbLogger()

    marker_pop_matrix = pd.read_csv(config.marker_pop_path, index_col=0)

    if config.wandb.save_marker_pop_matrix:
        wandb.log({"marker_pop_matrix": marker_pop_matrix})

    marker_pop_matrix = marker_pop_matrix.replace({0: -1})

    adata = sc.read_h5ad(config.data_path)
    adata = adata[:, marker_pop_matrix.index]

    model = hydra.utils.instantiate(
        config.model,
        adata=adata,
        marker_pop_matrix=marker_pop_matrix,
        _convert_="partial",
    )

    callbacks: List[Callback] = []
    if "callbacks" in config:
        callbacks = [
            hydra.utils.instantiate(cb_conf) for cb_conf in config.callbacks.values()
        ]

    trainer = hydra.utils.instantiate(
        config.trainer, logger=wandb_logger, callbacks=callbacks, _convert_="partial"
    )
    trainer.fit(model)

    model.predict()

    if config.wandb.save_umap:
        umap = wandb_plt_image(
            lambda: sc.pl.umap(
                adata,
                color="scyan_pop",
                show=False,
                palette={
                    "NA": "gray",
                    "B-Cell": "C10",
                    "PMN/Mono/Macro": "C1",
                    "Dendritic-Cell": "C11",
                    "Epcam+": "C7",
                    "Tcd4_naive/SCM": "C3",
                    "Tcd4_CM": "C4",
                    "Tcd4_EF/EM/activ": "C5",
                    "Tcd4_reg": "C6",
                    "Tcd8_naive/SCM": "C9",
                    "Tcd8_CM": "C8",
                    "Tcd8_EF/EM/activ": "C0",
                    "Tcd8_RM": "C2",
                },
            )
        )
        wandb.log({"umap": umap})

    metric = trainer.logged_metrics.get(config.optimized_metric)
    wandb.finish()

    return metric


if __name__ == "__main__":
    main()
