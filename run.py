import wandb
import pandas as pd
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import dotenv

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
    wandb.log({"marker_pop_matrix": marker_pop_matrix})

    marker_pop_matrix = marker_pop_matrix.replace({0: -1})

    adata = sc.read_h5ad(config.data_path)
    adata = adata[:, marker_pop_matrix.index]

    model = hydra.utils.instantiate(
        config.model, marker_pop_matrix=marker_pop_matrix, _convert_="partial"
    )

    trainer = hydra.utils.instantiate(
        config.trainer, logger=wandb_logger, _convert_="partial"
    )
    trainer.fit(model)

    predictions = model.predict(adata)

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
    loss = trainer.logged_metrics["loss"].item()

    wandb.finish()

    return loss


if __name__ == "__main__":
    main()
