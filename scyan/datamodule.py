import torch
import pytorch_lightning as pl
from anndata import AnnData


class AdataDataModule(pl.LightningDataModule):
    def __init__(self, adata: AnnData, batch_size: int = 16384):
        super().__init__()
        self.adata = adata
        self.batch_size = batch_size
        self.X = torch.Tensor(self.adata.X)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.X, batch_size=self.batch_size)
