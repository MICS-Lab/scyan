import torch
from torch import Tensor
from torch import nn
from torch import distributions
import pytorch_lightning as pl
import torch.nn.functional as F
from anndata import AnnData
import pandas as pd
from typing import Tuple, Union

from scyan.module import RealNVP


class Scyan(pl.LightningModule):
    eps: float = 1e-20

    def __init__(
        self,
        adata: AnnData,
        marker_pop_matrix: pd.DataFrame,
        hidden_size: int = 64,
        n_hidden_layers: int = 1,
        alpha_dirichlet: float = 1.02,
        n_layers: int = 6,
        prior_var: float = 0.2,
        lr: float = 5e-3,
        batch_size: int = 16384,
    ):
        super().__init__()
        self.adata = adata
        self.marker_pop_matrix = marker_pop_matrix
        self.save_hyperparameters(ignore=["adata", "marker_pop_matrix"])

        self.x = torch.tensor(adata.X)
        self.n_markers, self.n_pop = marker_pop_matrix.shape
        self.rho = torch.tensor(marker_pop_matrix.values.T[None, :, :])

        self.mask = nn.Parameter(torch.arange(self.n_markers) % 2, requires_grad=False)

        self.prior_h = distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.n_markers),
            self.hparams.prior_var * torch.eye(self.n_markers),
        )
        self.prior_pi = distributions.dirichlet.Dirichlet(
            torch.tensor([self.hparams.alpha_dirichlet] * self.n_pop)
        )

        self.pi_logit = nn.Parameter(torch.randn(self.n_pop))
        self.log_softmax = torch.nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=1)

        self.real_nvp = RealNVP(
            self.n_markers,
            self.hparams.hidden_size,
            self.hparams.n_hidden_layers,
            self.mask,
            self.hparams.n_layers,
        )
        self.mse = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.real_nvp(x)

    def inverse(self, z: Tensor) -> Tensor:
        return self.real_nvp.inverse(z)

    @property
    def prior_z(self) -> distributions.distribution.Distribution:
        return distributions.categorical.Categorical(F.softmax(self.pi_logit, dim=0))

    @torch.no_grad()
    def sample(self, n_samples: int) -> Tuple[Tensor, Tensor]:
        sample_shape = torch.Size([n_samples])
        z = self.prior_z.sample(sample_shape)
        h = self.prior_h.sample(sample_shape) + self.rho[0, z]
        x = self.inverse(h).detach()
        return x, z

    def compute_probabilities(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        log_pi = self.log_softmax(self.pi_logit)
        log_prob_pi = self.prior_pi.log_prob(torch.exp(log_pi) + self.eps)

        h, ldj_sum = self(x)
        log_probs = self.prior_h.log_prob(h[:, None, :] - self.rho) + log_pi
        probs = self.softmax(log_probs).detach()
        return probs, log_probs, ldj_sum, log_prob_pi

    def training_step(self, x: Tensor, _):
        probs, log_probs, ldj_sum, log_prob_pi = self.compute_probabilities(x)
        loss = -(ldj_sum.mean() + log_prob_pi + (probs * log_probs).sum() / x.shape[0])
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    @torch.no_grad()
    def predict(
        self, x: Union[Tensor, None] = None, key_added: str = "scyan_pop"
    ) -> pd.Series:
        df = self.predict_proba(x=x)
        populations = df.idxmax(axis=1).astype("category")

        if key_added:
            self.adata.obs[key_added] = populations.values

        return populations

    @torch.no_grad()
    def predict_proba(self, x: Union[Tensor, None] = None) -> pd.DataFrame:
        predictions, *_ = self.compute_probabilities(self.x if x is None else x)
        return pd.DataFrame(predictions.numpy(), columns=self.marker_pop_matrix.columns)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.x, batch_size=self.hparams.batch_size)
