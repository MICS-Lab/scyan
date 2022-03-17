import torch
from torch import Tensor
from torch import nn
from torch import distributions
import torch.nn.functional as F
from typing import Tuple
import pytorch_lightning as pl

from .real_nvp import RealNVP


class ScyanModule(pl.LightningModule):
    eps: float = 1e-20

    def __init__(
        self,
        rho: Tensor,
        n_covariates: int,
        hidden_size: int,
        n_hidden_layers: int,
        alpha_dirichlet: float,
        n_layers: int,
        prior_std: float,
        lr: float,
        batch_size: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["rho", "n_covariates"])

        self.n_pop, self.n_markers = rho.shape
        self.rho = nn.Parameter(rho, requires_grad=False)

        self.rho_mask = self.rho.isnan()
        self.rho[self.rho_mask] = 0

        self.prior_h = distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.n_markers),
            prior_std ** 2 * torch.eye(self.n_markers),
        )
        self.prior_pi = distributions.dirichlet.Dirichlet(
            torch.tensor([self.hparams.alpha_dirichlet] * self.n_pop)
        )

        self.pi_logit = nn.Parameter(torch.randn(self.n_pop))

        self.real_nvp = RealNVP(
            self.n_markers + n_covariates,
            self.hparams.hidden_size,
            self.n_markers,
            self.hparams.n_hidden_layers,
            self.hparams.n_layers,
        )

    def forward(self, x: Tensor, covariates: Tensor) -> Tuple[Tensor, Tensor]:
        return self.real_nvp(x, covariates)

    def inverse(self, z: Tensor, covariates: Tensor) -> Tensor:
        return self.real_nvp.inverse(z, covariates)

    @property
    def prior_z(self) -> distributions.distribution.Distribution:
        return distributions.categorical.Categorical(self.pi)

    @property
    def log_pi(self) -> Tensor:
        return torch.log_softmax(self.pi_logit, dim=0)

    @property
    def pi(self) -> Tensor:
        return torch.exp(self.log_pi)

    @torch.no_grad()
    def sample(self, n_samples: int, covariates: Tensor) -> Tuple[Tensor, Tensor]:
        sample_shape = torch.Size([n_samples])
        z = self.prior_z.sample(sample_shape)
        h = self.prior_h.sample(sample_shape) + self.rho[z]  # TODO: use kde for NaN
        x = self.inverse(h, covariates).detach()
        return x, z

    def compute_probabilities(
        self, x: Tensor, covariates: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        u, _, ldj_sum = self(x, covariates)

        h = u[:, None, :] - self.rho[None, ...]
        h[:, self.rho_mask] = 0

        log_probs = self.prior_h.log_prob(h) + self.log_pi
        probs = torch.softmax(log_probs, dim=1)
        log_prob_pi = self.prior_pi.log_prob(self.pi + self.eps)

        return probs, log_probs, ldj_sum, log_prob_pi

    def loss(self, x: Tensor, covariates: Tensor) -> Tensor:
        probs, log_probs, ldj_sum, log_prob_pi = self.compute_probabilities(
            x, covariates
        )
        return -(ldj_sum + torch.logsumexp(log_probs, dim=1) + log_prob_pi).mean()
