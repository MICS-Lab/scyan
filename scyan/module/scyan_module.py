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
        prior_std_nan: float,
        lr: float,
        batch_size: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["rho", "n_covariates"])

        self.n_pop, self.n_markers = rho.shape
        self.rho = nn.Parameter(rho, requires_grad=False)

        self.rho_mask = self.rho.isnan()
        self.rho[self.rho_mask] = 0

        self.std_diags = prior_std + (prior_std_nan - prior_std) * self.rho_mask
        self.log_det_sigma = torch.log(self.std_diags).sum(dim=1)

        self.rho_logit = nn.Parameter(
            torch.randn(self.rho.shape) * self.rho_mask, requires_grad=True
        )

        self.prior_h = distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.n_markers),
            torch.eye(self.n_markers),
        )
        self.prior_pi = distributions.dirichlet.Dirichlet(
            torch.tensor([self.hparams.alpha_dirichlet] * self.n_pop)
        )

        self.log_pi = torch.log(torch.ones(self.n_pop) / self.n_pop)

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
    def pi(self) -> Tensor:
        return torch.exp(self.log_pi)

    @property
    def rho_inferred(self) -> Tensor:
        return self.rho + torch.tanh(10 * self.rho_logit * self.rho_mask)

    @torch.no_grad()
    def sample(self, n_samples: int, covariates: Tensor) -> Tuple[Tensor, Tensor]:
        sample_shape = torch.Size([n_samples])
        z = self.prior_z.sample(sample_shape)
        h = self.prior_h.sample(sample_shape) * self.std_diags[z] + self.rho_inferred[z]
        x = self.inverse(h, covariates).detach()
        return x, z

    def compute_probabilities(
        self, x: Tensor, covariates: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        h, _, ldj_sum = self(x, covariates)

        log_probs = (
            self.prior_h.log_prob(
                (h[:, None, :] - self.rho_inferred[None, ...]) / self.std_diags
            )
            - self.log_det_sigma
            + self.log_pi
        )
        probs = torch.softmax(log_probs, dim=1)  # Expectation step
        log_prob_pi = self.prior_pi.log_prob(probs.mean(dim=0))  # self.pi + self.eps)

        return probs, log_probs, ldj_sum, log_prob_pi

    def _update_log_pi(self, x: Tensor, covariates: Tensor) -> None:
        probs, *_ = self.compute_probabilities(x, covariates)
        self.log_pi = torch.log(probs.mean(dim=0).detach() + self.eps)

    def loss(self, x: Tensor, covariates: Tensor) -> Tensor:
        n_samples = x.shape[0]
        probs, log_probs, ldj_sum, log_prob_pi = self.compute_probabilities(
            x, covariates
        )
        return -(ldj_sum.mean() + log_prob_pi + (probs * log_probs).sum() / n_samples)
