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
        hidden_size: int,
        n_hidden_layers: int,
        alpha_dirichlet: float,
        n_layers: int,
        prior_var: float,
        lr: float,
        batch_size: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_pop, self.n_markers = rho.shape
        self.rho = nn.Parameter(rho[None, :, :], requires_grad=False)

        self.mask = nn.Parameter(torch.arange(self.n_markers) % 2, requires_grad=False)

        self.prior_h = distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.n_markers),
            self.hparams.prior_var * torch.eye(self.n_markers),
        )
        self.prior_pi = distributions.dirichlet.Dirichlet(
            torch.tensor([self.hparams.alpha_dirichlet] * self.n_pop)
        )

        self.log_pi = torch.log(torch.ones(self.n_pop) / self.n_pop)
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.real_nvp(x)

    def inverse(self, z: Tensor) -> Tensor:
        return self.real_nvp.inverse(z)

    @property
    def prior_z(self) -> distributions.distribution.Distribution:
        return distributions.categorical.Categorical(self.pi)

    @property
    def pi(self) -> Tensor:
        return torch.exp(self.log_pi)

    @torch.no_grad()
    def sample(self, n_samples: int) -> Tuple[Tensor, Tensor]:
        sample_shape = torch.Size([n_samples])
        z = self.prior_z.sample(sample_shape)
        h = self.prior_h.sample(sample_shape) + self.rho[0, z]
        x = self.inverse(h).detach()
        return x, z

    def compute_probabilities(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        h, ldj_sum = self(x)

        log_probs = self.prior_h.log_prob(h[:, None, :] - self.rho) + self.log_pi
        probs = self.softmax(log_probs)

        log_prob_pi = self.prior_pi.log_prob(probs.mean(dim=0))  # self.pi + self.eps)
        self.log_pi = torch.log(probs.mean(dim=0).detach() + self.eps)
        return probs, log_probs, ldj_sum, log_prob_pi

    def loss(self, x: Tensor):
        probs, log_probs, ldj_sum, log_prob_pi = self.compute_probabilities(x)
        return -(ldj_sum.mean() + log_prob_pi + (probs * log_probs).sum() / x.shape[0])
