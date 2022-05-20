import torch
from torch import Tensor
from torch import nn
from torch import distributions
from typing import Tuple, Union
import pytorch_lightning as pl

from .real_nvp import RealNVP
from .distribution import PriorDistribution
from ..mmd import LossMMD
from ..utils import _truncate_n_samples


class ScyanModule(pl.LightningModule):
    pi_logit_ratio: float = 10  # To learn pi logits faster

    def __init__(
        self,
        rho: Tensor,
        n_covariates: int,
        hidden_size: int,
        n_hidden_layers: int,
        n_layers: int,
        prior_std: float,
        alpha: float,
        temperature_mmd: float,
        temp_lr_weights: float,
        mmd_max_samples: int,
    ):
        """Module containing the core logic behind the Scyan model

        Args:
            rho (Tensor): Tensor representing the marker-population matrix
            n_covariates (int): Number of covariates considered
            hidden_size (int, optional): Neural networks (s and t) hidden size. Defaults to 64.
            n_hidden_layers (int, optional): Neural networks (s and t) number of hidden layers. Defaults to 1.
            n_layers (int, optional): Number of coupling layers. Defaults to 6.
            prior_std (float, optional): Standard deviation of the base distribution (H). Defaults to 0.25.
            lr (float, optional): Learning rate. Defaults to 5e-3.
            batch_size (int): Batch size.
            alpha (float, optional): Constraint term weight in the loss function. Defaults to 1.0.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["rho", "n_covariates"])

        self.n_pops, self.n_markers = rho.shape
        self.register_buffer("rho", rho)

        self.rho_mask = self.rho.isnan()
        self.rho[self.rho_mask] = 0

        self.pi_logit = nn.Parameter(torch.zeros(self.n_pops))

        self.real_nvp = RealNVP(
            self.n_markers + n_covariates,
            self.hparams.hidden_size,
            self.n_markers,
            self.hparams.n_hidden_layers,
            self.hparams.n_layers,
        )

        self.prior = PriorDistribution(
            self.rho, self.rho_mask, self.hparams.prior_std, self.n_markers
        )

        self._no_mmd = self.hparams.alpha is None or self.hparams.alpha == 0
        self.mmd = LossMMD()

    def forward(self, x: Tensor, covariates: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward implementation, going through the complete flow

        Args:
            x (Tensor): Inputs
            covariates (Tensor): Covariates

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple of (outputs, covariates, lod_det_jacobian sum)
        """
        return self.real_nvp(x, covariates)

    @torch.no_grad()
    def inverse(self, u: Tensor, covariates: Tensor) -> Tensor:
        """Goes through the complete flow in reverse direction

        Args:
            u (Tensor): Inputs
            covariates (Tensor): Covariates

        Returns:
            Tensor: Outputs
        """
        return self.real_nvp.inverse(u, covariates)

    @property
    def prior_z(self) -> distributions.Distribution:
        """Population prior

        Returns:
            distributions.Distribution: Distribution of the population index prior
        """
        return distributions.Categorical(self.pi)

    @property
    def log_pi(self) -> Tensor:
        """Returns the log population weights

        Returns:
            Tensor: Log population weights
        """
        return torch.log_softmax(self.pi_logit_ratio * self.pi_logit, dim=0)

    @property
    def pi(self) -> Tensor:
        """Returns the population weights

        Returns:
            Tensor: Population weights
        """
        return torch.exp(self.log_pi)

    def pi_temperature(self, T):
        return torch.softmax(self.pi_logit_ratio * self.pi_logit / T, dim=0).detach()

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        covariates: Tensor,
        z: Union[int, Tensor, None] = None,
        return_z: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sample cells

        Args:
            n_samples (int): Number of cells to be sampled
            covariates_sample (Union[Tensor, None], optional): Sample of cobariates. Defaults to None.
            z (Union[str, List[str], int, Tensor, None], optional): Population indices. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Pair of (cell expressions, population)
        """
        if z is None:
            z = self.prior_z.sample((n_samples,))
        elif isinstance(z, int):
            z = torch.full((n_samples,), z)
        elif isinstance(z, torch.Tensor):
            pass
        else:
            raise ValueError(
                f"z has to be 'None', an 'int' or a 'torch.Tensor'. Found type {type(z)}."
            )

        u = self.prior.sample(z)
        x = self.inverse(u, covariates)

        return (x, z) if return_z else x

    def compute_probabilities(
        self, x: Tensor, covariates: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes probabilities used to define the loss function

        Args:
            x (Tensor): Inputs
            covariates (Tensor): Covariates

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Predicted probabilities, log probabilities and sum of the log det jacobian
        """
        u, _, ldj_sum = self(x, covariates)

        log_probs = self.prior.log_prob(u) + self.log_pi  # size N x P
        probs = torch.softmax(log_probs, dim=1)

        return probs, log_probs, ldj_sum, u

    @_truncate_n_samples
    def compute_mmd(self, u):
        if self._no_mmd:
            return 0

        pi_temperature = self.pi_temperature(self.hparams.temperature_mmd)
        z = distributions.Categorical(pi_temperature).sample((len(u),))

        u_sample = self.prior.sample(z)
        return self.mmd(u, u_sample)

    @_truncate_n_samples
    def batch_mmd(self, u_ref, u_other):
        n_samples = min(len(u_ref), len(u_other))
        assert n_samples >= 1000, "n_samples has to be >= 1000"
        return self.mmd(u_ref[:n_samples], u_other[:n_samples])

    def losses(
        self, x: Tensor, covariates: Tensor, batch: Tensor, ref: Union[int, None]
    ) -> Tensor:
        """Loss computation

        Args:
            x (Tensor): Inputs
            covariates (Tensor): Covariates

        Returns:
            Tensor: Loss
        """
        _, log_probs, ldj_sum, u = self.compute_probabilities(x, covariates)

        inv_pi_temperature = 1 / self.pi_temperature(self.hparams.temp_lr_weights)
        pop_weights = inv_pi_temperature / inv_pi_temperature.mean()
        cell_weights = pop_weights[log_probs.argmax(dim=1)]

        kl = -(cell_weights * (torch.logsumexp(log_probs, dim=1) + ldj_sum)).mean()
        weighted_mmd = self.hparams.alpha * self.compute_mmd(
            u[: self.hparams.mmd_max_samples]
        )

        if ref is not None:
            u_ref = u[batch == ref]
            u_others = [
                u[batch == other] for other in set(batch.tolist()) if other != ref
            ]

            batch_mmd = torch.stack(
                [self.batch_mmd(u_ref, u_other) for u_other in u_others]
            ).sum()
        else:
            batch_mmd = 0

        return kl, weighted_mmd, batch_mmd
