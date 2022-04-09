import torch
from torch import Tensor
from torch import nn
from torch import distributions
from typing import Tuple, Union
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
        n_layers: int,
        prior_std: float,
        lr: float,
        batch_size: int,
        ratio_threshold: float,
        alpha: float,
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
            ratio_threshold (float, optional): Minimum ratio of cells to be observed for each population. Defaults to 1e-4.
            alpha (float, optional): Constraint term weight in the loss function. Defaults to 1.0.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["rho", "n_covariates"])

        self.n_pop, self.n_markers = rho.shape
        self.register_buffer("rho", rho)

        self.rho_mask = self.rho.isnan()
        self.rho[self.rho_mask] = 0

        self.register_buffer("h_mean", torch.zeros(self.n_markers))
        self.register_buffer("h_var", prior_std ** 2 * torch.eye(self.n_markers))

        self.pi_logit = nn.Parameter(torch.zeros(self.n_pop))

        self.real_nvp = RealNVP(
            self.n_markers + n_covariates,
            self.hparams.hidden_size,
            self.n_markers,
            self.hparams.n_hidden_layers,
            self.hparams.n_layers,
        )

    def forward(self, x: Tensor, covariates: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward implementation, going through the complete flow

        Args:
            x (Tensor): Inputs
            covariates (Tensor): Covariates

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple of (outputs, covariates, lod_det_jacobian sum)
        """
        return self.real_nvp(x, covariates)

    def inverse(self, z: Tensor, covariates: Tensor) -> Tensor:
        """Goes through the complete flow in reverse direction

        Args:
            h (Tensor): Inputs
            covariates (Tensor): Covariates

        Returns:
            Tensor: Outputs
        """
        return self.real_nvp.inverse(z, covariates)

    @property
    def prior_h(self) -> distributions.distribution.Distribution:
        """Cell-specific term prior (H)

        Returns:
            distributions.distribution.Distribution: Distribution of H
        """
        return distributions.multivariate_normal.MultivariateNormal(
            self.h_mean,
            self.h_var,
        )

    @property
    def prior_z(self) -> distributions.distribution.Distribution:
        """Population prior

        Returns:
            distributions.distribution.Distribution: Distribution of the population index prior
        """
        return distributions.categorical.Categorical(self.pi)

    @property
    def log_pi(self) -> Tensor:
        """Returns the log population weights

        Returns:
            Tensor: Log population weights
        """
        return torch.log_softmax(10 * self.pi_logit, dim=0)

    @property
    def pi(self) -> Tensor:
        """Returns the population weights

        Returns:
            Tensor: Population weights
        """
        return torch.exp(self.log_pi)

    @torch.no_grad()
    def sample(
        self, n_samples: int, covariates: Tensor, z_pop: Union[int, Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Sample cells

        Args:
            n_samples (int): Number of cells to be sampled
            covariates_sample (Union[Tensor, None], optional): Sample of cobariates. Defaults to None.
            z_pop (Union[str, List[str], int, Tensor, None], optional): Population indices. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Pair of (cell expressions, population)
        """
        sample_shape = torch.Size([n_samples])

        if z_pop is None:
            z = self.prior_z.sample(sample_shape)
        elif isinstance(z_pop, int):
            z = torch.full(sample_shape, z_pop)
        elif isinstance(z_pop, torch.Tensor):
            z = z_pop
        else:
            raise ValueError(
                f"z_pop has to be 'None', an 'int' or a 'torch.Tensor'. Found type {type(z_pop)}."
            )

        u = self.prior_h.sample(sample_shape) + self.rho[z]  # TODO: use kde for NaN
        x = self.inverse(u, covariates).detach()
        return x, z

    def difference_to_modes(self, u: Tensor) -> Tensor:
        """Difference between the latent variable U and all the modes

        Args:
            u (Tensor): Latent variables tensor

        Returns:
            Tensor: Tensor of difference to all modes
        """
        h = u[:, None, :] - self.rho[None, ...]
        h[:, self.rho_mask] = 0
        return h

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

        h = self.difference_to_modes(u)

        log_probs = self.prior_h.log_prob(h) + self.log_pi
        probs = torch.softmax(log_probs, dim=1)

        return probs, log_probs, ldj_sum

    def compute_constraint(self, probs: Tensor) -> Tensor:
        """Computes the model constraint

        Args:
            probs (Tensor): Tensor of probabilities

        Returns:
            Tensor: Contraint
        """
        soft_ratio = torch.sigmoid((probs - 0.5) * 20).mean(dim=0)
        return (
            self.hparams.alpha
            * torch.relu(1 - soft_ratio / self.hparams.ratio_threshold).sum()
        )

    def loss(self, x: Tensor, covariates: Tensor) -> Tensor:
        """Loss computation

        Args:
            x (Tensor): Inputs
            covariates (Tensor): Covariates

        Returns:
            Tensor: Loss
        """
        probs, log_probs, ldj_sum = self.compute_probabilities(x, covariates)
        constraint = self.compute_constraint(probs)

        return -(torch.logsumexp(log_probs, dim=1) + ldj_sum).mean() + constraint
