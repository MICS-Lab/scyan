import torch
from torch import Tensor
from torch import nn
from torch import distributions
from typing import Tuple, Union
import pytorch_lightning as pl

from .real_nvp import RealNVP
from .distribution import PriorDistribution


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
            alpha (float, optional): Constraint term weight in the loss function. Defaults to 1.0.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["rho", "n_covariates"])

        self.n_pop, self.n_markers = rho.shape
        self.register_buffer("rho", rho)

        self.rho_mask = self.rho.isnan()
        self.rho[self.rho_mask] = 0

        self.pi_logit = nn.Parameter(torch.zeros(self.n_pop))

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
        if z_pop is None:
            z = self.prior_z.sample((n_samples,))
        elif isinstance(z_pop, int):
            z = torch.full((n_samples,), z_pop)
        elif isinstance(z_pop, torch.Tensor):
            z = z_pop
        else:
            raise ValueError(
                f"z_pop has to be 'None', an 'int' or a 'torch.Tensor'. Found type {type(z_pop)}."
            )

        u = self.prior.sample(z)
        x = self.inverse(u, covariates)
        return x, z

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

        return probs, log_probs, ldj_sum

    def compute_regularization(self, probs: Tensor) -> Tensor:
        """Computes the model regularization

        Args:
            probs (Tensor): Tensor of probabilities

        Returns:
            Tensor: Contraint
        """
        empirical_weights = probs.mean(dim=0)
        return (
            -self.hparams.alpha
            / self.n_markers
            * torch.log(empirical_weights + self.eps).sum()
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
        regularization = self.compute_regularization(probs)

        return -(torch.logsumexp(log_probs, dim=1) + ldj_sum).mean() + regularization
