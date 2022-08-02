import pytorch_lightning as pl
import torch
from torch import Tensor, distributions


class PriorDistribution(pl.LightningModule):
    """Prior distribution $U$"""

    def __init__(self, rho: Tensor, rho_mask: Tensor, prior_std: float, n_markers: int):
        """
        Args:
            rho: Tensor $\rho$ representing the knowledge table
            rho_mask: Mask telling where $\rho$ is NA
            prior_std: Standard deviation $\sigma$ for $H$.
            n_markers: Number of markers in the table.
        """
        super().__init__()
        self.prior_std = prior_std
        self.n_markers = n_markers

        self.register_buffer("rho", rho)
        self.register_buffer("rho_mask", rho_mask)
        self.register_buffer("loc", torch.zeros((n_markers)))
        self.register_buffer("cov", torch.eye((n_markers)) * self.prior_std**2)

        self.uniform = distributions.Uniform(-1, 1)
        self.normal = distributions.Normal(0, self.prior_std)

        self.compute_constant_terms()

    @property
    def prior_h(self) -> distributions.Distribution:
        """The distribution of $H$"""
        return distributions.MultivariateNormal(self.loc, self.cov)

    def compute_constant_terms(self) -> None:
        self.uniform_law_radius = 1 - self.prior_std

        _gamma = (
            self.uniform_law_radius
            / self.prior_std
            * torch.sqrt(2 / torch.tensor(torch.pi))
        )
        self.gamma = 1 / (1 + _gamma)

        na_constant_term = self.rho_mask.sum(dim=1) * torch.log(self.gamma)
        self.register_buffer("na_constant_term", na_constant_term)

    def difference_to_modes(self, u: Tensor) -> Tensor:
        """Difference between the latent variable $U$ and all the modes (one mode per population).

        Args:
            u: Latent variables tensor of size $(B, M)$.

        Returns:
            Tensor of size $(B, P, M)$ representing differences to all modes.
        """
        diff = u[:, None, :] - self.rho[None, ...]

        diff[:, self.rho_mask] = torch.clamp(
            diff[:, self.rho_mask].abs() - self.uniform_law_radius, min=0
        )  # Handling NA values

        return diff

    def log_prob_per_marker(self, u: Tensor) -> Tensor:
        """Log probability per marker and per population.

        Args:
            u: Latent variables tensor of size $(B, M)$.

        Returns:
            Log probabilities tensor of size $(B, P, M)$.
        """
        diff = self.difference_to_modes(u)  # size N x P x M

        return self.normal.log_prob(diff) + self.rho_mask * torch.log(self.gamma)

    def log_prob(self, u: Tensor) -> Tensor:
        """Log probability per population.

        Args:
            u: Latent variables tensor of size $(B, M)$.

        Returns:
            Log probabilities tensor of size $(B, P)$.
        """
        diff = self.difference_to_modes(u)  # size N x P x M

        return self.prior_h.log_prob(diff) + self.na_constant_term

    def sample(self, z: Tensor) -> Tensor:
        """Sampling latent cell-marker expressions.

        Args:
            z: Tensor of population indices.

        Returns:
            Latent expressions, i.e. a tensor of size $(len(Z), M)$.
        """
        (n_samples,) = z.shape

        e = self.rho[z] + self.rho_mask[z] * self.uniform.sample(
            (n_samples, self.n_markers)
        )
        h = self.prior_h.sample((n_samples,))

        return e + h
