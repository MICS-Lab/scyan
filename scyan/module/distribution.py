import torch
from torch import Tensor
from torch import distributions
import pytorch_lightning as pl


class PriorDistribution(pl.LightningModule):
    def __init__(self, rho: Tensor, rho_mask: Tensor, prior_std: float, n_markers: int):
        super().__init__()
        self.rho = rho
        self.rho_mask = rho_mask
        self.prior_std = prior_std
        self.n_markers = n_markers

        self.register_buffer("loc", torch.zeros((n_markers)))
        self.register_buffer("cov", torch.eye((n_markers)) * self.prior_std ** 2)
        self.prior_h = distributions.MultivariateNormal(self.loc, self.cov)
        self.uniform = distributions.Uniform(-1, 1)
        self.normal = distributions.Normal(0, self.prior_std)

        self.compute_constant_terms()

    def compute_constant_terms(self):
        self.uniform_law_radius = 1 - self.prior_std

        _gamma = (
            self.uniform_law_radius
            / self.prior_std
            * torch.sqrt(2 / torch.tensor(torch.pi))
        )
        self.gamma = 1 / (1 + _gamma)
        self.na_constant_term = self.rho_mask.sum(dim=1) * torch.log(self.gamma)

    def clamp_nan_values(self, x: Tensor) -> Tensor:
        return torch.clamp(x.abs() - self.uniform_law_radius, min=0)

    def difference_to_closest_mode(self, u: Tensor, argmax: Tensor) -> Tensor:
        diff = u - self.rho[argmax]
        diff[self.rho_mask[argmax]] = self.clamp_nan_values(diff[self.rho_mask[argmax]])

        return diff

    def difference_to_modes(self, u: Tensor) -> Tensor:
        """Difference between the latent variable U and all the modes

        Args:
            u (Tensor): Latent variables tensor

        Returns:
            Tensor: Tensor of difference to all modes
        """
        diff = u[:, None, :] - self.rho[None, ...]
        diff[:, self.rho_mask] = self.clamp_nan_values(diff[:, self.rho_mask])

        return diff

    def log_prob_per_marker(self, u: Tensor):
        diff = self.difference_to_modes(u)  # size N x P x M

        return self.normal.log_prob(diff) + self.rho_mask * torch.log(self.gamma)

    def log_prob(self, u: Tensor):
        diff = self.difference_to_modes(u)  # size N x P x M

        return self.prior_h.log_prob(diff) + self.na_constant_term

    def sample(self, z: Tensor):
        (n_samples,) = z.shape

        e = self.rho[z] + self.rho_mask[z] * self.uniform.sample(
            (n_samples, self.n_markers)
        )
        h = self.prior_h.sample((n_samples,))

        return e + h
