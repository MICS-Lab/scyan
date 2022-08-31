import logging
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, distributions, nn

from . import LossMMD, PriorDistribution, RealNVP

log = logging.getLogger(__name__)


class ScyanModule(pl.LightningModule):
    """Core logic contained inside the main class [Scyan][scyan.Scyan]. Do not use this class directly.

    Attributes:
        real_nvp (RealNVP): The Normalizing Flow (a [RealNVP][scyan.module.RealNVP] object)
        prior (PriorDistribution): The prior $U$ (a [PriorDistribution][scyan.module.PriorDistribution] object)
        loss_mmd (LossMMD): The MMD loss (a [LossMMD][scyan.module.LossMMD] object)
        pi_logit (Tensor): Logits used to learn the population weights
    """

    pi_logit_ratio: float = 100  # To learn pi logits faster

    def __init__(
        self,
        rho: Tensor,
        n_covariates: int,
        other_batches: List[int],
        hidden_size: int,
        n_hidden_layers: int,
        n_layers: int,
        prior_std: float,
        temperature: float,
        mmd_max_samples: int,
        batch_ref_id: Optional[int],
    ):
        """
        Args:
            rho: Tensor $\rho$ representing the knowledge table.
            n_covariates: Number of covariates $M_c$ considered.
            other_batches: List of batches that are not the reference.
            hidden_size: MLP (`s` and `t`) hidden size.
            n_hidden_layers: Number of hidden layers for the MLP (`s` and `t`).
            n_layers: Number of coupling layers.
            prior_std: Standard deviation $\sigma$ of the cell-specific random variable $H$.
            temperature: Temperature to favour small populations.
            mmd_max_samples: Maximum number of samples to give to the MMD.
            batch_ref_id: ID corresponding to the reference batch.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["rho", "n_covariates", "other_batches"])

        self.n_pops, self.n_markers = rho.shape
        self.register_buffer("rho", rho)

        self.rho_mask = self.rho.isnan()
        self.rho[self.rho_mask] = 0
        self.mean_na = self.rho_mask.sum() / self.n_markers

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

        self.other_batches = other_batches
        self.loss_mmd = LossMMD(self.n_markers, self.hparams.prior_std, self.mean_na)

    def forward(self, x: Tensor, covariates: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward implementation, going through the complete flow $f_{\phi}$.

        Args:
            x: Inputs of size $(B, M)$.
            covariates: Covariates of size $(B, M_c)$

        Returns:
            Tuple of (outputs, covariates, lod_det_jacobian sum)
        """
        return self.real_nvp(x, covariates)

    @torch.no_grad()
    def inverse(self, u: Tensor, covariates: Tensor) -> Tensor:
        """Go through the flow in reverse direction, i.e. $f_{\phi}^{-1}$.

        Args:
            u: Latent expressions of size $(B, M)$.
            covariates: Covariates of size $(B, M_c)$

        Returns:
            Outputs of size $(B, M)$.
        """
        return self.real_nvp.inverse(u, covariates)

    @property
    def prior_z(self) -> distributions.Distribution:
        """Population prior, i.e. $Categorical(\pi)$.

        Returns:
            Distribution of the population index.
        """
        return distributions.Categorical(self.pi)

    @property
    def log_pi(self) -> Tensor:
        """Log population weights $log \; \pi$."""
        return torch.log_softmax(self.pi_logit_ratio * self.pi_logit, dim=0)

    @property
    def pi(self) -> Tensor:
        """Population weights $\pi$"""
        return torch.exp(self.log_pi)

    def log_pi_temperature(self, T: float) -> Tensor:
        """Compute the log weights with temperature $log \; \pi^{(-T)}$

        Args:
            T: Temperature.

        Returns:
            Log weights with temperature.
        """
        return torch.log_softmax(self.pi_logit_ratio * self.pi_logit / T, dim=0).detach()

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        covariates: Tensor,
        z: Union[int, Tensor, None] = None,
        return_z: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sampling cell-marker expressions.

        Args:
            n_samples: Number of cells to sample.
            covariates: Tensor of covariates.
            z: Either one population index or a Tensor of population indices. If None, sampling from all populations.
            return_z: Whether to return the population Tensor.

        Returns:
            Sampled cells expressions and, if `return_z`, the populations associated to these cells.
        """
        if z is None:
            z = self.prior_z.sample((n_samples,))
        elif isinstance(z, int):
            z = torch.full((n_samples,), z)
        elif isinstance(z, torch.Tensor):
            z = z.to(int)
        else:
            raise ValueError(
                f"z has to be 'None', an 'int' or a 'torch.Tensor'. Found type {type(z)}."
            )

        u = self.prior.sample(z)
        x = self.inverse(u, covariates)

        return (x, z) if return_z else x

    def compute_probabilities(
        self, x: Tensor, covariates: Tensor, use_temp: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute probabilities used in the loss function.

        Args:
            x: Inputs of size $(B, M)$.
            covariates: Covariates of size $(B, M_c)$.

        Returns:
            Log probabilities of size $(B, P)$, the log det jacobian and the latent expressions of size $(B, M)$.
        """
        u, _, ldj_sum = self(x, covariates)

        log_pi = (
            self.log_pi_temperature(-self.hparams.temperature)
            if use_temp
            else self.log_pi
        )

        log_probs = self.prior.log_prob(u) + log_pi  # size N x P

        return log_probs, ldj_sum, u

    def batch_correction_mmd(self, u1, pop1, u2, pop2):
        n_samples = min(len(u1), len(u2), self.hparams.mmd_max_samples)

        if n_samples < 500:
            log.warn(f"Correcting batch effect with few samples ({n_samples})")

        pop_weights = 1 / self.pi.detach()

        indices1 = torch.multinomial(pop_weights[pop1], n_samples)
        indices2 = torch.multinomial(pop_weights[pop2], n_samples)

        return self.loss_mmd(u1[indices1], u2[indices2])

    def get_mmd_inputs(self, u: Tensor, batches: Tensor, probs: Tensor, b: int):
        condition = batches == b
        return u[condition], probs[condition].argmax(dim=1)

    def losses(
        self,
        x: Tensor,
        covariates: Tensor,
        batches: Tensor,
        use_temp: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the module loss for one mini-batch.

        Args:
            x: Inputs of size $(B, M)$.
            covariates: Covariates of size $(B, M_c)$.
            batches: Batch information used to correct batch-effect (tensor of size $(B)$)
            use_temp: Whether to consider temperature is the KL term.

        Returns:
            The KL loss term and the MMD loss term.
        """
        log_probs, ldj_sum, u = self.compute_probabilities(x, covariates, use_temp)

        kl = -(torch.logsumexp(log_probs, dim=1) + ldj_sum).mean()

        if self.hparams.batch_ref_id is None:
            return kl, 0

        u_ref, pop_ref = self.get_mmd_inputs(
            u, batches, log_probs, self.hparams.batch_ref_id
        )

        mmd = sum(
            self.batch_correction_mmd(
                u_ref, pop_ref, *self.get_mmd_inputs(u, batches, log_probs, other)
            )
            for other in self.other_batches
        )

        return kl, mmd
