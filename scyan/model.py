import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from anndata import AnnData

from .modules import RealNVP


class Scyan(pl.LightningModule):
    def __init__(
        self,
        marker_pop_matrix,
        hidden_size=64,
        alpha_dirichlet=0.01,
        n_layers=6,
        prior_var=0.2,
        lr=5e-3,
    ):
        super().__init__()
        self.marker_pop_matrix = marker_pop_matrix
        self.n_markers, self.n_pop = marker_pop_matrix.shape
        self.rho = torch.Tensor(marker_pop_matrix.values.T[None, :, :])
        self.n_layers = n_layers
        self.learning_rate = lr

        self.mask = nn.Parameter(torch.arange(self.n_markers) % 2, requires_grad=False)

        self.prior_h = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.n_markers), prior_var * torch.eye(self.n_markers)
        )
        self.prior_pi = torch.distributions.dirichlet.Dirichlet(
            torch.tensor([alpha_dirichlet] * self.n_pop)
        )

        self.pi_logit = nn.Parameter(torch.randn(self.n_pop))
        self.log_softmax = torch.nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=1)

        self.real_nvp = RealNVP(self.n_markers, hidden_size, self.mask, n_layers)
        self.mse = nn.MSELoss()

    def forward(self, x):
        return self.real_nvp(x)

    def inverse(self, z):
        return self.real_nvp.inverse(z)

    @property
    def prior_z(self):
        return torch.distributions.categorical.Categorical(
            F.softmax(self.pi_logit, dim=0)
        )

    def sample(self, n_samples):
        sample_shape = torch.Size([n_samples])
        z = self.prior_z.sample(sample_shape)
        h = self.prior_h.sample(sample_shape) + self.rho[0, z]
        x = self.inverse(h).detach()
        return x, z

    def get_all_probs(self, x):
        log_pi = self.log_softmax(self.pi_logit)
        log_prob_pi = -self.prior_pi.log_prob(torch.exp(log_pi) + 1e-20)

        h, ldj_sum = self(x)
        log_probs = self.prior_h.log_prob(h[:, None, :] - self.rho) + log_pi
        probs = self.softmax(log_probs).detach()
        return probs, log_probs, ldj_sum, log_prob_pi

    def predict(self, adata: AnnData):
        predictions, *_ = self.get_all_probs(torch.Tensor(adata.X))

        adata.obs[self.marker_pop_matrix.columns] = predictions.numpy()
        adata.obs["scyan_pop"] = adata.obs[self.marker_pop_matrix.columns].idxmax(
            axis=1
        )

        return predictions

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, x, _):
        probs, log_probs, ldj_sum, log_prob_pi = self.get_all_probs(x)
        loss = -(ldj_sum.mean() + log_prob_pi + (probs * log_probs).sum() / x.shape[0])
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        pass
