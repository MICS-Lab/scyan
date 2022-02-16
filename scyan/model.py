import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from anndata import AnnData
from sklearn.metrics import silhouette_score
import pandas as pd
from scipy.stats import wasserstein_distance
import umap
from sklearn.metrics.pairwise import euclidean_distances

from scyan.modules import RealNVP


class Scyan(pl.LightningModule):
    def __init__(
        self,
        adata: AnnData,
        marker_pop_matrix,
        hidden_size=64,
        alpha_dirichlet=1.02,
        n_layers=6,
        prior_var=0.2,
        lr=5e-3,
        batch_size=16384,
    ):
        super().__init__()
        self.adata = adata
        self.X = torch.Tensor(adata.X)
        self.marker_pop_matrix = marker_pop_matrix
        self.n_markers, self.n_pop = marker_pop_matrix.shape
        self.rho = torch.Tensor(marker_pop_matrix.values.T[None, :, :])
        self.n_layers = n_layers
        self.learning_rate = lr
        self.batch_size = batch_size

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

        self.init_metrics()

    def init_metrics(self, n_obs=10000):
        self.X_subsample = sc.pp.subsample(self.adata, n_obs=n_obs, copy=True).X
        X_subsample_umap = umap.UMAP(n_components=5).fit_transform(self.X_subsample)
        self.pairwise_distances = euclidean_distances(X_subsample)

    def forward(self, x):
        return self.real_nvp(x)

    def inverse(self, z):
        return self.real_nvp.inverse(z)

    @property
    def prior_z(self):
        return torch.distributions.categorical.Categorical(
            F.softmax(self.pi_logit, dim=0)
        )

    @torch.no_grad()
    def sample(self, n_samples):
        sample_shape = torch.Size([n_samples])
        z = self.prior_z.sample(sample_shape)
        h = self.prior_h.sample(sample_shape) + self.rho[0, z]
        x = self.inverse(h).detach()
        return x, z

    def get_all_probs(self, x):
        log_pi = self.log_softmax(self.pi_logit)
        log_prob_pi = self.prior_pi.log_prob(torch.exp(log_pi) + 1e-20)

        h, ldj_sum = self(x)
        log_probs = self.prior_h.log_prob(h[:, None, :] - self.rho) + log_pi
        probs = self.softmax(log_probs).detach()
        return probs, log_probs, ldj_sum, log_prob_pi

    def training_step(self, x, _):
        probs, log_probs, ldj_sum, log_prob_pi = self.get_all_probs(x)
        loss = -(ldj_sum.mean() + log_prob_pi + (probs * log_probs).sum() / x.shape[0])
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    @torch.no_grad()
    def predict(self, X=None, key_added="scyan_pop"):
        df = self.predict_proba(X=X)
        populations = df.idxmax(axis=1).astype("category")

        if key_added:
            self.adata.obs[key_added] = populations.values

        return populations

    @torch.no_grad()
    def predict_proba(self, X=None):
        predictions, *_ = self.get_all_probs(self.X if X is None else X)
        return pd.DataFrame(predictions.numpy(), columns=self.marker_pop_matrix.columns)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def eval(self):
        X_sample, _ = self.sample(self.X.shape[0])
        wd_sum = sum(
            wasserstein_distance(X_sample[:, i], self.adata.X[:, i])
            for i in range(self.adata.n_vars)
        )
        print("wasserstein_distance_sum", wd_sum)

        _silhouette_score = silhouette_score(
            self.pairwise_distances,
            self.predict(X=torch.Tensor(self.X_subsample), key_added=None).values,
            metric="precomputed",
        )
        print("silhouette_score", _silhouette_score)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.X, batch_size=self.batch_size)
