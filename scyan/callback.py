from pytorch_lightning.callbacks import Callback
import torch
from sklearn.metrics import silhouette_score
from scipy.stats import wasserstein_distance
import scanpy as sc
import umap
from sklearn.metrics.pairwise import euclidean_distances

from .model import Scyan


class AnnotationMetrics(Callback):
    def __init__(self, n_samples=10000, n_components=5) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_components = n_components

    def setup(self, trainer, scyan: Scyan):
        self.n_obs, self.n_vars = scyan.adata.shape

        self.X_subsample = torch.Tensor(
            sc.pp.subsample(scyan.adata, n_obs=self.n_samples, copy=True).X
        )
        X_subsample_umap = umap.UMAP(n_components=self.n_components).fit_transform(
            self.X_subsample
        )
        self.pairwise_distances = euclidean_distances(X_subsample_umap)

    def on_train_epoch_end(self, trainer, scyan: Scyan):
        X_sample, _ = scyan.sample(self.n_obs)
        wd_sum = sum(
            wasserstein_distance(X_sample[:, i], scyan.adata.X[:, i])
            for i in range(self.n_vars)
        )
        scyan.log("wasserstein_distance_sum", wd_sum)

        _silhouette_score = silhouette_score(
            self.pairwise_distances,
            scyan.predict(X=self.X_subsample, key_added=None).values,
            metric="precomputed",
        )
        scyan.log("silhouette_score", _silhouette_score)
