from pytorch_lightning.callbacks import Callback, EarlyStopping
import torch
from sklearn.metrics import silhouette_score
from scipy.stats import wasserstein_distance
import scanpy as sc
import umap
from sklearn.metrics.pairwise import euclidean_distances
import logging

log = logging.getLogger(__name__)


class AnnotationMetrics(Callback):
    def __init__(self, n_samples: int = 10000, n_components: int = 5) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_components = n_components

    def setup(self, trainer, scyan, _) -> None:
        log.info(f"AnnotationMetrics callback setup with n_samples={self.n_samples}")
        if self.n_samples >= 1e4:
            log.info(
                "n_samples is high, which leads to better metrics approximation, but longer callback setup"
            )
        self.n_obs, self.n_vars = scyan.adata.shape

        self.x_subsample = torch.tensor(
            sc.pp.subsample(scyan.adata, n_obs=self.n_samples, copy=True).X
        )
        x_subsample_umap = umap.UMAP(n_components=self.n_components).fit_transform(
            self.x_subsample
        )
        self.pairwise_distances = euclidean_distances(x_subsample_umap)

    def on_train_epoch_end(self, trainer, scyan) -> None:
        X_sample, _ = scyan.module.sample(self.n_samples)
        wd_sum = sum(
            wasserstein_distance(X_sample[:, i], self.x_subsample[:, i])
            for i in range(self.n_vars)
        )
        scyan.log("mean_wasserstein_distance", wd_sum / self.n_vars, prog_bar=True)

        _silhouette_score = silhouette_score(
            self.pairwise_distances,
            scyan.predict(x=self.x_subsample, key_added=None).values,
            metric="precomputed",
        )
        scyan.log("silhouette_score", _silhouette_score, prog_bar=True)
