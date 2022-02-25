from pytorch_lightning.callbacks import Callback, EarlyStopping
import torch
from sklearn.metrics import silhouette_score
from scipy.stats import wasserstein_distance
import scanpy as sc
import umap
from sklearn.metrics.pairwise import euclidean_distances
import logging
from sklearn.metrics import accuracy_score

log = logging.getLogger(__name__)


class AnnotationMetrics:
    def __init__(self, model, n_samples: int, n_components: int) -> None:
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.n_components = n_components

        log.info(f"AnnotationMetrics callback setup with n_samples={self.n_samples}")
        if self.n_samples >= 1e4:
            log.info(
                "n_samples is high, which leads to better metrics approximation, but longer callback setup"
            )
        self.n_obs, self.n_vars = self.model.adata.shape

        self.x_subsample = torch.tensor(
            sc.pp.subsample(self.model.adata, n_obs=self.n_samples, copy=True).X
        )
        x_subsample_umap = umap.UMAP(n_components=self.n_components).fit_transform(
            self.x_subsample
        )
        self.pairwise_distances = euclidean_distances(x_subsample_umap)

    def __call__(self) -> None:
        X_sample, _ = self.model.module.sample(self.n_samples)
        wd_sum = sum(
            wasserstein_distance(X_sample[:, i], self.x_subsample[:, i])
            for i in range(self.n_vars)
        )
        self.model.log("mean_wasserstein_distance", wd_sum / self.n_vars, prog_bar=True)

        _silhouette_score = silhouette_score(
            self.pairwise_distances,
            self.model.predict(x=self.x_subsample, key_added=None).values,
            metric="precomputed",
        )
        self.model.log("silhouette_score", _silhouette_score, prog_bar=True)

        if "cell_type" in self.model.adata.obs:
            self.model.log(
                "accuracy_score",
                accuracy_score(
                    self.model.adata.obs.cell_type,
                    self.model.predict(key_added=None).values,
                ),
                prog_bar=True,
            )
