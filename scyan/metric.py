import torch
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

    @torch.no_grad()
    def __call__(self) -> None:
        pi_rmse = torch.sqrt(((self.model.pi_hat - self.model.module.pi) ** 2).sum())
        self.model.log("pi_rmse", pi_rmse, prog_bar=True)

        if "cell_type" in self.model.adata.obs:
            self.model.log(
                "accuracy_score",
                accuracy_score(
                    self.model.adata.obs.cell_type,
                    self.model.predict(key_added=None).values,
                ),
                prog_bar=True,
            )
