import torch
import logging
from sklearn.metrics import accuracy_score
import numpy as np

log = logging.getLogger(__name__)


class AnnotationMetrics:
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        log.info("AnnotationMetrics callback setup.")

    @torch.no_grad()
    def __call__(self) -> None:
        # pi_rmse = torch.sqrt(((self.model.pi_hat - self.model.module.pi) ** 2).sum())
        # self.model.log("pi_rmse", pi_rmse, prog_bar=True)

        u = self.model()
        n_samples = min(4096, len(u))
        u = u[np.random.choice(len(u), n_samples, replace=False)]
        z = torch.randint(0, self.model.module.n_pops, size=(n_samples,))
        u_sample = self.model.module.prior.sample(z)
        mmd = self.model.module.mmd(u, u_sample)
        self.model.log("mmd", mmd, prog_bar=True)

        if "cell_type" in self.model.adata.obs:
            self.model.log(
                "accuracy_score",
                accuracy_score(
                    self.model.adata.obs.cell_type,
                    self.model.predict(key_added=None).values,
                ),
                prog_bar=True,
            )
