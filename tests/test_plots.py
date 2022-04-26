from sqlalchemy import false
import scyan


def test_plots():
    adata, marker_pop_matrix = scyan.data.load("aml", size="short")
    model = scyan.Scyan(adata, marker_pop_matrix)

    model.predict()

    pop = "CD8 T cells"
    ref = "CD4 T cells"

    scyan.plot.kde_per_population(model, pop, show=False)
    scyan.plot.latent_expressions(model, pop, show=False)
    scyan.plot.pop_weighted_kde(model, pop, n_samples=adata.n_obs, show=False)
    scyan.plot.pop_weighted_kde(model, pop, n_samples=adata.n_obs, ref=ref, show=False)
    scyan.plot.probs_per_marker(model, pop, show=False)
