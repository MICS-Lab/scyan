import logging
import scanpy as sc

logging.basicConfig(level=logging.INFO)
sc.settings.verbosity = 3
sc.set_figure_params(facecolor="white", fontsize=10)
