import logging
import scanpy as sc

sc.settings.verbosity = 3
sc.set_figure_params(facecolor="white", fontsize=10)

logging.basicConfig(level=logging.INFO)

from scyan.model import Scyan
import scyan.plot
