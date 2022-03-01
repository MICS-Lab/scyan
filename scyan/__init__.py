import logging
import scanpy as sc

sc.settings.verbosity = 3
sc.set_figure_params(facecolor="white", fontsize=10)

logging.basicConfig(level=logging.INFO)

import scyan.plot
from scyan.model import Scyan
