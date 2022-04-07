import logging
import scanpy as sc

sc.settings.verbosity = 3
sc.set_figure_params(facecolor="white", fontsize=10)

logging.basicConfig(level=logging.INFO)

from .model import Scyan
from . import plot, utils, data

# pl.seed_everything(0)
