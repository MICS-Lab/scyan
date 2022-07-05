import logging

import pytorch_lightning as pl
import scanpy as sc

__version__ = "0.1.0"  # TODO: update version

sc.settings.verbosity = 3
sc.set_figure_params(facecolor="white", fontsize=10)

logging.basicConfig(level=logging.INFO)

from .model import Scyan
from . import data, plot, utils

pl.seed_everything(0)
