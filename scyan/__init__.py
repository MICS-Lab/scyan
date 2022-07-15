import logging
import importlib.metadata

import pytorch_lightning as pl
import scanpy as sc

__version__ = importlib.metadata.version("scyan")

sc.settings.verbosity = 3  # TODO: remove
sc.set_figure_params(facecolor="white", fontsize=10)  # TODO: remove

logging.basicConfig(level=logging.INFO)

from .model import Scyan
from . import data, plot, utils, preprocess
from .utils import read_fcs, write_fcs

pl.seed_everything(0)
