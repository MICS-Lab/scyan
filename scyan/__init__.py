import importlib.metadata
import logging

import pytorch_lightning as pl
import scanpy as sc

from .model import Scyan
from ._io import read_fcs, write_fcs, write_csv
from . import data, plot, tools, utils
from . import preprocess

__version__ = importlib.metadata.version("scyan")

logging.basicConfig(level=logging.INFO)
sc.set_figure_params(facecolor="white", fontsize=10)
pl.seed_everything(0)
