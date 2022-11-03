import importlib.metadata
import logging

import pytorch_lightning as pl
import scanpy as sc

from .model import Scyan
from . import data, plot, tools, utils
from .utils import read_fcs, write_fcs, write_csv

__version__ = importlib.metadata.version("scyan")

logging.basicConfig(level=logging.INFO)
sc.set_figure_params(facecolor="white", fontsize=10)
pl.seed_everything(0)
