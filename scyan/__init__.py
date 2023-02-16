import importlib.metadata
import logging

import pytorch_lightning as pl

from .model import Scyan
from ._io import read_fcs, read_csv, write_fcs, write_csv
from . import data, plot, tools, utils
from . import preprocess

__version__ = importlib.metadata.version("scyan")

logging.getLogger().setLevel(logging.INFO)
plot.reset_plot_settings()
pl.seed_everything(0)
