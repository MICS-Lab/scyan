import importlib.metadata
import logging

import lightning as L

from .model import Scyan
from ._io import read_fcs, read_csv, write_fcs, write_csv
from . import data, plot, tools, utils
from . import preprocess

__version__ = importlib.metadata.version("scyan")

log = logging.getLogger("scyan")
utils.configure_logger(log)

plot.reset_plot_settings()

L.seed_everything(0)  # Set seed to 0 (using pytorch lightning)
