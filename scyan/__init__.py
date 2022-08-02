import logging
import importlib.metadata

import pytorch_lightning as pl

__version__ = importlib.metadata.version("scyan")

logging.basicConfig(level=logging.INFO)

from .model import Scyan
from . import data, plot, utils, preprocess
from .utils import read_fcs, write_fcs

pl.seed_everything(0)
