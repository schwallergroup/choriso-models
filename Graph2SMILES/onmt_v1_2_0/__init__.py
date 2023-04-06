""" Main entry point of the ONMT library """
from __future__ import division, print_function

from Graph2SMILES.onmt_v1_2_0 import inputters
from Graph2SMILES.onmt_v1_2_0 import encoders
from Graph2SMILES.onmt_v1_2_0 import decoders
from Graph2SMILES.onmt_v1_2_0 import models
from Graph2SMILES.onmt_v1_2_0 import utils
from Graph2SMILES.onmt_v1_2_0 import modules
from Graph2SMILES.onmt_v1_2_0.trainer import Trainer
import sys
from Graph2SMILES.onmt_v1_2_0.utils.optimizers import Optimizer as Optim
sys.modules["Graph2SMILES.onmt_v1_2_0.Optim"] = Optim

# For Flake
__all__ = ["inputters", "encoders", "decoders", "models", "utils", "modules", "Trainer"]

__version__ = "1.2.0"
