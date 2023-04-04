""" Main entry point of the ONMT library """
from __future__ import division, print_function

import Graph2SMILES.onmt_v1_2_0.inputters
import Graph2SMILES.onmt_v1_2_0.encoders
import Graph2SMILES.onmt_v1_2_0.decoders
import Graph2SMILES.onmt_v1_2_0.models
import Graph2SMILES.onmt_v1_2_0.utils
import Graph2SMILES.onmt_v1_2_0.modules
from Graph2SMILES.onmt_v1_2_0.trainer import Trainer
import sys
import Graph2SMILES.onmt_v1_2_0.utils.optimizers
Graph2SMILES.onmt_v1_2_0.utils.optimizers.Optim = Graph2SMILES.onmt_v1_2_0.utils.optimizers.Optimizer
sys.modules["Graph2SMILES.onmt_v1_2_0.Optim"] = Graph2SMILES.onmt_v1_2_0.utils.optimizers

# For Flake
__all__ = [Graph2SMILES.onmt_v1_2_0.inputters, Graph2SMILES.onmt_v1_2_0.encoders, Graph2SMILES.onmt_v1_2_0.decoders,
           Graph2SMILES.onmt_v1_2_0.models, Graph2SMILES.onmt_v1_2_0.utils, Graph2SMILES.onmt_v1_2_0.modules, "Trainer"]

__version__ = "1.2.0"
