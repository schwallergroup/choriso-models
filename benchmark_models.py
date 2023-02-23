import os
import sys
import torch

from reaction_model import *

from MolecularTransformer.train import main as train_MolecularTransformer

from Graph2SMILES.train import get_train_parser
from Graph2SMILES.utils.train_utils import set_seed, setup_logger
from Graph2SMILES.train import main as train_G2S
from Graph2SMILES.predict import get_predict_parser
from Graph2SMILES.predict import main as predict_G2S

# from megan.bin.train import train_megan


class MolecularTransformer(ReactionModel):

    def __init__(self):
        super().__init__()
        pass

    def load_checkpoint(self, path):
        """Load a pre-trained model from a provided path"""
        pass

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        pass

    def predict(self, data):
        """Predict provided data with the reaction model"""
        pass

    def save_results(self):
        """Save the results obtained by training"""
        pass

    def load_results(self, path):
        """Load the results from a given path"""
        pass


class G2S(ReactionModel):

    def __init__(self, model_dir: str):
        super().__init__(model_dir)

    def load_checkpoint(self, path):
        """Load a pre-trained model from a provided path"""
        pass

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system("sh Graph2SMILES/scripts/train_g2s.sh")

    def predict(self, data):
        """Predict provided data with the reaction model"""
        os.system("sh Graph2SMILES/scripts/predict.sh")

    def save_results(self):
        """Save the results obtained by training"""
        pass

    def load_results(self, path):
        """Load the results from a given path"""
        pass


class MEGAN(ReactionModel):

    def __init__(self):
        super().__init__()
        pass

    def load_checkpoint(self, path):
        """Load a pre-trained model from a provided path"""
        pass

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        pass

    def predict(self, data):
        """Predict provided data with the reaction model"""
        pass

    def save_results(self):
        """Save the results obtained by training"""
        pass

    def load_results(self, path):
        """Load the results from a given path"""
        pass


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.realpath(__file__))

    models_dir = os.path.join(this_dir, "Models")
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    g2s_dir = os.path.join(models_dir, "G2S")

    test_model = G2S(g2s_dir)

