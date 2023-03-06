import os
import sys
import torch
import wandb

from reaction_model import *


class MolecularTransformer(ReactionModel):

    def __init__(self):
        self.name = "MolecularTransformer"
        super().__init__()

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh MolecularTransformer/preprocess.sh")

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system("sh MolecularTransformer/train.sh")

    def predict(self, data):
        """Predict provided data with the reaction model"""
        pass


class OpenNMT(ReactionModel):

    def __init__(self):
        self.name = "OpenNMT_Transformer"
        super().__init__()

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh OpenNMT_Transformer/preprocess.sh")

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system("sh OpenNMT_Transformer/train.sh")

    def predict(self, data):
        """Predict provided data with the reaction model"""
        pass


class G2S(ReactionModel):

    def __init__(self):
        self.name = "Graph2SMILES"
        super().__init__()

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh Graph2SMILES/scripts/preprocess.sh")

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system("sh Graph2SMILES/scripts/train_g2s.sh")

    def predict(self, data):
        """Predict provided data with the reaction model"""
        os.system("sh Graph2SMILES/scripts/predict.sh")


class MEGAN(ReactionModel):

    def __init__(self):
        self.name = "MEGAN"
        super().__init__()

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh megan/preprocess.sh")

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system("sh megan/train.sh")

    def predict(self, data):
        """Predict provided data with the reaction model"""
        pass


class BenchmarkPipeline:

    def __init__(self, model: ReactionModel):
        self.model = model

    def run_train_pipeline(self):
        self.model.preprocess()
        self.model.train()


if __name__ == "__main__":
    reaction_model = OpenNMT()
    pipeline = BenchmarkPipeline(model=reaction_model)
    pipeline.run_train_pipeline()
