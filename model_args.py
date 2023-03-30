import abc


class ReactionModelArgs(abc.ABC):
    """Abstract class for wrapping arguments for all stages of model training. Inspired by the Graph2SMILES args code"""
    def __init__(self):
        pass

    def preprocess_args(self):
        pass

    def training_args(self):
        pass

    def predict_args(self):
        pass
