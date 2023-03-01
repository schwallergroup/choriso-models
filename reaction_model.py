import abc
import os


class Parameters:

    def __init__(self, params: dict):
        self.params = params


class ReactionModel(abc.ABC):

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        # create model folder if not already existing
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

    def load_checkpoint(self, path):
        """Load a pre-trained model from a provided path"""
        pass

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
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

