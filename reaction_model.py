import abc
import os


class Parameters:

    def __init__(self, params: dict):
        self.params = params


class ReactionModel(abc.ABC):

    name:  str

    def __init__(self):
        assert self.name is not None, "PLease set name before calling super-class"
        # get the path of this file
        this_dir = os.path.dirname(os.path.realpath(__file__))

        # define the path of the current model. point to the directory
        self.model_dir = os.path.join(this_dir, self.name)

        # if it doesn't exist, create it
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

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


