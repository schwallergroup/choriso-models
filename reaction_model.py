import abc
import os
from model_args import ReactionModelArgs
from utils import transfer_data


class ReactionModel(abc.ABC):

    name:  str
    args: ReactionModelArgs

    def __init__(self):
        assert self.name is not None, "Please set name before calling super-class"
        # get the path of this file
        this_dir = os.path.dirname(os.path.realpath(__file__))

        # define the path of the current model. point to the directory
        self.model_dir = os.path.join(this_dir, self.name)

        # if it doesn't exist, create it
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def setup_tsv(self, dataset="cjhif", clone_name="cjhif-dataset"):
        """Setup the tsv files for the model. Skip if already exists"""
        origin_data_dir = os.path.join(os.path.dirname(os.path.dirname(self.model_dir)), clone_name, "data", "processed")
        target_data_dir = os.path.join(os.path.dirname(self.model_dir), "data", dataset)
        transfer_data(origin_data_dir, target_data_dir)

    def preprocess(self, dataset="cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        pass

    def train(self, dataset="cjhif"):
        """Train the reaction model. Should also contain validation and test steps"""
        pass

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""
        pass


