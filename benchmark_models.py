from reaction_model import *


class MolecularTransformer(ReactionModel):

    def __init__(self):
        self.name = "MolecularTransformer"
        super().__init__()

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh MolecularTransformer/preprocess.sh")

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system("sh MolecularTransformer/train.sh")

    def predict(self, data):
        """Predict provided data with the reaction model"""
        pass


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

    def predict(self, dataset):
        self.model.predict(dataset=dataset)

    def run_pipeline(self):
        self.model.preprocess()
        self.model.train()
        self.model.predict(dataset="cjhif")
