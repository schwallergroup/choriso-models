import os

from benchmark_models import ReactionModel, BenchmarkPipeline
from model_args import ReactionModelArgs

from Graph2SMILES.preprocess import get_preprocess_parser
from Graph2SMILES.train import get_train_parser
from Graph2SMILES.predict import get_predict_parser


class G2SArgs(ReactionModelArgs):

    def __init__(self):
        super().__init__()
        pass

    def preprocess_args(self):
        return get_preprocess_parser()

    def training_args(self):
        return get_train_parser()

    def predict_args(self):
        return get_predict_parser()


class G2S(ReactionModel):

    def __init__(self):
        self.name = "Graph2SMILES"
        super().__init__()
        self.args = G2SArgs()

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh Graph2SMILES/scripts/preprocess.sh")

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system("sh Graph2SMILES/scripts/train_g2s.sh")

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""
        os.system("sh Graph2SMILES/scripts/predict.sh")


if __name__ == "__main__":
    reaction_model = G2S()
    pipeline = BenchmarkPipeline(model=reaction_model)
    # pipeline.run_train_pipeline()
    pipeline.predict(dataset="cjhif")
