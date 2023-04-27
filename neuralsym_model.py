import argparse
import os
import wandb

from benchmark_models import ReactionModel, BenchmarkPipeline
from model_args import ReactionModelArgs
from utils import add_mode_parser


class NeuralsymArgs(ReactionModelArgs):

    def __init__(self):
        super().__init__()
        pass

    def preprocess_args(self):
        parser = argparse.ArgumentParser(description='preprocess')
        pass
        return parser

    def training_args(self):
        parser = argparse.ArgumentParser(description='train')
        pass
        return parser

    def predict_args(self):
        parser = argparse.ArgumentParser(description='predict')
        pass
        return parser


class Neuralsym(ReactionModel):

    def __init__(self):
        self.name = "Neuralsym"
        super().__init__()

        self.args = NeuralsymArgs()

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("python neuralsym/prepare_data.py")  # TODO add args

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        wandb.init(project="Neuralsym", sync_tensorboard=True)
        os.system(f"sh neuralsym/train.sh ")  # TODO add args
        wandb.finish()

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""
        os.system("sh neuralsym/infer_all.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONMT parser')

    add_mode_parser(parser)

    args = parser.parse_args()

    reaction_model = Neuralsym()
    pipeline = BenchmarkPipeline(model=reaction_model)

    if args.mode == "t":
        pipeline.run_train_pipeline()

    elif args.mode == "p":
        pipeline.predict(dataset="cjhif")
