import argparse
import os
import wandb

from benchmark_models import ReactionModel, BenchmarkPipeline
from model_args import ReactionModelArgs
from utils import csv_to_jsonl

class DiffuSeqArgs(ReactionModelArgs):

    def __init__(self):
        super().__init__()
        pass

    def preprocess_args(self):
        parser = argparse.ArgumentParser(description='preprocess')
        return parser

    def training_args(self):
        parser = argparse.ArgumentParser(description='train')
        return parser

    def predict_args(self):
        parser = argparse.ArgumentParser(description='predict')
        return parser


class DiffuSeq(ReactionModel):

    def __init__(self):
        self.name = "DiffuSeq"
        super().__init__()

        self.args = DiffuSeqArgs()

    def preprocess(self, dataset="cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        # TODO make flexible for different datasets
        data_dir = os.path.join(os.path.dirname(self.model_dir), "data", dataset)
        target_dir = os.path.join(self.model_dir, "processed")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        target_dir_dataset = os.path.join(target_dir, dataset)
        if not os.path.exists(target_dir_dataset):
            os.makedirs(target_dir_dataset)

        csv_to_jsonl(data_dir, target_dir_dataset)


    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system(f"sh DiffuSeq/scripts/train.sh")

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""
        os.system("sh DiffuSeq/scripts/run_decode.sh")


if __name__ == "__main__":
    reaction_model = DiffuSeq()
    pipeline = BenchmarkPipeline(model=reaction_model)
    # pipeline.run_train_pipeline()
    pipeline.predict(dataset="cjhif")