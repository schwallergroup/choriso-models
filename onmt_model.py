import argparse
import os
import wandb

from benchmark_models import ReactionModel, BenchmarkPipeline
from model_args import ReactionModelArgs

from onmt.opts import train_opts, translate_opts, dynamic_prepare_opts, config_opts


class OpenNMTArgs(ReactionModelArgs):

    def __init__(self):
        super().__init__()
        pass

    def preprocess_args(self):
        parser = argparse.ArgumentParser(description='preprocess')
        dynamic_prepare_opts(parser, build_vocab_only=True)
        return parser

    def training_args(self):
        parser = argparse.ArgumentParser(description='train')
        train_opts(parser)
        return parser

    def predict_args(self):
        parser = argparse.ArgumentParser(description='predict')
        config_opts(parser)
        translate_opts(parser, dynamic=True)
        return parser


class OpenNMT(ReactionModel):

    def __init__(self):
        self.name = "OpenNMT_Transformer"
        super().__init__()

        self.args = OpenNMTArgs()

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh OpenNMT_Transformer/preprocess.sh")
        # onmt_preprocess()

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        log_dir = os.path.join(self.model_dir, "log_dir", "*") # , datetime.datetime.now().strftime("%b-%d_%H-%M-%S"))
        wandb.init(project="OpenNMT_Transformer", sync_tensorboard=True)
        # wandb.tensorboard.patch(root_logdir=log_dir)
        os.system(f"sh OpenNMT_Transformer/train.sh")
        # onmt_train()
        wandb.finish()

    def predict(self, data):
        """Predict provided data with the reaction model"""
        os.system("sh OpenNMT_Transformer/predict.sh")


if __name__ == "__main__":
    reaction_model = OpenNMT()
    pipeline = BenchmarkPipeline(model=reaction_model)
    pipeline.run_train_pipeline()
