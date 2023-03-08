import os
import sys
import torch
import wandb
import pandas as pd
import json
from transformers import AutoConfig, AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
from transformers.integrations import WandbCallback, CodeCarbonCallback

from reaction_model import *
from utils import prepare_data, ReactionForwardDataset


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
        wandb.init(project="OpenNMT_Transformer")
        wandb.tensorboard.patch(root_logdir=os.path.join(self.model_dir, "log_dir"))
        os.system("sh OpenNMT_Transformer/train.sh")
        wandb.finish()

    def predict(self, data):
        """Predict provided data with the reaction model"""
        os.system("sh OpenNMT_Transformer/predict.sh")


class HuggingFaceTransformerPretrained(ReactionModel):

    def __init__(self, pretrained_model: str, model_kwargs: dict = None):
        self.name = f"HuggingFaceTransformerPretrained_{pretrained_model}"
        super().__init__()
        model_kwargs = {} if model_kwargs is None else model_kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, **model_kwargs)
        self.model = AutoModel.from_pretrained(pretrained_model, **model_kwargs)

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


class HuggingFaceTransformerCustom(ReactionModel):

    def __init__(self, model_architecture: str, train_args: dict, model_args: dict = None):
        name_suffix = model_architecture.split("/")[-1]
        self.name = f"HuggingFaceTransformerCustom_{name_suffix}"
        super().__init__()
        # if we already have saved a config, load it
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            print("WARNING: Saved parameters are used! If you want to change the architecture, please delete the "
                  "config file.")
            config = AutoConfig.from_pretrained(config_path)
        else:
            model_kwargs = {} if model_args is None else model_args
            config = AutoConfig.from_pretrained(model_architecture, **model_kwargs)

        # choose model based on config
        self.model = AutoModelForSeq2SeqLM.from_config(config)

        # choose tokenizer based on the model architecture
        self.tokenizer = AutoTokenizer.from_pretrained(model_architecture)

        # route the given dir to the model dir
        train_args["output_dir"] = os.path.join(self.model_dir, train_args["output_dir"])
        train_args["logging_dir"] = os.path.join(self.model_dir, train_args["logging_dir"])
        self.train_args = Seq2SeqTrainingArguments(**train_args)

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def preprocess(self, dataset: str = "cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        root_dir = os.path.dirname(self.model_dir)

        preprocessed_dir = os.path.join(self.model_dir, "preprocessed")
        if not os.path.exists(preprocessed_dir):
            os.mkdir(preprocessed_dir)

        data_dir = os.path.join(root_dir, "data", dataset)

        for file_name in ["val", "train"]:
            save_path = os.path.join(preprocessed_dir, f"{file_name}_dataset.json")

            # if there is saved data, load it
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    reaction_dataset = json.load(f)

            # else do preprocessing and save it in a json file so it can be retrieved more easily
            else:
                reactions = pd.read_csv(os.path.join(data_dir, f"{file_name}.tsv"), sep="\t", error_bad_lines=False)

                split_reactions = prepare_data(reactions[:1000], rsmiles_col="canonic_rxn")

                split_reactions = split_reactions.to_dict("list")

                reaction_dataset = self.tokenizer(split_reactions["reactants"], text_target=split_reactions["products"],
                                               truncation=True, max_length=1000)
                print("reaction_dataset: ", reaction_dataset)
                reaction_dataset = dict(reaction_dataset)
                print("type of reaction_dataset: ", type(reaction_dataset))
                with open(save_path, "w") as outfile:
                    json.dump(reaction_dataset, outfile)

            if file_name == "train":
                self.train_dataset = reaction_dataset
            else:
                self.val_dataset = reaction_dataset

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        print("self.train_dataset.keys: ", self.train_dataset.keys())
        print("self.val_dataset: ", self.val_dataset.keys())
        breakpoint()
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        trainer = Seq2SeqTrainer(model=self.model,
                                 args=self.train_args,
                                 train_dataset=self.train_dataset,
                                 eval_dataset=self.val_dataset,
                                 data_collator=data_collator,
                                 tokenizer=self.tokenizer,
                                 callbacks=[CodeCarbonCallback])
        trainer.train()

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

    def predict(self):
        self.model.predict(data=None)


if __name__ == "__main__":
    train_args = {
        "output_dir": 'results',  # output directory
        "evaluation_strategy": "epoch",
        "learning_rate": 2e-5,
        "save_total_limit": 3,
        "num_train_epochs": 3,  # total number of training epochs
        "per_device_train_batch_size": 16,  # batch size per device during training
        "per_device_eval_batch_size": 64,  # batch size for evaluation
        "warmup_steps": 500,  # number of warmup steps for learning rate scheduler
        "weight_decay": 0.01,  # strength of weight decay
        "logging_dir": 'logs',  # directory for storing logs
        "logging_steps": 10,
    }
    reaction_model = HuggingFaceTransformerCustom(model_architecture="t5-small",
                                                  train_args=train_args)
    pipeline = BenchmarkPipeline(model=reaction_model)
    pipeline.run_train_pipeline()
    # pipeline.predict()
