import sys
import argparse
import os
import wandb
import tempfile
import pandas as pd
import json
from tokenizers import models as tokenizer_models
from tokenizers import Regex, Tokenizer, pre_tokenizers
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

from benchmark_models import ReactionModel, BenchmarkPipeline
from model_args import ReactionModelArgs
from utils import csv_to_jsonl
import torch.multiprocessing


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

        vocab_file = os.path.join(target_dir_dataset, "vocab.json")
        if not os.path.exists(vocab_file):
            # For obtaining vocabulary
            pattern = Regex(
                r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])")
            # define simple Tokenizer that creates the vocabulary from the regex pattern
            chem_tokenizer = Tokenizer(tokenizer_models.WordLevel())

            chem_tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern, "isolated")

            trainer = WordLevelTrainer(special_tokens=[])

            paths = [os.path.join(data_dir, f"{file_name}.tsv") for file_name in ["train", "val"]]

            # obtain reaction smiles and merge them into one dataframe
            reactions = [pd.read_csv(path, sep="\t")["canonic_rxn"] for path in paths]
            reactions = pd.concat(reactions)

            # open a temporary file for the train+val smiles so it doesn't need to be stored
            with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
                reactions.to_csv(tmp, sep="\t", index=False, header=False)

                chem_tokenizer.train([tmp.name], trainer)

                chem_tokenizer = PreTrainedTokenizerFast(tokenizer_object=chem_tokenizer)

                # save the vocabulary
                vocab_dict = chem_tokenizer.vocab
                with open(vocab_file, "w") as f:
                    json.dump(vocab_dict, f)
                # chem_tokenizer.save_vocabulary(vocab_file)

    def train(self, dataset="cjhif"):
        """Train the reaction model. Should also contain validation and test steps"""

        processed_dir = os.path.join(self.model_dir, "processed")
        data_dir = os.path.join(processed_dir, dataset)
        vocab_file = os.path.join(data_dir, "vocab.json")
        os.chdir(os.path.join(self.model_dir, "scripts"))

        # TODO make more flexible for different parameters
        cmd = f"export MKL_SERVICE_FORCE_INTEL=1\n " \
              f"python -m torch.distributed.launch --nproc_per_node=1 --master_port=12233 --use_env run_train.py " \
              f"--diff_steps 2000 --lr 0.0001 --learning_steps 80000 --save_interval 10000 --seed 102 " \
              f"--noise_schedule sqrt --hidden_dim 128 --bsz 2048 --dataset {dataset} --data_dir {data_dir} " \
              f"--vocab {vocab_file} --seq_len 128 --schedule_sampler lossaware --notes {dataset} "

        print(cmd)

        os.system(cmd)

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""
        torch.multiprocessing.set_sharing_strategy('file_system')

        os.chdir(os.path.join(self.model_dir, "scripts"))

        cmd = f"export MKL_SERVICE_FORCE_INTEL=1\n " \
              f"python -u run_decode.py " \
              f"--model_dir diffusion_models/diffuseq_final_model " \
              f"--seed 123 --split test"

        print(cmd)

        os.system(cmd)


if __name__ == "__main__":
    # Get the current value of PYTHONPATH (if it exists)
    pythonpath = os.getenv('PYTHONPATH', '')

    # Add ~/reaction_forward to PYTHONPATH
    pythonpath += ':' + os.getcwd()

    # Set the updated PYTHONPATH
    os.environ['PYTHONPATH'] = pythonpath
    sys.path.append(pythonpath)

    os.chdir("DiffuSeq")

    reaction_model = DiffuSeq()
    pipeline = BenchmarkPipeline(model=reaction_model)
    # pipeline.run_train_pipeline()
    pipeline.predict(dataset="cjhif")
