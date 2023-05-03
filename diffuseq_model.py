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
from utils import csv_to_jsonl, prepare_parser, set_pythonpath
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

        self.hidden_dim = 128
        self.lr = 0.0001
        self.diff_steps = 2000
        self.noise_schedule = "sqrt"
        self.schedule_sampler = "lossaware"
        self.seed = 42

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

        # transfer the tsv files, if not yet done. Please set clone_name to the name of your git clone of the dataset
        self.setup_tsv(dataset=dataset, clone_name="cjhif-dataset")
        csv_to_jsonl(data_dir, target_dir_dataset)

        vocab_file = os.path.join(target_dir_dataset, "vocab.txt")
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

                # we only need the key for diffuseq, index is handled in the code
                vocab_list = [key for key in vocab_dict.keys()]

                with open(vocab_file, "w") as f:
                    for element in vocab_list:
                        f.write(element + "\n")

    def train(self, dataset="cjhif"):
        """Train the reaction model. Should also contain validation and test steps"""

        processed_dir = os.path.join(self.model_dir, "processed")
        data_dir = os.path.join(processed_dir, dataset)
        vocab_file = os.path.join(data_dir, "vocab.txt")
        os.chdir(os.path.join(self.model_dir, "scripts"))

        # TODO make more flexible for different parameters
        cmd = f"export MKL_SERVICE_FORCE_INTEL=1\n " \
              f"python -m torch.distributed.launch " \
              f"--nproc_per_node=1 " \
              f"--master_port=12233 " \
              f"--use_env run_train.py " \
              f"--diff_steps {self.diff_steps} " \
              f"--lr {self.lr} " \
              f"--learning_steps 80000 " \
              f"--save_interval 10000 " \
              f"--seed {self.seed} " \
              f"--noise_schedule {self.noise_schedule} " \
              f"--hidden_dim {self.hidden_dim} " \
              f"--bsz 2048 " \
              f"--dataset {dataset} " \
              f"--data_dir {data_dir} " \
              f"--vocab {vocab_file} " \
              f"--seq_len 512 " \
              f"--schedule_sampler {self.schedule_sampler} "

        print(cmd)

        os.system(cmd)

    def predict_once(self, dataset="cjhif", seed=123):
        """Predict provided data with the reaction model"""
        model_file = f"diffuseq_{dataset}_h{self.hidden_dim}_lr{self.lr}" \
                     f"_t{self.diff_steps}_{self.noise_schedule}_{self.schedule_sampler}" \
                     f"_seed{self.seed}"

        os.chdir(os.path.join(self.model_dir, "scripts"))

        cmd = f"export MKL_SERVICE_FORCE_INTEL=1\n " \
              f"python -u run_decode.py " \
              f"--model_dir diffusion_models/{model_file} " \
              f"--seed {seed} --split test"

        print(cmd)

        os.system(cmd)

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""

        preds = []
        seeds = [1, 3, 5, 7, 9, 11, 13, 17, 19, 23]
        for seed in seeds:
            # predict with different seeds
            self.predict_once(dataset=dataset, seed=seed)

            # load predictions

        # save predictions in csv file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DiffuSeq parser')

    prepare_parser(parser)

    args = parser.parse_args()

    set_pythonpath(path=os.getcwd())

    os.chdir("DiffuSeq")

    reaction_model = DiffuSeq()
    pipeline = BenchmarkPipeline(model=reaction_model)

    pipeline.run_mode_from_args(args)
