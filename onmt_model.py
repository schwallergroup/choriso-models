import argparse
import os
import wandb
import yaml
import tempfile
import pandas as pd

from benchmark_models import ReactionModel, BenchmarkPipeline
from model_args import ReactionModelArgs
from utils import prepare_parser, csv_to_txt, set_pythonpath

from onmt.opts import train_opts, translate_opts, dynamic_prepare_opts, config_opts


def substitute_dataset_name(d, dataset_name):
    if isinstance(d, str):
        return d.replace("${{dataset_name}}", dataset_name)
    elif isinstance(d, dict):
        return {k: substitute_dataset_name(v, dataset_name) for k, v in d.items()}
    elif isinstance(d, list):
        return [substitute_dataset_name(v, dataset_name) for v in d]
    else:
        return d


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

    def preprocess(self, dataset="cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        data_dir = os.path.join(os.path.dirname(self.model_dir), "data", dataset)
        # transfer the tsv files, if not yet done. Please set clone_name to the name of your git clone of the dataset
        self.setup_tsv(dataset=dataset, clone_name="cjhif-dataset")

        csv_to_txt(data_dir)
        with open("run_config.yaml", "r") as yaml_file:
            yaml_content = yaml.full_load(yaml_file)
            yaml_content["dataset_name"] = dataset
            yaml_content = substitute_dataset_name(yaml_content, dataset)

        with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
            with open(tmp.name, "w") as tmp_yaml:
                yaml.dump(yaml_content, tmp_yaml, default_flow_style=False, sort_keys=False, allow_unicode=True)

            cmd = f"export MKL_SERVICE_FORCE_INTEL=1\n" \
                  f"onmt_build_vocab -config {tmp.name} -src_seq_length 1000 -tgt_seq_length 1000 " \
                  f"-src_vocab_size 1000 -tgt_vocab_size 1000 -n_sample -1"

            os.system(cmd)

    def train(self, dataset="cjhif"):
        """Train the reaction model. Should also contain validation and test steps"""
        with open("run_config.yaml", "r") as yaml_file:
            yaml_content = yaml.full_load(yaml_file)
            yaml_content["dataset_name"] = dataset
            yaml_content = substitute_dataset_name(yaml_content, dataset)

        with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
            with open(tmp.name, "w") as tmp_yaml:
                yaml.dump(yaml_content, tmp_yaml, default_flow_style=False, sort_keys=False, allow_unicode=True)

            cmd = f"export MKL_SERVICE_FORCE_INTEL=1\n" \
                  f"onmt_train -config {tmp.name} -seed 42 -gpu_ranks 0 -param_init 0 -param_init_glorot " \
                  f"-max_generator_batches 32 -batch_type tokens -batch_size 6144 -normalization tokens " \
                  f"-max_grad_norm 0 -accum_count 4 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam " \
                  f"-warmup_steps 8000 -learning_rate 2 -label_smoothing 0.0 -layers 4 -hidden_size  384 " \
                  f"-word_vec_size 384 -encoder_type transformer -decoder_type transformer -dropout 0.1 " \
                  f"-position_encoding -share_embeddings -global_attention general -global_attention_function softmax " \
                  f"-self_attn_type scaled-dot -heads 8 -transformer_ff 2048"

            os.system(cmd)

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""

        with open("run_config.yaml", "r") as yaml_file:
            yaml_content = yaml.full_load(yaml_file)
        best_model_step = yaml_content["train_steps"]
        # TODO make best model configurable
        best_model = os.path.join(self.model_dir, "runs", dataset, f"{dataset}_step_{best_model_step}.pt")
        out_file = os.path.join(self.model_dir, "runs", dataset, "predictions.txt")
        tgt_file = f"../data/{dataset}/tgt-test.txt"

        n_outputs = 5
        cmd = f"export MKL_SERVICE_FORCE_INTEL=1\n" \
              f"onmt_translate -model {best_model} -gpu 0 --src ../data/{dataset}/src-test.txt " \
              f"--tgt {tgt_file} --output {out_file} --n_best {n_outputs} --beam_size 10 " \
              f"--max_length 300 --batch_size 64"

        os.system(cmd)

        # TODO implement evaluation, standardize output format
        predictions = [result.replace(" ", "").replace("\n", "") for result in open(out_file, "r").readlines()]
        predictions = [predictions[i:i + n_outputs] for i in range(0, len(predictions), n_outputs)]

        targets = [target.replace(" ", "").replace("\n", "") for target in open(tgt_file, "r").readlines()]

        reaction_file = os.path.join(os.path.dirname(tgt_file), "test.tsv")
        reactions = pd.read_csv(reaction_file, sep="\t", error_bad_lines=False)["canonic_rxn"].tolist()

        # Create a list of column names for the predictions
        pred_cols = [f"pred_{i}" for i in range(len(predictions[0]))]

        # Create a list of dictionaries representing each row of the DataFrame
        rows = []
        for rxn, prod, preds in zip(reactions, targets, predictions):
            row = {"canonical_rxn": rxn, "target": prod}
            row.update({pred_col: pred for pred_col, pred in zip(pred_cols, preds)})
            rows.append(row)

        # Create the DataFrame
        df = pd.DataFrame(rows)

        # Save the DataFrame to a CSV file
        csv_file = f"./runs/{dataset}.all_results.csv"
        df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONMT parser')

    prepare_parser(parser)

    args = parser.parse_args()

    set_pythonpath(path=os.getcwd())

    os.chdir("OpenNMT_Transformer")

    reaction_model = OpenNMT()
    pipeline = BenchmarkPipeline(model=reaction_model)

    pipeline.run_mode_from_args(args)
