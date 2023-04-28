import argparse
import os
import wandb

from benchmark_models import ReactionModel, BenchmarkPipeline
from model_args import ReactionModelArgs
from utils import prepare_parser, csv_to_txt, set_pythonpath

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

    def preprocess(self, dataset="cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        data_dir = os.path.join(os.path.dirname(self.model_dir), "data", dataset)
        # transfer the tsv files, if not yet done. Please set clone_name to the name of your git clone of the dataset
        self.setup_tsv(dataset=dataset, clone_name="cjhif-dataset")

        csv_to_txt(data_dir)

        cmd = f"export MKL_SERVICE_FORCE_INTEL=1\n" \
              f"onmt_build_vocab -config run_config.yaml -src_seq_length 1000 -tgt_seq_length 1000 -src_vocab_size 1000 " \
              f"-tgt_vocab_size 1000 -n_sample -1"

        os.system(cmd)

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""

        cmd = f"export MKL_SERVICE_FORCE_INTEL=1\n" \
              f"onmt_train -config run_config.yaml -seed 42 -gpu_ranks 0 -param_init 0 -param_init_glorot " \
              f"-max_generator_batches 32 -batch_type tokens -batch_size 6144 -normalization tokens -max_grad_norm 0  " \
              f"-accum_count 4 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 " \
              f"-learning_rate 2 -label_smoothing 0.0 -layers 4 -hidden_size  384 -word_vec_size 384 " \
              f"-encoder_type transformer -decoder_type transformer -dropout 0.1 -position_encoding -share_embeddings " \
              f"-global_attention general -global_attention_function softmax -self_attn_type scaled-dot -heads 8 " \
              f"-transformer_ff 2048"

        os.system(cmd)

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""
        # TODO make best model configurable
        best_model = os.path.join(self.model_dir, "runs", dataset, "step_200000.pt")
        out_file = os.path.join(self.model_dir, "runs", dataset, "predictions.txt")

        cmd = f"export MKL_SERVICE_FORCE_INTEL=1\n" \
              f"onmt_translate -model {best_model} -gpu 0 --src ../data/{dataset}/src-test.txt " \
              f"--tgt ../data/{dataset}/tgt-test.txt --output {out_file} --n_best 5 --beam_size 10 --max_length 300 " \
              f"--batch_size 64"

        os.system(cmd)

        # TODO implement evaluation, standardize output format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONMT parser')

    prepare_parser(parser)

    args = parser.parse_args()

    set_pythonpath(path=os.getcwd())

    os.chdir("OpenNMT_Transformer")

    reaction_model = OpenNMT()
    pipeline = BenchmarkPipeline(model=reaction_model)

    if args.mode == "t":
        pipeline.run_train_pipeline()

    elif args.mode == "p":
        pipeline.predict(dataset=args.dataset)
