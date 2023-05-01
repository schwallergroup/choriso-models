import os
import argparse

from benchmark_models import ReactionModel, BenchmarkPipeline
from model_args import ReactionModelArgs
from utils import prepare_parser, csv_to_txt, set_pythonpath, transfer_data

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

        self.model_name = "g2s_series_rel"

        self.exp_no = 0

        self.bs = 30
        self.T = 1.0
        self.nbest = 30
        self.mpn_type = "dgat"

        self.rel_pos = "emb_only"

    def preprocess(self, dataset="cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        data_dir = os.path.join(os.path.dirname(self.model_dir), "data", dataset)

        # transfer the tsv files, if not yet done. Please set clone_name to the name of your git clone of the dataset
        self.setup_tsv(dataset=dataset, clone_name="cjhif-dataset")

        csv_to_txt(data_dir)

        prefix = f"{dataset}_{self.model_name}"
        task = "reaction_prediction"

        cmd = f"python preprocess.py " \
              f"--model={self.model_name} " \
              f"--data_name={dataset} " \
              f"--task={task} " \
              f"--representation_start=smiles " \
              f"--representation_end=smiles " \
              f"--train_src=../data/{dataset}/src-train.txt " \
              f"--train_tgt=../data/{dataset}/tgt-train.txt " \
              f"--val_src=../data/{dataset}/src-val.txt " \
              f"--val_tgt=../data/{dataset}/tgt-val.txt " \
              f"--test_src=../data/{dataset}/src-test.txt " \
              f"--test_tgt=../data/{dataset}/tgt-test.txt " \
              f"--log_file={prefix}.preprocess.log " \
              f"--preprocess_output_path=./preprocessed/{prefix}/ " \
              f"--seed=42 " \
              f"--max_src_len=1024 " \
              f"--max_tgt_len=1024 " \
              f"--num_workers=8 "

        os.system(cmd)

        # os.system("sh Graph2SMILES/scripts/preprocess.sh")

    def train(self, dataset="cjhif"):
        """Train the reaction model. Should also contain validation and test steps"""
        # os.system("sh Graph2SMILES/scripts/train_g2s.sh")

        load_from = ""
        task = "reaction_prediction"
        max_rel_pos = 4
        acc_count = 4
        enc_pe = "none"
        enc_h = 256
        batch_size = 4096
        enc_emb_scale = "sqrt"
        max_steps = 40
        enc_layer = 4
        enc_norm = "none"
        enc_sc = "none"
        batch_type = "tokens"
        rel_buckets = 11

        attn_layer = 6
        lr = 4
        dropout = 0.3

        prefix = f"{dataset}_{self.model_name}"

        cmd = f"python train.py " \
              f"--model={self.model_name} " \
              f"--data_name={dataset} " \
              f"--task={task} " \
              f"--representation_end=smiles " \
              f"--load_from={load_from} " \
              f"--train_bin=./preprocessed/{prefix}/train_0.npz " \
              f"--valid_bin=./preprocessed/{prefix}/val_0.npz " \
              f"--log_file={prefix}.train.{self.exp_no}.log " \
              f"--vocab_file=./preprocessed/{prefix}/vocab_smiles.txt " \
              f"--save_dir=./checkpoints/{prefix}.{self.exp_no} " \
              f"--embed_size=256 " \
              f"--mpn_type={self.mpn_type} " \
              f"--encoder_num_layers={enc_layer} " \
              f"--encoder_hidden_size={enc_h} " \
              f"--encoder_norm={enc_norm} " \
              f"--encoder_skip_connection={enc_sc} " \
              f"--encoder_positional_encoding={enc_pe} " \
              f"--encoder_emb_scale={enc_emb_scale} " \
              f"--attn_enc_num_layers={attn_layer} " \
              f"--attn_enc_hidden_size=256 " \
              f"--attn_enc_heads=8 " \
              f"--attn_enc_filter_size=2048 " \
              f"--rel_pos={self.rel_pos} " \
              f"--rel_pos_buckets={rel_buckets} " \
              f"--decoder_num_layers=6 " \
              f"--decoder_hidden_size=256 " \
              f"--decoder_attn_heads=8 " \
              f"--decoder_filter_size=2048 " \
              f"--dropout={dropout} " \
              f"--attn_dropout={dropout} " \
              f"--max_relative_positions={max_rel_pos} " \
              f"--seed=42 " \
              f"--epoch=200000 " \
              f"--max_steps={max_steps} " \
              f"--warmup_steps=8000 " \
              f"--lr={lr} " \
              f"--weight_decay=0.0 " \
              f"--clip_norm=20.0 " \
              f"--batch_type={batch_type} " \
              f"--train_batch_size={batch_size} " \
              f"--valid_batch_size={batch_size} " \
              f"--predict_batch_size={batch_size} " \
              f"--accumulation_count={acc_count} " \
              f"--num_workers=8 " \
              f"--beam_size=5 " \
              f"--predict_min_len=1 " \
              f"--predict_max_len=512 " \
              f"--log_iter=100 " \
              f"--eval_iter=5000 " \
              f"--save_iter=5000 " \
              f"--compute_graph_distance "

        os.system(cmd)

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""
        # os.system("sh Graph2SMILES/scripts/predict.sh")

        prefix = f"{dataset}_{self.model_name}"

        # TODO make this automatic
        checkpoint = f"./checkpoints/{prefix}.{self.exp_no}/model.20000_3.pt"

        cmd = f"python predict.py " \
              f"--do_predict " \
              f"--do_score " \
              f"--model={self.model_name} " \
              f"--data_name={dataset} " \
              f"--test_bin=./preprocessed/{prefix}/test_0.npz " \
              f"--test_tgt=../data/{dataset}/tgt-test.txt " \
              f"--result_file=./results/{prefix}.{self.exp_no}.result.txt " \
              f"--log_file={prefix}.predict.{self.exp_no}.log " \
              f"--load_from={checkpoint} " \
              f"--mpn_type={self.mpn_type} " \
              f"--rel_pos={self.rel_pos} " \
              f"--seed=42 " \
              f"--batch_type=tokens " \
              f"--predict_batch_size=4096 " \
              f"--beam_size={self.bs} " \
              f"--n_best={self.nbest} " \
              f"--temperature={self.T} " \
              f"--predict_min_len=1 " \
              f"--predict_max_len=512 " \
              f"--log_iter=100 "

        os.system(cmd)

        # TODO implement evaluation, standardize output format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='G2S parser')

    prepare_parser(parser)

    args = parser.parse_args()

    set_pythonpath(path=os.getcwd())

    os.chdir("Graph2SMILES")

    reaction_model = G2S()
    pipeline = BenchmarkPipeline(model=reaction_model)

    pipeline.run_mode_from_args(args)
