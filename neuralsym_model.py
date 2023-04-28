import argparse
import os
import wandb

from benchmark_models import ReactionModel, BenchmarkPipeline
from model_args import ReactionModelArgs
from utils import prepare_parser, set_pythonpath


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

    def preprocess(self, dataset="cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        # transfer the tsv files, if not yet done. Please set clone_name to the name of your git clone of the dataset
        self.setup_tsv(dataset=dataset, clone_name="cjhif-dataset")

        os.system(f"python prepare_data.py --dataset {dataset}")  # TODO add args

    def train(self, dataset="cjhif"):
        """Train the reaction model. Should also contain validation and test steps"""
        """wandb.init(project="Neuralsym", sync_tensorboard=True)
        os.system(f"sh neuralsym/train.sh ")  # TODO add args
        wandb.finish()"""

        model = "Highway"
        seed = 77777777

        cmd = f"CUDA_LAUNCH_BLOCKING=1\n" \
              f"python train.py " \
              f"--model {model} " \
              f"--expt_name {model}_{seed}_depth0_dim300_lr1e3_stop2_fac30_pat1 " \
              f"--log_file {model}_{seed}_depth0_dim300_lr1e3_stop2_fac30_pat1 " \
              f"--do_test " \
              f"--reacfps_prefix {dataset}_to_32681_reac_fps" \
              f"--labels_prefix {dataset}_to_32681_labels " \
              f"--csv_prefix {dataset}_to_32681_csv " \
              f"--bs 300 " \
              f"--bs_eval 300 " \
              f"--random_seed {seed} " \
              f"--learning_rate 1e-3 " \
              f"--epochs 30 " \
              f"--early_stop " \
              f"--early_stop_patience 2 " \
              f"--depth 0 " \
              f"--hidden_size 300 " \
              f"--lr_scheduler_factor 0.3 " \
              f"--lr_scheduler_patience 1 " \
              f"--checkpoint"

        os.system(cmd)

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""
        # os.system("sh neuralsym/infer_all.sh")

        model = "Highway"
        seed = 77777777

        cmd = f"python infer_all.py " \
              f"--csv_prefix {dataset}_1000000dim_2rad_to_32681_csv " \
              f"--labels_prefix {dataset}_1000000dim_2rad_to_32681_labels " \
              f"--templates_file {dataset}_training_templates " \
              f"--rxn_smi_prefix {dataset}_clean_rxnsmi_noreagent_allmapped_canon " \
              f"--log_file 'infer_77777777_highway_depth0_dim300' " \
              f"--prodfps_prefix {dataset}_1000000dim_2rad_to_32681_prod_fps " \
              f"--hidden_size 300 " \
              f"--depth 0 " \
              f"--topk 200 " \
              f"--maxk 200 " \
              f"--model Highway " \
              f"--expt_name '{model}_{seed}_depth0_dim300_lr1e3_stop2_fac30_pat1' " \
              f"--seed {seed} " \

        os.system(cmd)

        # TODO implement evaluation, standardize output format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONMT parser')

    prepare_parser(parser)

    args = parser.parse_args()

    set_pythonpath(path=os.getcwd())

    os.chdir("neuralsym")

    reaction_model = Neuralsym()
    pipeline = BenchmarkPipeline(model=reaction_model)

    if args.mode == "t":
        pipeline.run_train_pipeline()

    elif args.mode == "p":
        pipeline.predict(dataset=args.dataset)
