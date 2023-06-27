import argparse
import os
from benchmark_models import ReactionModel, BenchmarkPipeline
from utils import prepare_parser, set_pythonpath


class Neuralsym(ReactionModel):

    def __init__(self):
        self.name = "neuralsym"
        super().__init__()

    def get_cmd(self, model, seed, dataset, mode):

        cmd = f"CUDA_LAUNCH_BLOCKING=1\n" \
              f"python train.py " \
              f"--model {model} " \
              f"--expt_name {model}_{seed}_depth0_dim300_lr1e3_stop2_fac30_pat1 " \
              f"--log_file {model}_{seed}_depth0_dim300_lr1e3_stop2_fac30_pat1 " \
              f"--reacfps_prefix {dataset}_to_32681_reac_fps " \
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
              f"--checkpoint " \
              f"--dataset {dataset} "

        if mode == "train":
            cmd += "--do_train "

        if mode == "test":
            cmd += "--do_test "

        return cmd

    def preprocess(self, dataset="cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""

        os.system(f"python prepare_data.py --output_file_prefix {dataset} --dataset {dataset}")  # TODO add args

    def train(self, dataset="cjhif"):
        """Train the reaction model. Should also contain validation and test steps"""

        model = "Highway"
        seed = 77777777

        cmd = self.get_cmd(model, seed, dataset, "train")

        os.system(cmd)

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""

        model = "Highway"
        seed = 77777777

        cmd = f"python infer_all.py " \
              f"--csv_prefix {dataset}_to_32681_csv " \
              f"--labels_prefix {dataset}_to_32681_labels " \
              f"--templates_file training_templates.txt " \
              f"--rxn_smi_prefix {dataset}_clean_rxnsmi_noreagent_allmapped_canon " \
              f"--log_file 'infer_77777777_highway_depth0_dim300' " \
              f"--prodfps_prefix {dataset}_to_32681_prod_fps " \
              f"--hidden_size 300 " \
              f"--depth 0 " \
              f"--topk 200 " \
              f"--maxk 200 " \
              f"--model Highway " \
              f"--expt_name '{model}_{seed}_depth0_dim300_lr1e3_stop2_fac30_pat1' " \
              f"--seed {seed} " \
              f"--dataset {dataset}"

        # cmd = self.get_cmd(model, seed, dataset, "test")
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

    pipeline.run_mode_from_args(args)
