import argparse
import os
from benchmark_models import ReactionModel, BenchmarkPipeline
from utils import prepare_parser, set_pythonpath


class Neuralsym(ReactionModel):

    def __init__(self):
        self.name = "neuralsym"
        super().__init__()

        self.template_col = "template_r0"

    def preprocess(self, dataset="cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        dataset = f"{dataset}_{self.template_col}"
        os.system(f"python prepare_data.py --output_file_prefix {dataset} --dataset {dataset}")  # TODO add args

    def train(self, dataset="cjhif"):
        """Train the reaction model. Should also contain validation and test steps"""

        model = "Highway"
        seed = 77777777
        dataset = f"{dataset}_{self.template_col}"

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
              f"--dataset {dataset} " \
              f"--do_train"

        os.system(cmd)

    def predict(self, dataset="cjhif"):
        """Predict provided data with the reaction model"""

        model = "Highway"
        seed = 77777777
        dataset = f"{dataset}_{self.template_col}"

        cmd = f"python infer_all.py " \
              f"--csv_prefix {dataset}_to_32681_csv " \
              f"--labels_prefix {dataset}_to_32681_labels " \
              f"--templates_file training_templates.txt " \
              f"--rxn_smi_prefix {dataset}_clean_rxnsmi_noreagent_allmapped_canon " \
              f"--log_file 'infer_{seed}_{model}_depth0_dim300' " \
              f"--reacfps_prefix {dataset}_to_32681_reac_fps " \
              f"--hidden_size 300 " \
              f"--depth 0 " \
              f"--topk 20 " \
              f"--maxk 20 " \
              f"--model Highway " \
              f"--expt_name '{model}_{seed}_depth0_dim300_lr1e3_stop2_fac30_pat1' " \
              f"--seed {seed} " \
              f"--dataset {dataset}"

        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONMT parser')

    prepare_parser(parser)

    args = parser.parse_args()

    set_pythonpath(path=os.getcwd())

    os.chdir("neuralsym")

    reaction_model = Neuralsym()
    pipeline = BenchmarkPipeline(model=reaction_model)

    pipeline.run_mode_from_args(args)
