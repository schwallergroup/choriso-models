import os
import json
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
import sys
from typing import List

from Graph2SMILES.utils.data_utils import tokenize_smiles


def prepare_parser(parser):
    parser.add_argument('--phase', '-phase', '--p', '-p', type=str, default='train', choices=['t', 'p', 'tp'],
                        help='Mode to run the model in. Either train(t) or predict(p)')
    parser.add_argument('--dataset', '-dataset', '--ds', '-ds', '--d', '-d', type=str, default='cjhif',
                        help='Dataset to use.')


def overwrite_config_with_tokenizer(config, tokenizer):
    config.decoder_start_token_id = tokenizer.cls_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.vocab_size = tokenizer.vocab_size
    return config


def transfer_data(origin_folder, destination):
    files = ["train.tsv", "val.tsv", "test.tsv"]
    files_to_copy = [os.path.join(origin_folder, file) for file in files if not os.path.exists(os.path.join(destination, file))]
    if len(files_to_copy) > 0:
        files_to_copy = " ".join(files_to_copy)
        os.system(f"cp {files_to_copy} {destination}")


def set_pythonpath(path):
    # Get the current value of PYTHONPATH (if it exists)
    pythonpath = os.getenv('PYTHONPATH', '')

    # Add ~/reaction_forward to PYTHONPATH
    pythonpath += ':' + path

    # Set the updated PYTHONPATH
    os.environ['PYTHONPATH'] = pythonpath
    sys.path.append(pythonpath)


def remove_spaces(spaced_str: str):
    return spaced_str.replace(" ", "")


def is_correct_pred(pred: str, target: str):
    if pred == target:
        return 1
    else:
        return 0


def top_k_accuracy(preds: List[List[str]], targets: List[str], k: int=1):
    assert k <= len(preds[0]), "k has to be smaller or equal than the amount of predictions per reaction"
    assert k > 0, "k has to be greater than 0"

    # transform to np arrays for easier handling. take the first k predictions
    preds = np.array(preds)[:, :k]
    preds = np.vectorize(lambda x: x.replace(" ", ""))(preds)
    preds = np.vectorize(canonicalize_smiles)(preds)

    targets = np.array(targets)
    targets = np.vectorize(lambda x: x.replace(" ", ""))(targets)
    targets = np.vectorize(canonicalize_smiles)(targets)

    top_k_accs = []
    # False valued rows will not be calculated. initialize with all True
    rows_to_calculate = np.ones(targets.shape, dtype=bool)
    for i in range(k):

        # select pred column
        kth_preds = preds[:, i]

        # get indices that have to be calculated. Needs to be done for overwriting with new values
        indices = np.argwhere(rows_to_calculate != False).squeeze()

        # apply mask to only calculate not yet correctly predicted values
        kth_preds = kth_preds[rows_to_calculate]
        temp_targets = targets[rows_to_calculate]

        # check if prediction was incorrect
        incorrect_pred = kth_preds != temp_targets

        # change mask to not calculate rows where there already was a correct prediction
        rows_to_calculate[indices] = incorrect_pred

        num_correct = len(rows_to_calculate) - np.sum(rows_to_calculate.astype(int), axis=0)

        top_k_acc = num_correct / len(rows_to_calculate)
        print(f"top-{i+1} accuracy: ", top_k_acc)
        top_k_accs.append(top_k_acc)

    return top_k_accs


def canonicalize_smiles(smiles, verbose=False):
    # will raise an Exception if invalid SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        if verbose:
            print(f'{smiles} is invalid.')
        return ''


def split_reaction(reaction: str):
    """Splits a given reaction SMILES into one reactant- and one product SMILES."""
    reactants, reagents, products = reaction.split(">")

    if reagents != "":
        reactants = reactants + "." + reagents

    return reactants, products


def prepare_data(data: pd.DataFrame, rsmiles_col="SMILES"):
    """Applies the split_reaction function to the reaction SMILES column of a pd.DataFrame."""
    data = data[rsmiles_col].apply(func=lambda x: pd.Series(split_reaction(x), index=['reactants', 'products']))
    return data


def random_split(data: pd.DataFrame, val_percent, test_percent):
    """Prepare a random data split, based on the given data and percentages."""
    val_size = int(val_percent * len(data))
    test_size = int(test_percent * len(data))

    np.random.shuffle(data.values)

    test_data = data[:test_size]
    val_data = data[test_size:test_size + val_size]
    train_data = data[test_size + val_size:]

    return train_data, val_data, test_data


class ReactionForwardDataset(torch.utils.data.Dataset):
    """Class wrapper for reaction forward prediction data. Adapted from
    https://huggingface.co/transformers/v3.2.0/custom_datasets.html"""

    def __init__(self, tokenizer_outs):
        self.tokenizer_outs = tokenizer_outs

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenizer_outs.items()}
        return item

    def __len__(self):
        return len(self.tokenizer_outs["labels"])


def csv_to_txt(data_dir: str):
    """Converts a train, val and test files to src and tgt txt files."""
    file_names = ["test", "val", "train"]

    for file_name in file_names:
        src_file = os.path.join(data_dir, f"src-{file_name}.txt")
        tgt_file = os.path.join(data_dir, f"tgt-{file_name}.txt")

        # skip if files already exist
        if os.path.exists(src_file) and os.path.exists(tgt_file):
            print(f"Files {src_file} and {tgt_file} already exist. Skipping...")
            continue

        reactions = pd.read_csv(os.path.join(data_dir, f"{file_name}.tsv"), sep="\t", error_bad_lines=False)

        split_reactions = prepare_data(reactions, rsmiles_col="canonic_rxn")

        reactant_data = split_reactions["reactants"].apply(lambda smi: tokenize_smiles(smi))
        product_data = split_reactions["products"].apply(lambda smi: tokenize_smiles(smi))

        reactant_data.to_csv(src_file, sep="\t", index=False, header=False)
        product_data.to_csv(tgt_file, sep="\t", index=False, header=False)


def csv_to_jsonl(data_dir: str, target_dir: str):
    """Converts a train, val and test files to jsonl files with the following format:
    train.jsonl: [{"src": "reac1", "trg": "prod1"}, {"src": "reac2", "tgt": "prod2"}]
    Required for DiffuSeq"""

    file_names = ["test", "valid", "train"]

    for file_name in file_names:
        final_file = os.path.join(target_dir, f"{file_name}.jsonl")
        if os.path.exists(final_file):
            print(f"File {final_file} already exists. Skipping.")
            continue
        # tsv is named val.tsv, diffuseq requires valid.jsonl
        tsv_name = file_name if file_name != "valid" else "val"
        reactions = pd.read_csv(os.path.join(data_dir, f"{tsv_name}.tsv"), sep="\t", error_bad_lines=False)

        split_reactions = prepare_data(reactions, rsmiles_col="canonic_rxn")

        reactant_data = split_reactions["reactants"].apply(lambda smi: tokenize_smiles(smi))
        product_data = split_reactions["products"].apply(lambda smi: tokenize_smiles(smi))

        # TODO test if this is necessary
        reactant_data = reactant_data.apply(lambda x: remove_spaces(x))
        product_data = product_data.apply(lambda x: remove_spaces(x))

        data = [{"src": reactant, "trg": product} for reactant, product in zip(reactant_data.values, product_data.values)]

        with open(final_file, "w") as f:
            for src_trg_dict in data:
                json.dump(src_trg_dict, f)
                f.write('\n')


if __name__ == "__main__":

    preds = pd.read_csv('OpenNMT_Transformer/runs/models/cjhif_model_step_200000_test_predictions.txt', sep="\t",
                        header=None).values
    preds = [pred[0] for pred in preds]

    targets = pd.read_csv('data/cjhif/tgt-test.txt', sep="\t", header=None).values
    targets = [target[0] for target in targets]

    print(targets)
    num_pred = int(len(preds)/len(targets))
    preds = [preds[i:i+num_pred] for i in range(0, len(preds), num_pred)]

    acc = top_k_accuracy(preds, targets, k=5)

    print(acc)

    """
    data_dir = "data/cjhif/"
    csv_to_txt(data_dir)
    """
