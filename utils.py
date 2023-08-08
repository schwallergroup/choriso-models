import os
import json
import pandas as pd
from rdkit import Chem
import sys

from Graph2SMILES.utils.data_utils import tokenize_smiles


def prepare_parser(parser):
    parser.add_argument('--phase', '-phase', '--p', '-p', type=str, default='train', choices=['t', 'p', 'tp'],
                        help='Mode to run the model in. Either train(t) or predict(p)')
    parser.add_argument('--dataset', '-dataset', '--ds', '-ds', '--d', '-d', type=str, default='choriso',
                        help='Dataset to use.')


def overwrite_config_with_tokenizer(config, tokenizer):
    config.decoder_start_token_id = tokenizer.cls_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.vocab_size = tokenizer.vocab_size
    return config


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

        reactions = pd.read_csv(os.path.join(data_dir, f"{file_name}.tsv"), sep="\t")

        split_reactions = prepare_data(reactions, rsmiles_col="canonic_rxn")

        reactant_data = split_reactions["reactants"].apply(lambda smi: tokenize_smiles(smi))
        product_data = split_reactions["products"].apply(lambda smi: tokenize_smiles(smi))

        reactant_data.to_csv(src_file, sep="\t", index=False, header=False)
        product_data.to_csv(tgt_file, sep="\t", index=False, header=False)


def standardize_output(reactants, targets, predictions, csv_out_path):

    # Create a list of column names for the predictions
    pred_cols = [f"pred_{i}" for i in range(len(predictions[0]))]

    # Create a list of dictionaries representing each row of the DataFrame
    rows = []
    for reac, prod, preds in zip(reactants, targets, predictions):
        row = {"canonical_rxn": reac + ">>" + prod, "target": prod}
        row.update({pred_col: pred for pred_col, pred in zip(pred_cols, preds)})
        rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_out_path, index=False)

