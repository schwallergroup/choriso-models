import os
import pandas as pd
import numpy as np

from Graph2SMILES.utils.data_utils import tokenize_smiles


def split_reaction(reaction: str):
    """Splits a given reaction SMILES into one reactant- and one product SMILES."""
    reactants, products = reaction.split(">>")

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


if __name__ == "__main__":

    file_names = ["test", "val", "train"]

    data_dir = "data/cjhif/"

    for file_name in file_names:

        reactions = pd.read_csv(os.path.join(data_dir, f"{file_name}.tsv"), sep="\t", error_bad_lines=False)

        split_reactions = prepare_data(reactions, rsmiles_col="canonic_rxn")

        reactant_data = split_reactions["reactants"].apply(lambda smi: tokenize_smiles(smi))
        product_data = split_reactions["products"].apply(lambda smi: tokenize_smiles(smi))

        reactant_data.to_csv(os.path.join(data_dir, f"src-{file_name}.txt"), sep="\t", index=False, header=False)
        product_data.to_csv(os.path.join(data_dir, f"tgt-{file_name}.txt"), sep="\t", index=False, header=False)
