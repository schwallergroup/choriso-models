import pandas as pd
import numpy as np


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

    test_rxn_smiles = "CCCCCCCCC>>CCCCCC.OOOOOOOO"

    test_df = pd.DataFrame([test_rxn_smiles]*100, columns=["SMILES"])

    test = prepare_data(test_df)

    tr, val, te = random_split(test, 0.1, 0.1)

    print("0")
