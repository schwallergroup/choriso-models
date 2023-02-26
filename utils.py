import pandas as pd


def split_reaction(reaction: str):
    """Splits a given reaction SMILES into one reactant- and one product SMILES."""
    reactants, products = reaction.split(">>")

    return reactants, products


def placeholder(data: pd.DataFrame, rsmiles_col="SMILES"):
    """Applies the split_reaction function to the reaction SMILES column of a pd.DataFrame."""
    data = data[rsmiles_col].apply(func=lambda x: pd.Series(split_reaction(x), index=['reactants', 'products']))
    return data


if __name__ == "__main__":

    test_rxn_smiles = "CCCCCCCCC>>CCCCCC.OOOOOOOO"

    test_df = pd.DataFrame([test_rxn_smiles, test_rxn_smiles, test_rxn_smiles], columns=["SMILES"])

    test = placeholder(test_df)

    print(r)