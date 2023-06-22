import pandas as pd
import os
import json
import torch
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from utils import prepare_data


def get_template_idx(templates: dict, template: str):
    return templates[template] if template in templates.keys() else len(templates)


def reactants_to_fp(reactant_smiles: str, radius: int = 2, fp_size: int = 1024, dtype: str = "int32") -> torch.Tensor:
    fp_gen = GetMorganGenerator(
        radius=radius, useCountSimulation=True, includeChirality=True, fpSize=fp_size
    )
    mol = Chem.MolFromSmiles(reactant_smiles)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    count_fp = torch.from_numpy(count_fp).squeeze()
    return count_fp


def process(data_dir: str):
    # load raw data from file

    templates_with_counts = {}

    train_df = pd.read_csv(os.path.join(data_dir, f"train.tsv"), sep="\t")
    val_df = pd.read_csv(os.path.join(data_dir, f"val.tsv"), sep="\t")
    test_df = pd.read_csv(os.path.join(data_dir, f"test.tsv"), sep="\t")

    templates_list = train_df["template"].tolist() + val_df["template"].tolist()

    # get unique templates only from train and val set, and count them
    for template in templates_list:
        if template not in templates_with_counts:
            templates_with_counts[template] = 0
        templates_with_counts[template] += 1

    # remove templates that appear less than 100 times, like in the Segler paper
    templates = [k for k, v in templates_with_counts.items() if v >= 100]
    templates = {k: torch.tensor(v) for v, k in enumerate(templates)}

    # save templates to file
    with open(os.path.join(data_dir, "templates.json"), "w") as f:
        json.dump(templates, f)

    dfs = {"train": train_df, "val": val_df, "test": test_df}
    for df_name in dfs:
        df = dfs[df_name]

        template_file = os.path.join(data_dir, f"{df_name}_template_idx.pt")
        fp_file = os.path.join(data_dir, f"{df_name}_fps.pt")

        # only do if files don't exist yet.
        if not os.path.exists(template_file) or not os.path.exists(fp_file):
            # assign template index to datasets.
            df["template_idx"] = df["template"].apply(lambda x: get_template_idx(templates, x))

            # get template_idx and save them to file
            template_idx = df["template_idx"].tolist()
            torch.save(template_idx, template_file)

            # get reactants
            df["reactants"] = df["canonic_rxn"].apply(lambda x: x.split(">>")[0])

            # get reactant fingerprints and save them to file
            reactants = df["reactants"].tolist()
            reactant_fps = [reactants_to_fp(reactants[i]) for i in range(len(df))]
            torch.save(reactant_fps, fp_file)
