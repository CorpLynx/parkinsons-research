#!/usr/bin/env python3
"""
cns_filter.py - Runs on Mac mini M2 (chem_m2 environment)
Filters SMILES strings for Blood-Brain Barrier permeability and embeds 3D conformers.

Usage: cns_filter.py INPUT_CSV OUTPUT_SDF OUTPUT_CSV
INPUT_CSV needs `ligand_id` and `smiles` columns.
"""
import os
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from pd_common import sanitize_id

MAX_INPUT_ROWS = 1_000_000
MAX_HEAVY_ATOMS = 100

# Generic smoke-test molecules. They are NOT validated reference compounds and
# must not be interpreted as named drugs; replace with a curated library
# (e.g. ChEMBL / PubChem exports) for real campaigns.
STARTER_CNS_LIBRARY = [
    {"ligand_id": "starter_01", "smiles": "CC1=CC(=NN1C)NC2=NC=C(C(=N2)NC3=CC=C(C=C3)N4CCOCC4)C(F)(F)F"},
    {"ligand_id": "starter_02", "smiles": "CC(C)OC1=CC(=NC(=N1)N2CCC(CC2)O)N3C=C(C4=CC=CC=C43)C#N"},
    {"ligand_id": "starter_03", "smiles": "CC1=CC(=C(C=C1NC2=NC=C(C(=N2)NC3=CC=CC=C3OC)C(F)(F)F)F)N4CCOCC4"},
    {"ligand_id": "starter_04", "smiles": "C1CC(CCC1NCC2=CC(=C(C(=C2)Br)N)Br)O"},
    {"ligand_id": "starter_05", "smiles": "CC1=CC=C(C=C1)S(=O)(=O)N2CCN(CC2)C(=O)C3=CC=CC=C3F"},
]


def passes_bbb_rules(mol):
    if mol is None:
        return False
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

    return (
        mw <= 450.0 and
        1.0 <= logp <= 4.5 and
        tpsa < 90.0 and
        hbd <= 3 and
        hba <= 7 and
        rot_bonds <= 8
    )


def load_library(input_csv):
    if not os.path.exists(input_csv):
        print(f"[Mac mini M2] {input_csv} not found. Generating starter CNS benchmark CSV...")
        os.makedirs(os.path.dirname(input_csv) or ".", exist_ok=True)
        pd.DataFrame(STARTER_CNS_LIBRARY).to_csv(input_csv, index=False)
    df = pd.read_csv(input_csv, nrows=MAX_INPUT_ROWS)
    missing = {"ligand_id", "smiles"} - set(df.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing required column(s): {sorted(missing)}")
    df = df.dropna(subset=["ligand_id", "smiles"]).copy()
    df["ligand_id"] = df["ligand_id"].map(sanitize_id)
    # Unique ids: they become directory/file names downstream.
    df = df.drop_duplicates(subset="ligand_id", keep="first")
    return df


def process_library(input_csv, output_sdf, output_csv):
    df = load_library(input_csv)
    os.makedirs(os.path.dirname(output_sdf) or ".", exist_ok=True)
    writer = Chem.SDWriter(output_sdf)
    passed_records = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if passes_bbb_rules(mol):
            mol_3d = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
            if res == 0:
                AllChem.MMFFOptimizeMolecule(mol_3d)
                mol_3d.SetProp("_Name", str(row["ligand_id"]))
                mol_3d.SetProp("SMILES", str(row["smiles"]))
                writer.write(mol_3d)
                passed_records.append({"ligand_id": row["ligand_id"], "smiles": row["smiles"]})

    writer.close()
    pd.DataFrame(passed_records, columns=["ligand_id", "smiles"]).to_csv(output_csv, index=False)
    print(f"[Mac mini M2] Filtered {len(passed_records)} CNS-penetrant 3D ligands out of {len(df)}.")
    return len(passed_records)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    process_library(sys.argv[1], sys.argv[2], sys.argv[3])
