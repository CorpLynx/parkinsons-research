#!/usr/bin/env python3
"""
cns_filter.py - Runs on Mac mini M2 (chem_m2 environment)
Filters SMILES strings for Blood-Brain Barrier permeability and embeds 3D conformers.
"""
import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

STARTER_CNS_LIBRARY = [
    {"ligand_id": "DNL201_LRRK2_Inhibitor", "smiles": "CC1=CC(=NN1C)NC2=NC=C(C(=N2)NC3=CC=C(C=C3)N4CCOCC4)C(F)(F)F"},
    {"ligand_id": "MLi2_LRRK2_Inhibitor", "smiles": "CC(C)OC1=CC(=NC(=N1)N2CCC(CC2)O)N3C=C(C4=CC=CC=C43)C#N"},
    {"ligand_id": "GNE7915_LRRK2_Inhibitor", "smiles": "CC1=CC(=C(C=C1NC2=NC=C(C(=N2)NC3=CC=CC=C3OC)C(F)(F)F)F)N4CCOCC4"},
    {"ligand_id": "Ambroxol_GBA1_Chaperone", "smiles": "C1CC(CCC1NCC2=CC(=C(C(=C2)Br)N)Br)O"},
    {"ligand_id": "LTI291_GBA1_Activator", "smiles": "CC1=CC=C(C=C1)S(=O)(=O)N2CCN(CC2)C(=O)C3=CC=CC=C3F"}
]

def passes_bbb_rules(mol):
    if mol is None:
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

def process_library(input_csv, output_sdf, output_csv):
    if not os.path.exists(input_csv):
        print(f"[Mac mini M2] {input_csv} not found. Generating starter CNS benchmark CSV...")
        os.makedirs(os.path.dirname(input_csv), exist_ok=True)
        pd.DataFrame(STARTER_CNS_LIBRARY).to_csv(input_csv, index=False)

    df = pd.read_csv(input_csv)
    os.makedirs(os.path.dirname(output_sdf), exist_ok=True)
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
    pd.DataFrame(passed_records).to_csv(output_csv, index=False)
    print(f"[Mac mini M2] Filtered {len(passed_records)} CNS-penetrant 3D ligands out of {len(df)}.")

if __name__ == "__main__":
    process_library(sys.argv[1], sys.argv[2], sys.argv[3])
