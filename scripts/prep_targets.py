#!/usr/bin/env python3
"""
prep_targets.py - Runs on Mac mini M2 (chem_m2 environment)
Downloads LRRK2 (7LHT) and GBA1 (6N3F), strips waters/heterogens, fixes missing atoms,
and protonates at physiological pH (7.4 for cytosolic LRRK2, 5.5 for lysosomal GBA1).
"""
import os
import urllib.request
from pdbfixer import PDBFixer
from openmm.app import PDBFile

SHARED_ROOT = "/mnt/pd_research"

TARGETS = {
    "LRRK2": {
        "pdb_id": "7LHT",
        "ph": 7.4,
        "dir": f"{SHARED_ROOT}/00_targets/LRRK2"
    },
    "GBA1": {
        "pdb_id": "6N3F",
        "ph": 5.5,
        "dir": f"{SHARED_ROOT}/00_targets/GBA1"
    }
}

def prepare_targets():
    for name, cfg in TARGETS.items():
        os.makedirs(cfg["dir"], exist_ok=True)
        raw_pdb = os.path.join(cfg["dir"], f"{cfg['pdb_id']}_raw.pdb")
        clean_pdb = os.path.join(cfg["dir"], f"{cfg['pdb_id']}_clean.pdb")

        print(f"[{name}] Downloading PDB {cfg['pdb_id']} from RCSB...")
        url = f"https://files.rcsb.org/download/{cfg['pdb_id']}.pdb"
        urllib.request.urlretrieve(url, raw_pdb)

        print(f"[{name}] Cleaning heterogens, rebuilding missing atoms, and protonating at pH {cfg['ph']}...")
        fixer = PDBFixer(filename=raw_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(cfg["ph"])

        with open(clean_pdb, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        print(f"[{name}] Ready for docking at: {clean_pdb}\n")

if __name__ == "__main__":
    prepare_targets()
