#!/usr/bin/env python3
"""
prep_targets.py - Runs on Mac mini M2 (chem_m2 environment)
Downloads LRRK2 and GBA1 structures, strips waters/heterogens, fixes missing atoms,
and protonates at physiological pH (7.4 for cytosolic LRRK2, 5.5 for lysosomal GBA1).
"""
import os
import urllib.request

from pd_common import SHARED_ROOT, TARGETS, validate_pdb_id

MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_MODELED_GAP = 10  # residues; longer missing loops are left unmodeled


def download_pdb(pdb_id, dest):
    pdb_id = validate_pdb_id(pdb_id)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    with urllib.request.urlopen(url, timeout=60) as resp:  # nosec B310 - fixed https host
        data = resp.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(f"{pdb_id}: download exceeds size limit")
    if not data.lstrip().startswith((b"HEADER", b"REMARK", b"TITLE", b"CRYST1", b"ATOM")):
        raise RuntimeError(f"{pdb_id}: response does not look like a PDB file")
    with open(dest, "wb") as fh:
        fh.write(data)


def drop_long_gaps(fixer, max_gap=MAX_MODELED_GAP):
    """Keep only short internal gaps; long loops/termini would be unreliable guesses."""
    fixer.missingResidues = {
        key: residues for key, residues in fixer.missingResidues.items()
        if len(residues) <= max_gap
    }


def prepare_targets():
    # Imported lazily so the module (and its helpers) can be imported without OpenMM.
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    for name, cfg in TARGETS.items():
        target_dir = os.path.join(SHARED_ROOT, "00_targets", name)
        os.makedirs(target_dir, exist_ok=True)
        raw_pdb = os.path.join(target_dir, f"{cfg['pdb_id']}_raw.pdb")
        clean_pdb = os.path.join(target_dir, f"{cfg['pdb_id']}_clean.pdb")

        print(f"[{name}] Downloading PDB {cfg['pdb_id']} from RCSB...")
        download_pdb(cfg["pdb_id"], raw_pdb)

        print(f"[{name}] Cleaning heterogens, rebuilding missing atoms, and protonating at pH {cfg['ph']}...")
        fixer = PDBFixer(filename=raw_pdb)
        fixer.findMissingResidues()
        drop_long_gaps(fixer)
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
