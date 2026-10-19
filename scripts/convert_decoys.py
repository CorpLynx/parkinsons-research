#!/usr/bin/env python3
"""
convert_decoys.py - Turn downloaded decoy files into the CSV that `benchmark.py prepare` expects.

Accepts DUD-E (`decoys_final.ism`), DUDE-Z / ZINC SMILES exports, or any .smi/.ism/.txt/.csv
(optionally .gz) whose first column is a SMILES string and whose optional second column is an
identifier. Validates and canonicalises each molecule, drops duplicates, and (with --library)
removes anything that matches a known active so decoys cannot leak actives.

Usage:
  convert_decoys.py decoys_final.ism --output decoys.csv --library data/chembl_lrrk2_gba1_library.csv
"""
import argparse
import csv
import gzip
import io
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
MAX_LINES = 5_000_000
MAX_DECOMPRESSED_BYTES = 2 * 1024**3


def open_text(path):
    if path.endswith(".gz"):
        raw = gzip.open(path, "rb")
        return io.TextIOWrapper(_LimitedReader(raw, MAX_DECOMPRESSED_BYTES), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


class _LimitedReader(io.RawIOBase):
    """Guards against decompression bombs."""

    def __init__(self, raw, limit):
        self.raw, self.left = raw, limit

    def readable(self):
        return True

    def readinto(self, buf):
        data = self.raw.read(min(len(buf), self.left))
        if self.left <= 0:
            raise ValueError("decompressed size limit exceeded")
        self.left -= len(data)
        buf[:len(data)] = data
        return len(data)


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(rdMolStandardize.LargestFragmentChooser().choose(mol))


def parse_lines(path):
    with open_text(path) as fh:
        for n, line in enumerate(fh):
            if n >= MAX_LINES:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = next(csv.reader([line.replace("\t", ",").replace(" ", ",")]))
            smiles = parts[0]
            if smiles.lower() in ("smiles", "canonical_smiles"):
                continue  # header row
            yield smiles, (parts[1] if len(parts) > 1 else "")


def convert(src, output, library=None):
    known_actives = set()
    if library:
        lib = pd.read_csv(library)
        known_actives = {c for c in (canonical(s) for s in lib["smiles"].dropna()) if c}
    seen, rows, stats = set(), [], {"invalid": 0, "duplicate": 0, "matches_active": 0}
    for smiles, ident in parse_lines(src):
        can = canonical(smiles)
        if can is None:
            stats["invalid"] += 1
        elif can in seen:
            stats["duplicate"] += 1
        elif can in known_actives:
            stats["matches_active"] += 1
        else:
            seen.add(can)
            rows.append({"smiles": can, "id": ident})
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=["smiles", "id"]).to_csv(output, index=False)
    print(f"Wrote {len(rows)} decoys to {output} (skipped: {stats})")
    return len(rows), stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--output", required=True)
    ap.add_argument("--library", help="library CSV; decoys equal to any of its molecules are dropped")
    a = ap.parse_args()
    if not os.path.isfile(a.input):
        sys.exit(f"{a.input} not found")
    convert(a.input, a.output, a.library)
