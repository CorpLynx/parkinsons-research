#!/usr/bin/env python3
"""
check_pipeline.py - Runs on Mac mini M2 (chem_m2 environment)

Produces a human-readable triage report from the pipeline's Stage 4 output.

Reads either:
  * the raw ADMET-AI output      -> {target}_validated_candidates.csv
  * the filtered safe-leads CSV  -> {target}_FINAL_SAFE_LEADS.csv

and prints:
  * total candidate count
  * per-target breakdown (if multiple targets in one file)
  * BBB / hERG / DILI pass rates
  * top leads ranked by CNNscore (when a CNNscore column is present)
  * reference-compound presence (MLi-2, Ambroxol, Lestaurtinib)

Usage:
    python scripts/check_pipeline.py 06_final_hits_admet/LRRK2_FINAL_SAFE_LEADS.csv
    python scripts/check_pipeline.py 06_final_hits_admet/LRRK2_validated_candidates.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Columns ADMET-AI is expected to emit.
ADMET_COLS = {"BBB_Martins", "hERG", "DILI"}

# Reference compounds the library is built around (chembl_id -> name).
REFERENCES = {
    "CHEMBL4098877": "MLi-2 (LRRK2)",
    "CHEMBL153479": "Ambroxol (GBA1)",
    "CHEMBL603469": "Lestaurtinib (LRRK2)",
    "CHEMBL206468": "Isofagomine/Afegostat (GBA1)",
}


def load_csv(path):
    df = pd.read_csv(path)
    if df.empty:
        print(f"[check] {path}: empty file, nothing to report.")
        return df
    return df


def report(df, label):
    print(f"\n=== Pipeline triage report: {label} ===")
    print(f"Total rows: {len(df)}")

    # Identify target column if present.
    target_col = None
    for c in ("target", "Target"):
        if c in df.columns:
            target_col = c
            break
    if target_col:
        print("\n-- Per target --")
        print(df[target_col].value_counts().to_string())

    # ADMET pass rates.
    admet_present = ADMET_COLS <= set(df.columns)
    if admet_present:
        print("\n-- Neuro-ADMET filters (BBB >= 0.80, hERG <= 0.30, DILI <= 0.40) --")
        for col in ("BBB_Martins", "hERG", "DILI"):
            pass_mask = (
                (df[col] >= 0.80) if col == "BBB_Martins" else (df[col] <= 0.30 if col == "hERG" else df[col] <= 0.40)
            )
            print(f"  {col}: {pass_mask.sum()}/{len(df)} pass "
                  f"({100.0 * pass_mask.sum() / len(df):.1f}%)")
        safe = df[
            (df["BBB_Martins"] >= 0.80)
            & (df["hERG"] <= 0.30)
            & (df["DILI"] <= 0.40)
        ]
        print(f"\n  Safe leads (all three pass): {len(safe)}")
    else:
        print("\n-- Neuro-ADMET columns not found in this file (raw docking hits?) --")
        safe = df

    # Top leads by CNNscore if available.
    if "CNNscore" in safe.columns and len(safe) > 0:
        top = safe.sort_values("CNNscore", ascending=False).head(10)
        print("\n-- Top leads by CNNscore --")
        cols = [c for c in ("ligand_id", "target", "CNNscore", "CNNaffinity", "smiles")
                if c in top.columns]
        print(top[cols].to_string(index=False))
    elif len(safe) > 0 and "ligand_id" in safe.columns:
        print("\n-- Leads --")
        print(safe[["ligand_id"] + [c for c in ("target",) if c in safe.columns]]
              .head(10).to_string(index=False))

    # Reference-compound check.
    if "chembl_id" in safe.columns:
        print("\n-- Reference compounds --")
        for chembl_id, name in REFERENCES.items():
            hits = safe[safe["chembl_id"].astype(str) == chembl_id]
            if len(hits) and "CNNscore" in hits.columns:
                state = f"present (CNNscore={hits.iloc[0]['CNNscore']:.2f})"
            elif len(hits):
                state = "present"
            else:
                state = "absent"
            print(f"  {name} ({chembl_id}): {state}")
    elif "ligand_id" in safe.columns:
        print("\n-- Reference compounds (by ligand_id) --")
        for chembl_id, name in REFERENCES.items():
            lid = f"LRRK2_{chembl_id}" if "LRRK2" in name else f"GBA1_{chembl_id}"
            hits = safe[safe["ligand_id"].astype(str).str.contains(chembl_id, na=False)]
            state = "present" if len(hits) else "absent"
            print(f"  {name} ({lid}): {state}")

    return safe


def main():
    parser = argparse.ArgumentParser(
        description="Triage report for the PD pipeline Stage 4 output."
    )
    parser.add_argument(
        "csv",
        help="Path to *_validated_candidates.csv or *_FINAL_SAFE_LEADS.csv.",
    )
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.is_file():
        sys.exit(f"[check] CSV not found: {path}")
    df = load_csv(path)
    report(df, path.name)


if __name__ == "__main__":
    main()
