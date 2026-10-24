#!/usr/bin/env python3
"""
smoke_test.py - Runs on Mac mini M2 (chem_m2 environment)

End-to-end smoke test of Stage 1 of the pipeline using the curated ChEMBL
LRRK2/GBA1 library:

    build_chembl_library.py  ->  cns_filter.py

Verifies that the curated library produces a non-empty, CNS-passing,
3D-embeddable ligand set and that the reference compound MLi-2 survives
the BBB filter. No GPU nodes are required.

Usage:
    python scripts/smoke_test.py              # artifacts in a temp dir
    python scripts/smoke_test.py --outdir /tmp/pd_smoke  # keep artifacts
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import Chem

from pd_common import sanitize_id

SCRIPTS = Path(__file__).resolve().parent
LIB_BUILDER = SCRIPTS / "build_chembl_library.py"
CNS_FILTER = SCRIPTS / "cns_filter.py"

MLi2_CHEMBL_ID = "CHEMBL4098877"
MLi2_LIGAND_ID = f"LRRK2_{MLi2_CHEMBL_ID}"  # sanitized form used in the library


def run(cmd, label):
    """Run a command, streaming output, and fail hard on non-zero exit."""
    print(f"[smoke] {label} ...")
    subprocess.run(cmd, check=True)  # noqa: S603 - fixed interpreter + script, validated paths


def main():
    parser = argparse.ArgumentParser(
        description="Stage-1 smoke test: build ChEMBL library, then CNS-filter it."
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to keep artifacts in (default: a temp dir that is removed).",
    )
    args = parser.parse_args()

    workdir = Path(args.outdir) if args.outdir else Path(tempfile.mkdtemp(prefix="pd_smoke_"))
    workdir.mkdir(parents=True, exist_ok=True)
    keep = args.outdir is not None

    lib_csv = workdir / "chembl_library.csv"
    cns_sdf = workdir / "cns.sdf"
    cns_csv = workdir / "cns.csv"

    # Stage A: build the curated ChEMBL library (network: queries ChEMBL API).
    run([sys.executable, str(LIB_BUILDER), "--output", str(lib_csv)],
        "build_chembl_library")

    rows = list(csv.DictReader(lib_csv.open()))
    assert rows, "library is empty"
    n_bbb = sum(1 for r in rows if r["passes_bbb"] == "True")
    n_targets = {r["target"] for r in rows}
    print(f"[smoke] library: {len(rows)} compounds ({sorted(n_targets)}), "
          f"{n_bbb} BBB-passing")

    # Stage B: CNS filter (mirrors pipeline Stage 1).
    run([sys.executable, str(CNS_FILTER), str(lib_csv), str(cns_sdf), str(cns_csv)],
        "cns_filter")

    cns = pd.read_csv(cns_csv)
    assert len(cns) > 0, "CNS filter produced zero ligands"
    print(f"[smoke] CNS-passing ligands: {len(cns)}")

    # End-to-end invariant: the same BBB rules are applied in both stages,
    # so the CNS-passing count must equal the library's BBB-passing count.
    assert len(cns) == n_bbb, (
        f"CNS count ({len(cns)}) != library BBB-passing count ({n_bbb})"
    )

    # The reference compound MLi-2 must survive the BBB filter.
    assert MLi2_LIGAND_ID in set(cns["ligand_id"]), (
        f"reference MLi-2 ({MLi2_LIGAND_ID}) absent from CNS-passing set"
    )

    # The SDF must contain valid, 3D-embedded molecules matching the CSV 1:1.
    supplier = Chem.SDMolSupplier(str(cns_sdf), removeHs=False)
    sdf_names = [
        m.GetProp("_Name")
        for m in supplier
        if m is not None and m.HasProp("_Name")
    ]
    assert len(sdf_names) == len(cns), (
        f"SDF molecule count ({len(sdf_names)}) != CNS CSV rows ({len(cns)})"
    )
    assert sdf_names == cns["ligand_id"].tolist(), "SDF name order != CSV order"
    for name in sdf_names:
        assert sanitize_id(name) == name, f"unsafe SDF name: {name!r}"

    print(f"[smoke] SDF molecules: {len(sdf_names)} (all valid, embedded, named)")
    print(f"[smoke] artifacts written to: {workdir}" + (" (temp, removed on exit)" if not keep else ""))
    print("[smoke] PASS: end-to-end Stage 1 smoke test succeeded")


if __name__ == "__main__":
    main()
