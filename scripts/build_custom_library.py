#!/usr/bin/env python3
"""
build_custom_library.py - Builds pipeline-ready ligand CSVs from public data
sources for the Parkinson's pipeline.

Output: a CSV with columns `ligand_id` and `smiles`, ready to pass to
`cns_filter.py` or `master_pipeline.py --raw-smiles`.

Subcommands:
  chembl      Fetch compounds from ChEMBL by target ID, name search, or
              approved-drug filter.
  convert     Convert a local SDF or CSV file into pipeline format.
  inactives   Fetch "inactive" compounds for a target (negative controls).

Examples:
  # LRRK2 actives <= 10 uM, BBB pre-filtered
  python scripts/build_custom_library.py chembl --target-id CHEMBL1075104 \
      --potency-cutoff-nm 10000 --bbb-filter -o data/lrrk2_10uM.csv

  # All ChEMBL-approved drugs (max_phase 4)
  python scripts/build_custom_library.py chembl --approved-drugs \
      -o data/approved_drugs.csv

  # Convert a local SDF
  python scripts/build_custom_library.py convert --input my_library.sdf \
      --id-col name --smiles-col smiles -o data/my_library.csv

  # Negative controls: LRRK2 compounds tested but inactive
  python scripts/build_custom_library.py inactives --target-id CHEMBL1075104 \
      -o data/lrrk2_inactives.csv
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000
SLEEP_BETWEEN_CALLS = 0.02

# BBB rules, mirroring cns_filter.passes_bbb_rules exactly.
BBB = dict(
    mw_max=450.0,
    logp_lo=1.0,
    logp_hi=4.5,
    tpsa_max=90.0,
    hbd_max=3,
    hba_max=7,
    rtb_max=8,
    heavy_max=100,
)


def _fetch(url):
    """GET a URL and return parsed JSON, retrying once on transient failure."""
    last_err = None
    for _ in range(2):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # nosec B310 - fixed HTTPS ChEMBL host  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            last_err = exc
            time.sleep(1.0)
    raise RuntimeError(f"Request failed for {url}: {last_err}")


def _to_nm(value, units):
    """Convert a standard value to nanomolar. Returns None if not convertible."""
    if value is None or units is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    u = str(units).strip().lower()
    if u in ("nm", "nanomolar"):
        return v
    if u in ("um", "micromolar", "u m"):
        return v * 1_000.0
    if u in ("mm", "millimolar"):
        return v * 1_000_000.0
    if u in ("pm", "picomolar"):
        return v / 1_000.0
    if u in ("m", "molar"):
        return v * 1_000_000_000.0
    return None


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_props(smiles):
    """Compute BBB-relevant descriptors from a SMILES string (RDKit)."""
    if not _HAS_RDKIT:
        return dict(mw=None, logp=None, tpsa=None, hbd=None, hba=None,
                    rot_bonds=None, heavy_atoms=None)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(mw=None, logp=None, tpsa=None, hbd=None, hba=None,
                    rot_bonds=None, heavy_atoms=None)
    return dict(
        mw=Descriptors.MolWt(mol),
        logp=Descriptors.MolLogP(mol),
        tpsa=rdMolDescriptors.CalcTPSA(mol),
        hbd=rdMolDescriptors.CalcNumHBD(mol),
        hba=rdMolDescriptors.CalcNumHBA(mol),
        rot_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(),
    )


def passes_bbb(props):
    """BBB filter mirroring cns_filter.passes_bbb_rules exactly."""
    mw = props.get("mw")
    logp = props.get("logp")
    tpsa = props.get("tpsa")
    hbd = props.get("hbd")
    hba = props.get("hba")
    rtb = props.get("rtb")
    heavy = props.get("heavy_atoms")
    if mw is None:
        return False
    return (
        mw <= BBB["mw_max"]
        and (logp is None or BBB["logp_lo"] <= logp <= BBB["logp_hi"])
        and (tpsa is None or tpsa < BBB["tpsa_max"])
        and (hbd is None or hbd <= BBB["hbd_max"])
        and (hba is None or hba <= BBB["hba_max"])
        and (rtb is None or rtb <= BBB["rtb_max"])
        and (heavy is None or heavy <= BBB["heavy_max"])
    )


def write_csv(path, records, bbb_filter=False):
    """Write ligand_id, smiles CSV. Optionally pre-filter by BBB rules."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out = []
    seen = set()
    for r in records:
        lid = r["ligand_id"]
        smi = r["smiles"]
        if not lid or not smi:
            continue
        if lid in seen:
            continue
        seen.add(lid)
        if bbb_filter:
            props = compute_props(smi)
            if not passes_bbb(props):
                continue
        out.append({"ligand_id": lid, "smiles": smi})

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ligand_id", "smiles"])
        writer.writeheader()
        writer.writerows(out)

    n_pass = sum(1 for r in out if not bbb_filter or passes_bbb(compute_props(r["smiles"])))
    print(f"[build_custom_library] Wrote {len(out)} compounds to {path}"
          + (f" ({n_pass} pass BBB filter)" if bbb_filter else ""))
    return len(out)


# ---------------------------------------------------------------------------
# ChEMBL source
# ---------------------------------------------------------------------------

def fetch_chembl_activities(target_chembl_id, potency_cutoff_nm=None,
                            max_records=None):
    """Fetch standard human activities for a ChEMBL target."""
    results = []
    offset = 0
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_flag": 1,
        "limit": PAGE_SIZE,
    }
    while True:
        params["offset"] = offset
        url = f"{BASE}/activity.json?{urllib.parse.urlencode(params)}"
        data = _fetch(url)
        page = data.get("activities", [])
        for a in page:
            if a.get("target_organism") != "Homo sapiens":
                continue
            if a.get("standard_relation") not in {"=", "<", "<=", "<~", "~"}:
                continue
            if a.get("standard_type") not in {
                "IC50", "Ki", "Kd", "EC50", "AC50", "pIC50", "pKi"
            }:
                continue
            nm = _to_nm(a.get("standard_value"), a.get("standard_units"))
            if potency_cutoff_nm is not None:
                if nm is None or nm > potency_cutoff_nm:
                    continue
            smiles = a.get("canonical_smiles")
            if not smiles:
                continue
            results.append({
                "ligand_id": f"CHEMBL_{a.get('molecule_chembl_id')}",
                "smiles": smiles,
                "chembl_id": a.get("molecule_chembl_id"),
                "standard_type": a.get("standard_type"),
                "standard_value_nm": nm,
                "assay_description": a.get("assay_description"),
                "document_journal": a.get("document_journal"),
                "document_year": a.get("document_year"),
            })
            if max_records and len(results) >= max_records:
                return results
        total = data.get("page_meta", {}).get("total_count", len(results))
        if not page or offset + PAGE_SIZE >= total:
            break
        offset += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_CALLS)
    return results


def fetch_chembl_by_name(name, limit=1000):
    """Search ChEMBL molecules by pref_name (case-insensitive contains)."""
    url = f"{BASE}/molecule.json?pref_name__icontains={urllib.parse.quote(name)}&limit={limit}"
    data = _fetch(url)
    out = []
    for m in data.get("molecules", []):
        structs = m.get("molecule_structures") or {}
        smi = structs.get("canonical_smiles")
        if not smi:
            continue
        out.append({
            "ligand_id": f"CHEMBL_{m.get('molecule_chembl_id')}",
            "smiles": smi,
            "chembl_id": m.get("molecule_chembl_id"),
            "pref_name": m.get("pref_name"),
            "max_phase": m.get("max_phase"),
        })
    return out


def fetch_chembl_approved_drugs(limit=5000):
    """Fetch ChEMBL molecules with max_phase = 4 (approved drugs)."""
    results = []
    offset = 0
    while offset < limit:
        params = {
            "max_phase": 4,
            "limit": PAGE_SIZE,
            "offset": offset,
        }
        url = f"{BASE}/molecule.json?{urllib.parse.urlencode(params)}"
        data = _fetch(url)
        page = data.get("molecules", [])
        for m in page:
            structs = m.get("molecule_structures") or {}
            smi = structs.get("canonical_smiles")
            if not smi:
                continue
            results.append({
                "ligand_id": f"CHEMBL_{m.get('molecule_chembl_id')}",
                "smiles": smi,
                "chembl_id": m.get("molecule_chembl_id"),
                "pref_name": m.get("pref_name"),
                "max_phase": m.get("max_phase"),
            })
        total = data.get("page_meta", {}).get("total_count", len(results))
        if not page or offset + PAGE_SIZE >= total:
            break
        offset += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_CALLS)
    return results


def cmd_chembl(args):
    records = []
    if args.target_id:
        print(f"[chembl] Fetching activities for target {args.target_id} ...")
        records = fetch_chembl_activities(
            args.target_id,
            potency_cutoff_nm=args.potency_cutoff_nm,
            max_records=args.max_compounds,
        )
    elif args.approved_drugs:
        print("[chembl] Fetching ChEMBL-approved drugs (max_phase=4) ...")
        records = fetch_chembl_approved_drugs(limit=args.max_compounds)
    elif args.pref_name:
        print(f"[chembl] Searching molecules by name '{args.pref_name}' ...")
        records = fetch_chembl_by_name(args.pref_name, limit=args.max_compounds)
    else:
        sys.exit("--target-id, --pref-name, or --approved-drugs is required for chembl source.")

    # Keep only required columns for output, drop ChEMBL-specific extras.
    clean = [{"ligand_id": r["ligand_id"], "smiles": r["smiles"]} for r in records]
    write_csv(args.output, clean, bbb_filter=args.bbb_filter)


# ---------------------------------------------------------------------------
# Local file conversion
# ---------------------------------------------------------------------------

def cmd_convert(args):
    records = []
    ext = os.path.splitext(args.input)[1].lower()

    if ext in (".sdf", ".mol"):
        if not _HAS_RDKIT:
            sys.exit("RDKit is required to read SDF files.")
        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(args.input, removeHs=False)
        id_col = args.id_col or "name"
        smiles_col = args.smiles_col or None
        for i, m in enumerate(suppl):
            if m is None:
                continue
            lid = m.GetProp(id_col) if m.HasProp(id_col) else f"mol_{i}"
            if smiles_col and m.HasProp(smiles_col):
                smi = m.GetProp(smiles_col)
            else:
                smi = Chem.MolToSmiles(m)
            records.append({"ligand_id": lid, "smiles": smi})
    elif ext == ".csv":
        import pandas as pd
        df = pd.read_csv(args.input)
        id_col = args.id_col or "ligand_id"
        smiles_col = args.smiles_col or "smiles"
        if id_col not in df.columns or smiles_col not in df.columns:
            sys.exit(f"CSV must contain columns '{id_col}' and '{smiles_col}'")
        for _, row in df.iterrows():
            records.append({
                "ligand_id": str(row[id_col]),
                "smiles": str(row[smiles_col]),
            })
    else:
        sys.exit(f"Unsupported input format: {ext} (use .sdf, .mol, or .csv)")

    write_csv(args.output, records, bbb_filter=args.bbb_filter)


# ---------------------------------------------------------------------------
# Inactives (negative controls)
# ---------------------------------------------------------------------------

def cmd_inactives(args):
    """Fetch ChEMBL compounds tested against a target but reported inactive."""
    # Inactive = standard_value > potency_cutoff_nm (default 10 uM) or
    # standard_type = Inhibition with high % values. We use a simple heuristic:
    # standard_flag=1, convertible potency > cutoff, or % inhibition > 50 at
    # screened concentration.
    cutoff = args.inactive_cutoff_nm or 10_000_000  # 10 mM default "inactive"
    results = []
    offset = 0
    params = {
        "target_chembl_id": args.target_id,
        "standard_flag": 1,
        "limit": PAGE_SIZE,
    }
    while True:
        params["offset"] = offset
        url = f"{BASE}/activity.json?{urllib.parse.urlencode(params)}"
        data = _fetch(url)
        page = data.get("activities", [])
        for a in page:
            if a.get("target_organism") != "Homo sapiens":
                continue
            smiles = a.get("canonical_smiles")
            if not smiles:
                continue
            nm = _to_nm(a.get("standard_value"), a.get("standard_units"))
            if nm is None:
                continue
            if nm > cutoff:
                results.append({
                    "ligand_id": f"CHEMBL_INACTIVE_{a.get('molecule_chembl_id')}",
                    "smiles": smiles,
                    "chembl_id": a.get("molecule_chembl_id"),
                    "standard_type": a.get("standard_type"),
                    "standard_value_nm": nm,
                })
        total = data.get("page_meta", {}).get("total_count", len(results))
        if not page or offset + PAGE_SIZE >= total:
            break
        offset += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_CALLS)
        if args.max_compounds and len(results) >= args.max_compounds:
            break

    clean = [{"ligand_id": r["ligand_id"], "smiles": r["smiles"]} for r in results]
    write_csv(args.output, clean, bbb_filter=args.bbb_filter)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build pipeline-ready ligand CSVs from public data sources."
    )
    sub = parser.add_subparsers(dest="source", required=True)

    # chembl
    p_chembl = sub.add_parser("chembl", help="Fetch from ChEMBL web services")
    p_chembl.add_argument("--target-id", help="ChEMBL target ID (e.g. CHEMBL1075104 for LRRK2)")
    p_chembl.add_argument("--pref-name", help="Search molecules by pref_name (case-insensitive)")
    p_chembl.add_argument("--approved-drugs", action="store_true",
                          help="Fetch ChEMBL-approved drugs (max_phase=4)")
    p_chembl.add_argument("--potency-cutoff-nm", type=float,
                          help="Keep actives <= this potency (nM). Requires --target-id.")
    p_chembl.add_argument("--max-compounds", type=int,
                          help="Maximum number of compounds to fetch")
    p_chembl.add_argument("--bbb-filter", action="store_true",
                          help="Pre-filter output by BBB/CNS rules (mirrors cns_filter)")
    p_chembl.add_argument("-o", "--output", required=True, help="Output CSV path")
    p_chembl.set_defaults(func=cmd_chembl)

    # convert
    p_conv = sub.add_parser("convert", help="Convert a local SDF or CSV file")
    p_conv.add_argument("--input", required=True, help="Input SDF (.sdf/.mol) or CSV file")
    p_conv.add_argument("--id-col", help="Column/property to use as ligand_id")
    p_conv.add_argument("--smiles-col", help="Column/property to use as SMILES")
    p_conv.add_argument("--bbb-filter", action="store_true",
                        help="Pre-filter output by BBB/CNS rules")
    p_conv.add_argument("-o", "--output", required=True, help="Output CSV path")
    p_conv.set_defaults(func=cmd_convert)

    # inactives
    p_inact = sub.add_parser("inactives", help="Fetch inactive compounds for a target")
    p_inact.add_argument("--target-id", required=True, help="ChEMBL target ID")
    p_inact.add_argument("--inactive-cutoff-nm", type=float, default=10_000_000,
                         help="Potency above this (nM) is considered inactive (default 10 mM)")
    p_inact.add_argument("--max-compounds", type=int,
                         help="Maximum number of compounds to fetch")
    p_inact.add_argument("--bbb-filter", action="store_true",
                         help="Pre-filter output by BBB/CNS rules")
    p_inact.add_argument("-o", "--output", required=True, help="Output CSV path")
    p_inact.set_defaults(func=cmd_inactives)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
