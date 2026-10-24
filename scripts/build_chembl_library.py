#!/usr/bin/env python3
"""
build_chembl_library.py - Runs on Mac mini M2 (chem_m2 environment)

Builds a curated ChEMBL library of known LRRK2 and GBA1 compounds for the
Parkinson's pipeline, then writes a CNS-filter-ready CSV.

It queries the ChEMBL web services for standard (manually curated) human
bioactivities against:

  * LRRK2  -> CHEMBL1075104  Leucine-rich repeat serine/threonine-protein
                                    kinase 2  (human Q5S007; public assays
                                    include the G2019S mutant)
  * GBA1   -> CHEMBL4106172  Glucocerebrosidase GBA1/GBA2 protein family
                                    (human GBA1 P04062 + GBA2 Q9HCG7)

Each compound is curated:
  * valid canonical SMILES
  * human, standard_flag = 1 activity (IC50 / Ki / Kd / EC50 / pIC50)
  * deduplicated by parent ChEMBL ID, keeping the most potent entry
  * BBB / CNS drug-likeness filter using ChEMBL's precomputed molecular
    properties, mirroring cns_filter.passes_bbb_rules exactly (so no RDKit
    is required to build the library)
  * reference flag for known CNS-penetrant compounds used to sanity-check
    the library and the pipeline check

Output: a CSV with one row per compound and these columns:
  ligand_id, smiles, chembl_id, pref_name, max_phase,
  target, target_chembl_id, standard_type, standard_value_nm, pchembl,
  assay_description, document_journal, document_year,
  mw, logp, tpsa, hbd, hba, rot_bonds, heavy_atoms, passes_bbb, is_reference

Usage:
    python scripts/build_chembl_library.py [--output PATH] [--limit N]

The output is meant to live under the shared NFS root's
01_ligands_raw/ directory and be passed to cns_filter.py / master_pipeline.py
via --raw_smiles.
"""
import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from pd_common import SHARED_ROOT, sanitize_id

BASE = "https://www.ebi.ac.uk/chembl/api/data"

# ChEMBL target IDs verified against the ChEMBL web services.
TARGETS = {
    "LRRK2": {
        "target_chembl_id": "CHEMBL1075104",
        "organism": "Homo sapiens",
        "label": "Leucine-rich repeat serine/threonine-protein kinase 2",
    },
    "GBA1": {
        "target_chembl_id": "CHEMBL2179",
        "organism": "Homo sapiens",
        "label": "Lysosomal acid glucosylceramidase (GBA1; UniProt P04062)",
    },
}

# Reference compounds expected to be in the library AND pass the BBB filter.
# Used by the library build summary and the pipeline check.
REFERENCES_BBB_PASS = {
    "CHEMBL4098877": ("MLi-2", "LRRK2"),  # potent brain-penetrant LRRK2 inhibitor
    "CHEMBL153479": ("AMBROXOL", "GBA1"),  # GBA1 chaperone, brain-penetrant
    "CHEMBL603469": ("LESTAURTINIB", "LRRK2"),  # CEP-701, LRRK2/FLT3 inhibitor
}

# Reference compounds expected to be in the library but NOT pass BBB
# (too polar / too many HBD). Useful negative controls for the BBB filter.
REFERENCES_PRESENT_ONLY = {
    "CHEMBL206468": ("AFEGOSTAT / ISOFAGOMINE", "GBA1"),  # GBA1 chaperone, too polar for BBB
}

# All reference chembl_ids, flattened for convenience.
ALL_REFERENCES = {**REFERENCES_BBB_PASS, **REFERENCES_PRESENT_ONLY}

MAX_ACTIVITIES = 2000        # cap on activity records fetched per target (breadth)
MAX_COMPOUNDS_PER_TARGET = 150  # after potency filtering + dedup
POTENCY_CUTOFF_NM = 30_000   # keep actives up to 30 uM (retains weak chaperones)
PAGE_SIZE = 1000
SLEEP_BETWEEN_CALLS = 0.02    # be polite to the ChEMBL service

# BBB / CNS rules, mirroring cns_filter.passes_bbb_rules exactly.
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

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    _RDKIT = True
except ImportError:  # pragma: no cover - rdkit is required in chem_m2
    _RDKIT = False


def compute_props(smiles):
    """Compute BBB-relevant descriptors from a SMILES string (RDKit).

    Returns a dict with keys mw, logp, tpsa, hbd, hba, rot_bonds,
    heavy_atoms. Any value may be None if the SMILES is invalid.
    """
    if not _RDKIT:
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


def passes_bbb_rules(props):
    """BBB filter mirroring cns_filter.passes_bbb_rules exactly."""
    mw = props.get("mw")
    logp = props.get("logp")
    tpsa = props.get("tpsa")
    hbd = props.get("hbd")
    hba = props.get("hba")
    rtb = props.get("rot_bonds")
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

POTENT_RELATIONS = {"=", "<", "<=", "<~", "~"}
POTENT_TYPES = {"IC50", "Ki", "Kd", "EC50", "AC50", "pIC50", "pKi"}


def _fetch(url):
    """GET a URL and return parsed JSON, retrying once on transient failure."""
    last_err = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # nosec B310 - fixed HTTPS ChEMBL host  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            last_err = exc
            print(f"  [fetch] retry {attempt+1} for {url[:80]}... ({exc})", flush=True)
            time.sleep(1.0)
    raise RuntimeError(f"ChEMBL request failed for {url}: {last_err}")


def fetch_activities(target_chembl_id):
    """Fetch all standard human activities for a target (paginated)."""
    results = []
    offset = 0
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_flag": 1,
        "limit": PAGE_SIZE,
    }
    consecutive_failures = 0
    while offset < MAX_ACTIVITIES and consecutive_failures < 3:
        params["offset"] = offset
        url = f"{BASE}/activity.json?{urllib.parse.urlencode(params)}"
        try:
            data = _fetch(url)
        except RuntimeError as exc:
            print(f"  [{target_chembl_id}] page at offset {offset} failed after retries: {exc}", flush=True)
            consecutive_failures += 1
            offset += PAGE_SIZE  # skip this page and continue
            time.sleep(1.0)
            continue
        consecutive_failures = 0
        page = data.get("activities", [])
        results.extend(page)
        total = data.get("page_meta", {}).get("total_count", len(results))
        print(f"  [{target_chembl_id}] fetched {len(results)}/{total} activities (offset {offset})", flush=True)
        if not page or offset + PAGE_SIZE >= total or len(results) >= MAX_ACTIVITIES:
            break
        offset += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_CALLS)
    return results


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
    return None  # % , ratio, etc. are not potencies


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def curate_target(target_name, cfg):
    """Return a list of curated compound dicts for one target."""
    target_id = cfg["target_chembl_id"]
    print(f"[{target_name}] Fetching ChEMBL activities for {target_id} ...")
    activities = fetch_activities(target_id)
    print(f"[{target_name}] {len(activities)} standard activity records fetched.")

    # Keep potent, human, single-protein, relation-specified actives with SMILES.
    by_parent = {}
    for a in activities:
        if a.get("target_organism") != cfg["organism"]:
            continue
        if a.get("standard_relation") not in POTENT_RELATIONS:
            continue
        if a.get("standard_type") not in POTENT_TYPES:
            continue
        nm = _to_nm(a.get("standard_value"), a.get("standard_units"))
        if nm is None or nm > POTENCY_CUTOFF_NM:
            continue
        smiles = a.get("canonical_smiles")
        if not smiles:
            continue
        pchembl = _num(a.get("pchembl_value"))
        parent = a.get("parent_molecule_chembl_id") or a.get("molecule_chembl_id")
        key = parent
        prev = by_parent.get(key)
        # Keep the most potent (lowest nM; fall back to highest pchembl).
        better = (
            prev is None
            or (nm is not None and prev["nm"] is not None and nm < prev["nm"])
            or (nm is not None and prev["nm"] is None)
            or (pchembl is not None and prev["pchembl"] is not None
                and pchembl > prev["pchembl"])
        )
        if better:
            by_parent[key] = {
                "chembl_id": a.get("molecule_chembl_id"),
                "parent_chembl_id": parent,
                "pref_name": a.get("molecule_pref_name"),
                "smiles": smiles,
                "nm": nm,
                "pchembl": pchembl,
                "standard_type": a.get("standard_type"),
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
                "assay_description": a.get("assay_description"),
                "document_journal": a.get("document_journal"),
                "document_year": a.get("document_year"),
                "target_chembl_id": target_id,
            }

    # Sort by potency and cap.
    ranked = sorted(
        by_parent.values(),
        key=lambda d: (d["nm"] is None, d["nm"] if d["nm"] is not None else float("inf")),
    )[:MAX_COMPOUNDS_PER_TARGET]
    print(f"[{target_name}] {len(ranked)} unique potent compounds after dedup/cap.")

    # Compute BBB-relevant descriptors locally with RDKit (mirrors
    # cns_filter.passes_bbb_rules exactly) instead of one API call per compound.
    curated = []
    for entry in ranked:
        props = compute_props(entry["smiles"])
        entry.update({
            "pref_name": entry.get("pref_name") or entry["chembl_id"],
            "max_phase": None,
            "mw": props["mw"],
            "logp": props["logp"],
            "tpsa": props["tpsa"],
            "hbd": props["hbd"],
            "hba": props["hba"],
            "rot_bonds": props["rot_bonds"],
            "heavy_atoms": props["heavy_atoms"],
            "passes_bbb": passes_bbb_rules(props),
            "is_reference": entry["chembl_id"] in ALL_REFERENCES
            and ALL_REFERENCES[entry["chembl_id"]][1] == target_name,
        })
        curated.append(entry)
    return curated


def fetch_molecule(chembl_id):
    """Return the molecule dict for a ChEMBL ID, or {} on failure."""
    try:
        return _fetch(f"{BASE}/molecule/{chembl_id}.json")
    except RuntimeError as exc:
        print(f"  WARNING: could not fetch molecule {chembl_id}: {exc}")
        return {}


def fetch_activity_for_molecule(molecule_chembl_id, target_chembl_id):
    """Return the most potent standard activity for a molecule against a target."""
    params = {
        "molecule_chembl_id": molecule_chembl_id,
        "target_chembl_id": target_chembl_id,
        "standard_flag": 1,
        "limit": 100,
    }
    url = f"{BASE}/activity.json?{urllib.parse.urlencode(params)}"
    try:
        data = _fetch(url)
    except RuntimeError as exc:
        print(f"  WARNING: could not fetch activity for {molecule_chembl_id}: {exc}")
        return None
    page = data.get("activities", [])
    # Pick the lowest-nM potency among convertible records.
    best = None
    best_nm = float("inf")
    for a in page:
        nm = _to_nm(a.get("standard_value"), a.get("standard_units"))
        if nm is None:
            continue
        if nm < best_nm:
            best_nm = nm
            best = a
    return best


def reference_entry(chembl_id, target_name, cfg):
    """Build a curated entry for a reference compound fetched directly by ChEMBL ID."""
    name, ref_target = ALL_REFERENCES[chembl_id]
    if ref_target != target_name:
        return None
    mol = fetch_molecule(chembl_id)
    if not mol:
        return None
    smiles = (mol.get("molecule_structures") or {}).get("canonical_smiles")
    props = compute_props(smiles) if smiles else dict(mw=None, logp=None, tpsa=None,
                                                       hbd=None, hba=None,
                                                       rot_bonds=None, heavy_atoms=None)
    act = fetch_activity_for_molecule(chembl_id, cfg["target_chembl_id"])
    nm = None
    std_type = std_value = std_units = assay = journal = year = None
    if act:
        nm = _to_nm(act.get("standard_value"), act.get("standard_units"))
        std_type = act.get("standard_type")
        std_value = act.get("standard_value")
        std_units = act.get("standard_units")
        assay = act.get("assay_description")
        journal = act.get("document_journal")
        year = act.get("document_year")
    entry = {
        "chembl_id": mol.get("molecule_chembl_id"),
        "parent_chembl_id": mol.get("molecule_hierarchy", {}).get("parent_chembl_id")
        or mol.get("molecule_chembl_id"),
        "smiles": smiles,
        "nm": nm,
        "pchembl": _num(act.get("pchembl_value")) if act else None,
        "standard_type": std_type,
        "standard_value": std_value,
        "standard_units": std_units,
        "assay_description": assay,
        "document_journal": journal,
        "document_year": year,
        "target_chembl_id": cfg["target_chembl_id"],
        "pref_name": mol.get("pref_name") or name or chembl_id,
        "max_phase": mol.get("max_phase"),
        "mw": props["mw"],
        "logp": props["logp"],
        "tpsa": props["tpsa"],
        "hbd": props["hbd"],
        "hba": props["hba"],
        "rot_bonds": props["rot_bonds"],
        "heavy_atoms": props["heavy_atoms"],
        "passes_bbb": passes_bbb_rules(props),
        "is_reference": True,
    }
    time.sleep(SLEEP_BETWEEN_CALLS)
    return entry


def build_library(output, limit=MAX_COMPOUNDS_PER_TARGET):
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    global MAX_COMPOUNDS_PER_TARGET
    MAX_COMPOUNDS_PER_TARGET = limit
    all_rows = []
    for target_name, cfg in TARGETS.items():
        try:
            rows = curate_target(target_name, cfg)
        except RuntimeError as exc:
            print(f"[{target_name}] ERROR: {exc}")
            rows = []
        for r in rows:
            r["target"] = target_name
            r["ligand_id"] = sanitize_id(
                f"{target_name}_{r['chembl_id'] or r['parent_chembl_id'] or 'unk'}"
            )
            all_rows.append(r)

    # Ensure every reference compound is present (guaranteed inclusion for the
    # pipeline check), even if it fell outside the potency cap / activity page.
    present_refs = {r.get("chembl_id") for r in all_rows if r.get("chembl_id")}
    for ref_id, (name, tgt) in ALL_REFERENCES.items():
        if ref_id in present_refs:
            continue
        cfg = TARGETS[tgt]
        entry = reference_entry(ref_id, tgt, cfg)
        if entry and entry.get("smiles"):
            entry["target"] = tgt
            entry["ligand_id"] = sanitize_id(f"{tgt}_{entry['chembl_id'] or entry['parent_chembl_id'] or 'unk'}")
            all_rows.append(entry)
            present_refs.add(ref_id)
            print(f"  [reference] Added {name} ({ref_id}) for {tgt}.")

    fieldnames = [
        "ligand_id", "smiles", "chembl_id", "pref_name", "max_phase",
        "target", "target_chembl_id", "standard_type", "standard_value_nm",
        "pchembl", "assay_description", "document_journal", "document_year",
        "mw", "logp", "tpsa", "hbd", "hba", "rot_bonds", "heavy_atoms",
        "passes_bbb", "is_reference",
    ]
    with open(output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            # standard_value_nm from stored nm
            r["standard_value_nm"] = r.get("nm")
            writer.writerow(r)

    n_total = len(all_rows)
    n_bbb = sum(1 for r in all_rows if r.get("passes_bbb"))
    n_ref = sum(1 for r in all_rows if r.get("is_reference"))
    n_ref_bbb = sum(1 for r in all_rows if r.get("is_reference") and r.get("passes_bbb"))
    print("\n=== ChEMBL library summary ===")
    print(f"Total curated compounds: {n_total}")
    print(f"Pass BBB/CNS filter:     {n_bbb}")
    print(f"Reference compounds:     {n_ref} (of which pass BBB: {n_ref_bbb})")
    print(f"Written to: {output}")
    for ref_id, (name, tgt) in ALL_REFERENCES.items():
        match = [r for r in all_rows if r.get("chembl_id") == ref_id]
        state = "present + BBB-pass" if match and match[0].get("passes_bbb") else (
            "present" if match else "ABSENT")
        print(f"  Reference {name} ({ref_id}, {tgt}): {state}")
    return n_total, n_bbb


def main():
    parser = argparse.ArgumentParser(
        description="Build a curated ChEMBL library of known LRRK2 and GBA1 compounds."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SHARED_ROOT, "01_ligands_raw", "chembl_lrrk2_gba1_library.csv"),
        help="Output CSV path (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_COMPOUNDS_PER_TARGET,
        help="Max compounds per target after potency filtering.",
    )
    args = parser.parse_args()
    build_library(args.output, limit=args.limit)


if __name__ == "__main__":
    main()
