#!/usr/bin/env python3
"""
benchmark.py - Does the docking/scoring stage rank known actives above decoys?

Retrospective virtual-screening benchmark for the LRRK2 / GBA1 pipeline:

  check     Verify the decoys mimic the actives (before spending GPU time).
  prepare   Build a labelled set: known actives (ChEMBL library) + property-matched,
            structurally dissimilar decoys, written as a 3D SDF plus labels.csv.
            (Mac mini, needs RDKit only)
  run       Dock the set with the same P2Rank pocket + GNINA settings the pipeline uses.
            (Mac mini orchestrates; GNINA runs on arch-2080 over SSH)
  evaluate  Compute ROC-AUC, average precision and enrichment factors with bootstrap
            confidence intervals, compare against trivial baselines, write a report.

If the pipeline cannot separate known actives from decoys, new "hits" from it are not
meaningful. See the generated report.md for a plain-language verdict and its caveats.

Decoys: pass --decoy-pool with a CSV of drug-like molecules (column `smiles`), e.g.
DUD-E / DUDE-Z / ZINC decoys (convert them with convert_decoys.py). Without one, the other target's
actives are used as a WEAK cross-target control and the report says so.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from pd_common import SHARED_ROOT, TARGETS, sanitize_id, validate_pdb_filename

DEFAULT_LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                               "chembl_lrrk2_gba1_library.csv")
MAX_POOL_ROWS = 2_000_000
MAX_DECOY_TANIMOTO = 0.35   # DUD-E-style: decoys must be topologically unlike every active
SCORE_COLUMNS = {           # name -> (SDF property, higher_is_better)
    "CNNscore": ("CNNscore", True),
    "CNNaffinity": ("CNNaffinity", True),
    "Vina (minimizedAffinity)": ("minimizedAffinity", False),
}
RDLogger.DisableLog("rdApp.*")
_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ----------------------------------------------------------------------------- chemistry
def standardize(smiles):
    """Largest fragment, canonical SMILES; None if unparsable."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    return Chem.MolToSmiles(mol), mol


def property_vector(mol):
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        Chem.GetFormalCharge(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
    ], dtype=float)


# roughly one "bin" per property: MW, logP, HBD, HBA, rotB, charge, TPSA, aromatic rings
PROP_SCALE = np.array([25.0, 0.5, 0.7, 1.0, 1.5, 0.7, 12.0, 0.8])


def select_decoys(active_mols, pool_smiles, ratio, seed=0, max_tanimoto=MAX_DECOY_TANIMOTO):
    """For each active pick `ratio` pool molecules with the closest property vector,
    excluding any pool molecule too similar (Morgan2 Tanimoto) to ANY active. No reuse."""
    rng = np.random.default_rng(seed)
    active_fps = [_FPGEN.GetFingerprint(m) for m in active_mols]
    pool = []
    seen = set()
    for smi in pool_smiles:
        can, mol = standardize(smi)
        if mol is None or can in seen or mol.GetNumHeavyAtoms() < 8:
            continue
        seen.add(can)
        fp = _FPGEN.GetFingerprint(mol)
        if max(DataStructs.BulkTanimotoSimilarity(fp, active_fps)) >= max_tanimoto:
            continue
        pool.append((can, mol))
    if not pool:
        return [], 0
    pool_props = np.array([property_vector(m) for _, m in pool])
    taken = np.zeros(len(pool), dtype=bool)
    chosen = []
    order = rng.permutation(len(active_mols))  # avoid ordering bias in who gets first pick
    for idx in order:
        target = property_vector(active_mols[idx])
        dist = np.linalg.norm((pool_props - target) / PROP_SCALE, axis=1)
        dist[taken] = np.inf
        for j in np.argsort(dist)[:ratio]:
            if not np.isfinite(dist[j]):
                break
            taken[j] = True
            chosen.append(pool[j][0])
    return chosen, len(pool)


# ----------------------------------------------------------------------------- metrics
def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0:
        return float("nan")
    # pessimistic tie-breaking: among equal scores, decoys come first
    order = np.lexsort((labels.astype(int), -np.asarray(scores, dtype=float)))
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def enrichment_factor(labels, scores, fraction):
    labels = np.asarray(labels, dtype=bool)
    n = len(labels)
    n_top = max(1, math.ceil(fraction * n))
    order = np.lexsort((labels.astype(int), -np.asarray(scores, dtype=float)))
    hits_top = labels[order][:n_top].sum()
    base = labels.mean()
    return float((hits_top / n_top) / base) if base > 0 else float("nan")


def bootstrap_ci(labels, scores, metric, n_boot=1000, seed=0):
    """Stratified bootstrap 95% CI (actives and decoys resampled separately)."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(metric(labels[idx], scores[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_scores(labels, scores, n_boot=1000, seed=0):
    out = {
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "ef1": enrichment_factor(labels, scores, 0.01),
        "ef5": enrichment_factor(labels, scores, 0.05),
        "ef10": enrichment_factor(labels, scores, 0.10),
    }
    out["roc_auc_ci95"] = bootstrap_ci(labels, scores, roc_auc, n_boot, seed)
    return out


def verdict(auc, ci_low, baseline_best_auc):
    if not np.isfinite(auc) or ci_low <= 0.5:
        return "No detectable signal (confidence interval includes random, AUC 0.5)."
    if baseline_best_auc is not None and auc <= baseline_best_auc + 0.05:
        return ("Signal is no better than a trivial property baseline (e.g. molecular weight): "
                "the score may only be tracking molecule size.")
    if auc < 0.6:
        return "Barely better than random."
    if auc < 0.7:
        return "Weak signal."
    if auc < 0.8:
        return "Moderate signal."
    return "Good signal (still validate on an independent target/set before relying on it)."


def bundled_nci_pool():
    """Convenience decoy pool shipped with RDKit (NCI first 5k structures; offline, no download).
    Not verified-inactive: prefer DUD-E / ZINC decoys for any result you intend to rely on."""
    path = os.path.join(RDConfig.RDDataDir, "NCI", "first_5K.smi")
    with open(path) as fh:
        return [line.split()[0] for line in fh if line.strip()]


def baseline_table(df):
    """Structure-only baselines (no docking) for a labelled frame with `smiles` and `label`."""
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    out = df.copy()
    out["Baseline: molecular weight"] = [Descriptors.MolWt(m) for m in mols]
    out["Baseline: heavy atoms"] = [m.GetNumHeavyAtoms() for m in mols]
    out["Baseline: logP"] = [Descriptors.MolLogP(m) for m in mols]
    out["Baseline: H-bond donors"] = [rdMolDescriptors.CalcNumHBD(m) for m in mols]
    out["Baseline: TPSA"] = [rdMolDescriptors.CalcTPSA(m) for m in mols]
    return out, [c for c in out.columns if c.startswith("Baseline")]


def decoy_quality(df, n_boot=200, seed=0):
    """AUC of each trivial property for actives vs decoys. Values far from 0.5 mean the decoys
    do not mimic the actives and a docking score could win just by tracking that property."""
    df, cols = baseline_table(df)
    y = df["label"].to_numpy()
    return {c: {"roc_auc": roc_auc(y, df[c].to_numpy()),
                "roc_auc_ci95": bootstrap_ci(y, df[c].to_numpy(), roc_auc, n_boot, seed)} for c in cols}


def cmd_check(args):
    """Decoy sanity check that needs no docking: how well do decoys match the actives?"""
    outdir = check_under_root(args.outdir or default_outdir(args.target))
    df = pd.read_csv(os.path.join(outdir, "labels.csv"))
    res = decoy_quality(df, args.bootstrap, args.seed)
    print(f"Decoy quality for {args.target} ({int(df['label'].sum())} actives, "
          f"{int((df['label'] == 0).sum())} decoys). AUC near 0.5 = well matched.")
    worst = 0.0
    for name, r in res.items():
        dev = abs(r["roc_auc"] - 0.5)
        worst = max(worst, dev)
        flag = "  <-- poorly matched" if dev > 0.15 else ""
        print(f"  {name:<28} AUC {r['roc_auc']:.3f}{flag}")
    print("OK: properties look matched." if worst <= 0.15 else
          "WARNING: some properties differ strongly; a docking score may win by tracking them.")
    return res


# ----------------------------------------------------------------------------- stages
def default_outdir(target):
    return os.path.join(SHARED_ROOT, "07_benchmark", target)


def check_under_root(path):
    real, root = os.path.realpath(path), os.path.realpath(SHARED_ROOT)
    if real != root and not real.startswith(root + os.sep):
        sys.exit(f"{path} must live under {SHARED_ROOT}")
    return real


def load_actives(library_csv, target, min_pchembl, mode, source_class):
    df = pd.read_csv(library_csv)
    df = df[(df["target"] == target) & (df["pchembl"] >= min_pchembl)]
    if mode != "any" and "mode_hint" in df:
        df = df[df["mode_hint"] == mode]
    if source_class != "any" and "source_class" in df:
        df = df[df["source_class"] == source_class]
    rows, mols = [], []
    for _, r in df.iterrows():
        can, mol = standardize(r["smiles"])
        if mol is None:
            continue
        rows.append({"id": "act_" + sanitize_id(r["ligand_id"]), "smiles": can, "label": 1,
                     "role": "active", "pchembl": r["pchembl"],
                     "source_class": r.get("source_class", ""), "mode_hint": r.get("mode_hint", "")})
        mols.append(mol)
    return rows, mols


def embed_to_sdf(records, sdf_path, seed):
    """Write 3D conformers; returns ids that failed to embed."""
    writer = Chem.SDWriter(sdf_path)
    failed = []
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    for rec in records:
        mol = Chem.AddHs(Chem.MolFromSmiles(rec["smiles"]))
        try:
            embedded = AllChem.EmbedMolecule(mol, params) == 0
        except RuntimeError:  # some exotic molecules violate distance-geometry bounds
            embedded = False
        if not embedded:
            failed.append(rec["id"])
            continue
        with contextlib.suppress(Exception):  # optimisation is best-effort
            AllChem.MMFFOptimizeMolecule(mol)
        mol.SetProp("_Name", rec["id"])
        writer.write(mol)
    writer.close()
    return failed


def cmd_prepare(args):
    outdir = check_under_root(args.outdir or default_outdir(args.target))
    os.makedirs(outdir, exist_ok=True)
    actives, active_mols = load_actives(args.library, args.target, args.min_pchembl,
                                        args.mode, args.source_class)
    if len(actives) < 5:
        sys.exit(f"Only {len(actives)} actives for {args.target} after filtering; need >= 5.")

    warnings = []
    if args.decoy_pool == "rdkit-nci":
        pool_smiles = bundled_nci_pool()
        warnings.append("Decoys come from RDKit's bundled NCI set (convenience pool, not verified "
                        "inactives). Use DUD-E / ZINC decoys for results you intend to rely on.")
        print("WARNING:", warnings[-1])
    elif args.decoy_pool:
        pool_df = pd.read_csv(args.decoy_pool, nrows=MAX_POOL_ROWS)
        if "smiles" not in pool_df:
            sys.exit("--decoy-pool needs a `smiles` column")
        pool_smiles = pool_df["smiles"].dropna().tolist()
    else:
        other = [t for t in TARGETS if t != args.target][0]
        other_rows, _ = load_actives(args.library, other, 0.0, "any", "any")
        pool_smiles = [r["smiles"] for r in other_rows]
        warnings.append(
            f"No --decoy-pool given: decoys are {other} actives (cross-target control). These are "
            "real bioactive molecules, not property-matched inactives, so the result is only a weak "
            "sanity check. Use DUD-E / ZINC decoys for a meaningful benchmark.")
        print("WARNING:", warnings[-1])

    decoy_smiles, pool_size = select_decoys(active_mols, pool_smiles, args.decoy_ratio, args.seed)
    if not decoy_smiles:
        sys.exit("No usable decoys after the dissimilarity filter.")
    decoys = [{"id": f"dec_{i:05d}", "smiles": s, "label": 0, "role": "decoy",
               "pchembl": np.nan, "source_class": "", "mode_hint": ""}
              for i, s in enumerate(decoy_smiles)]
    if len(decoys) < args.decoy_ratio * len(actives) * 0.5:
        warnings.append(f"Only {len(decoys)} decoys for {len(actives)} actives (pool of {pool_size} "
                        "after filtering); enrichment estimates will be noisy.")
        print("WARNING:", warnings[-1])

    records = actives + decoys
    failed = embed_to_sdf(records, os.path.join(outdir, "ligands.sdf"), args.seed)
    labels = pd.DataFrame(records)
    labels["embedded"] = ~labels["id"].isin(failed)
    labels.to_csv(os.path.join(outdir, "labels.csv"), index=False)

    with open(args.library, "rb") as fh:
        lib_hash = hashlib.sha256(fh.read()).hexdigest()
    manifest = {
        "target": args.target, "n_actives": len(actives), "n_decoys": len(decoys),
        "n_failed_embedding": len(failed), "decoy_pool_after_filter": pool_size,
        "decoy_ratio": args.decoy_ratio, "min_pchembl": args.min_pchembl, "mode": args.mode,
        "source_class": args.source_class, "seed": args.seed,
        "decoy_source": args.decoy_pool or "cross-target actives (weak control)",
        "library_sha256": lib_hash, "warnings": warnings,
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Prepared {len(actives)} actives + {len(decoys)} decoys in {outdir}")


def cmd_run(args):
    # Imported here so `prepare`/`evaluate` work without paramiko-configured nodes.
    import master_pipeline as mp

    outdir = check_under_root(args.outdir or default_outdir(args.target))
    ligands = os.path.join(outdir, "ligands.sdf")
    if not os.path.isfile(ligands):
        sys.exit("Run `prepare` first.")
    cx, cy, cz, pdb_path = mp.get_top_pocket_coords(args.target, args.pdb, args.pocket_index)
    scored = os.path.join(outdir, "scored.sdf")
    nodes = mp.get_nodes()
    gpu = nodes["arch_2080"]["gpu_id"]
    cmd = mp.gnina_docker_cmd(gpu, (
        f"-r {mp.q(pdb_path)} -l {mp.q(ligands)} "
        f"--center_x {cx:.2f} --center_y {cy:.2f} --center_z {cz:.2f} "
        f"--size_x 22 --size_y 22 --size_z 22 --num_modes 3 "
        f"--cnn_scoring rescore --cpu 4 --seed 0 -o {mp.q(scored)}"
    ), out_dir=outdir)
    mp.run_remote_ssh("arch_2080", cmd, "Benchmark-GNINA")
    with open(os.path.join(outdir, "run.json"), "w") as fh:
        json.dump({"pdb": args.pdb, "pocket_index": args.pocket_index,
                   "center": [cx, cy, cz], "box": 22, "gnina_image": mp.GNINA_IMAGE}, fh, indent=2)
    print(f"Docking finished: {scored}\nNext: benchmark.py evaluate --target {args.target}")


def read_best_scores(scored_sdf):
    """Best value per ligand for each score column."""
    best = {}
    for mol in Chem.SDMolSupplier(scored_sdf, removeHs=False):
        if mol is None or not mol.HasProp("_Name"):
            continue
        entry = best.setdefault(mol.GetProp("_Name"), {})
        for name, (prop, higher) in SCORE_COLUMNS.items():
            if not mol.HasProp(prop):
                continue
            val = float(mol.GetProp(prop))
            val = val if higher else -val
            if name not in entry or val > entry[name]:
                entry[name] = val
    return pd.DataFrame.from_dict(best, orient="index")


def cmd_evaluate(args):
    outdir = check_under_root(args.outdir or default_outdir(args.target))
    labels = pd.read_csv(os.path.join(outdir, "labels.csv"))
    labels = labels[labels["embedded"]].copy()
    scores = read_best_scores(os.path.join(outdir, "scored.sdf"))
    df = labels.merge(scores, left_on="id", right_index=True, how="left")
    manifest = json.load(open(os.path.join(outdir, "manifest.json")))

    # Trivial baselines computed from structure only (no docking).
    df, baseline_cols = baseline_table(df)

    y = df["label"].to_numpy()
    results, n_undocked = {}, {}
    for col in list(SCORE_COLUMNS) + baseline_cols:
        if col not in df:
            continue
        s = df[col]
        n_undocked[col] = int(s.isna().sum())
        s = s.fillna(s.min() - 1.0 if s.notna().any() else 0.0)  # failed dockings rank last
        results[col] = evaluate_scores(y, s.to_numpy(), args.bootstrap, args.seed)

    base_auc = max((abs(results[c]["roc_auc"] - 0.5) + 0.5 for c in baseline_cols if c in results),
                   default=None)
    for col in SCORE_COLUMNS:
        if col in results:
            results[col]["verdict"] = verdict(results[col]["roc_auc"], results[col]["roc_auc_ci95"][0],
                                              base_auc)
            results[col]["n_missing_scores"] = n_undocked[col]

    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump({"manifest": manifest, "results": results}, fh, indent=2)
    write_report(outdir, manifest, results, df)
    print(open(os.path.join(outdir, "report.md")).read())


def write_report(outdir, manifest, results, df):
    n_act, n_dec = int(df["label"].sum()), int((df["label"] == 0).sum())
    lines = [
        f"# Benchmark report: {manifest['target']}", "",
        f"{n_act} known actives vs {n_dec} decoys (ratio 1:{n_dec / max(n_act, 1):.0f}). "
        "AUC 0.5 = random; EF = how many times more actives than chance appear in the top fraction.", "",
        "| Score | ROC-AUC (95% CI) | Avg precision | EF1% | EF5% | EF10% |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lo, hi = r["roc_auc_ci95"]
        lines.append(f"| {name} | {r['roc_auc']:.3f} ({lo:.3f}-{hi:.3f}) | {r['average_precision']:.3f} "
                     f"| {r['ef1']:.1f} | {r['ef5']:.1f} | {r['ef10']:.1f} |")
    lines += ["", "## Verdicts", ""]
    for name in SCORE_COLUMNS:
        if name in results:
            lines.append(f"- **{name}**: {results[name]['verdict']}")
    lines += ["", "## How to read this", "",
              f"- Actives are ChEMBL compounds with pChEMBL >= {manifest['min_pchembl']} "
              f"(mode `{manifest['mode']}`, source `{manifest['source_class']}`).",
              f"- Decoy source: {manifest['decoy_source']}.",
              "- The `Baseline:` rows use no docking at all. If one of them scores as well as the "
              "docking scores, the decoys are not well matched or the docking is only tracking size.",
              "- A good result here is necessary, not sufficient: it does not show a new compound "
              "works, and it does not transfer automatically to other targets."]
    if manifest.get("warnings"):
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in manifest["warnings"]]
    if manifest["target"] == "LRRK2":
        lines += ["", "- Most LRRK2 library actives come from broad kinome panels (many promiscuous "
                      "kinase inhibitors). Re-run with `--source-class dedicated_assay` for a stricter set "
                      "(very few compounds)."]
    if manifest["target"] == "GBA1":
        lines += ["", "- GBA1 actives are mostly active-site inhibitors; the default pocket choice for "
                      "GBA1 is an unverified assumption. Check `run.json` and the structure."]
    with open(os.path.join(outdir, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--target", required=True, choices=sorted(TARGETS))
        p.add_argument("--outdir", help=f"default: {SHARED_ROOT}/07_benchmark/<target>")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prepare", help="build actives + decoys")
    common(p)
    p.add_argument("--library", default=DEFAULT_LIBRARY)
    p.add_argument("--decoy-pool",
                   help="CSV with a `smiles` column (e.g. DUD-E / ZINC decoys), or `rdkit-nci` for the "
                        "small convenience pool bundled with RDKit")
    p.add_argument("--decoy-ratio", type=int, default=20, help="decoys per active (default 20)")
    p.add_argument("--min-pchembl", type=float, default=6.0, help="activity cutoff (6.0 = 1 uM)")
    p.add_argument("--mode", default="any", choices=["any", "inhibitor_or_binder", "activator_or_chaperone"])
    p.add_argument("--source-class", default="any", choices=["any", "dedicated_assay", "kinome_panel"])
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="dock the set with GNINA on arch-2080")
    common(p)
    p.add_argument("--pdb", required=True, type=validate_pdb_filename, help="e.g. 7LHT_clean.pdb")
    p.add_argument("--pocket-index", type=int, default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check", help="decoy-quality check (no docking needed)")
    common(p)
    p.add_argument("--bootstrap", type=int, default=200)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("evaluate", help="compute metrics and write report.md")
    common(p)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
