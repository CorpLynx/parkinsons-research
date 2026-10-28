#!/usr/bin/env python3
"""
check_controls.py - Benchmark the screen against known binders and decoys.

A docking/scoring pipeline is only trustworthy if it ranks molecules that are
*known* to bind a target above molecules that are *known* not to. This script
turns that sanity check into a number.

Two modes:

  spike     Merge the control compounds for a target into a CNS-filtered ligand
            CSV so they are docked alongside the real library. Controls bypass
            the blood-brain-barrier filter on purpose: a negative control such
            as glucose would normally be dropped before docking, and then it
            could never provide a low score to compare against.

                check_controls.py spike --target LRRK2 \
                    --filtered run/.../02_cns_filtered/LRRK2_cns.csv \
                    --controls data/controls.csv \
                    --out run/.../02_cns_filtered/LRRK2_cns_spiked.csv

  evaluate  Read a scored results CSV (any CSV with a ligand_id column and a
            score column, e.g. the pipeline's *_all_scores.csv) and report how
            well the positive controls separated from the negative controls:
            per-control scores, median of each group, and the ROC-AUC (the
            probability a random positive outranks a random negative; 1.0 is
            perfect, 0.5 is coin-flip). Exits non-zero if controls are missing
            or, with --min-auc, if separation is below the given threshold.

                check_controls.py evaluate --target LRRK2 \
                    --scores run/.../LRRK2_all_scores.csv \
                    --controls data/controls.csv \
                    --score-col CNNscore --min-auc 0.7

The verdict is a *sanity* signal, not proof of a good model: the negatives
here are obvious-inactive decoys, not property-matched DUD-E decoys, so a high
AUC rules out gross scoring failure rather than confirming fine discrimination.
"""
import argparse
import os
import re
import sys

import pandas as pd

# Mirror pd_common.sanitize_id without importing it, so this tool also runs
# standalone (e.g. in CI) outside the pipeline's environment.
_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")

# Cap pathological inputs the way cns_filter.py does, so a malformed or
# enormous CSV fails fast instead of exhausting memory.
MAX_INPUT_ROWS = 2_000_000
VALID_ROLES = {"positive", "negative"}


class InputError(ValueError):
    """A user-facing problem with an input file or argument (clean exit, no traceback)."""


def sanitize_id(value):
    cleaned = _ID_RE.sub("_", str(value)).strip(".")
    return cleaned[:64] or "unnamed"


def read_csv(path, what):
    """Read a CSV with existence/emptiness checks and a row cap; InputError on failure."""
    if not os.path.isfile(path):
        raise InputError(f"{what} file not found: {path}")
    try:
        df = pd.read_csv(path, nrows=MAX_INPUT_ROWS)
    except pd.errors.EmptyDataError as exc:
        raise InputError(f"{what} file is empty: {path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise InputError(f"{what} file could not be parsed as CSV ({path}): {exc}") from exc
    if df.empty:
        raise InputError(f"{what} file has no data rows: {path}")
    return df


def load_controls(controls_csv, target):
    df = read_csv(controls_csv, "controls")
    required = {"ligand_id", "smiles", "target", "control_role"}
    missing = required - set(df.columns)
    if missing:
        raise InputError(f"{controls_csv} missing column(s): {sorted(missing)}")
    df = df.dropna(subset=["ligand_id", "smiles", "target", "control_role"]).copy()
    df["control_role"] = df["control_role"].astype(str).str.strip().str.lower()
    bad_roles = set(df["control_role"]) - VALID_ROLES
    if bad_roles:
        raise InputError(
            f"{controls_csv} has unknown control_role value(s) {sorted(bad_roles)}; "
            f"expected only {sorted(VALID_ROLES)}")
    df = df[df["target"].astype(str).str.upper() == target.upper()].copy()
    if df.empty:
        raise InputError(f"No controls for target {target!r} in {controls_csv}")
    df["ligand_id"] = df["ligand_id"].map(sanitize_id)
    dupes = df["ligand_id"][df["ligand_id"].duplicated()].unique()
    if len(dupes):
        raise InputError(
            f"{controls_csv} has duplicate ligand_id(s) for {target}: {sorted(dupes)}")
    return df


def spike(args):
    if os.path.abspath(args.out) == os.path.abspath(args.filtered):
        raise InputError("--out must differ from --filtered (refusing to overwrite the input)")
    controls = load_controls(args.controls, args.target)
    filtered = read_csv(args.filtered, "filtered-ligand")
    missing = {"ligand_id", "smiles"} - set(filtered.columns)
    if missing:
        raise InputError(f"{args.filtered} needs column(s): {sorted(missing)}")
    filtered = filtered.dropna(subset=["ligand_id", "smiles"]).copy()
    filtered["ligand_id"] = filtered["ligand_id"].map(sanitize_id)

    ctrl_slim = controls[["ligand_id", "smiles"]]
    overlap = set(filtered["ligand_id"]) & set(ctrl_slim["ligand_id"])
    combined = pd.concat([filtered, ctrl_slim[~ctrl_slim["ligand_id"].isin(overlap)]],
                         ignore_index=True)
    combined = combined.drop_duplicates(subset="ligand_id", keep="first")
    combined.to_csv(args.out, index=False)
    n_added = len(combined) - len(filtered.drop_duplicates(subset="ligand_id"))
    print(f"[spike] {args.target}: wrote {len(combined)} ligands to {args.out} "
          f"({n_added} controls added, {len(overlap)} already present in library).")
    return 0


def roc_auc(pos_scores, neg_scores):
    """ROC-AUC via the Mann-Whitney U statistic, with average ranks for ties.

    Returns the probability that a random positive scores above a random
    negative (ties count as 0.5). None if either group is empty.
    """
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None
    combined = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    combined.sort(key=lambda t: t[0])
    # Average ranks (1-based) over tied score groups.
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, (_, lab) in zip(ranks, combined, strict=True) if lab == 1)
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_pos / (n_pos * n_neg)


def evaluate(args):
    controls = load_controls(args.controls, args.target)
    scores = read_csv(args.scores, "scores")
    if "ligand_id" not in scores.columns:
        raise InputError(f"{args.scores} needs a 'ligand_id' column")
    if args.score_col not in scores.columns:
        raise InputError(f"{args.scores} has no column {args.score_col!r}; "
                         f"available: {list(scores.columns)}")
    scores = scores.copy()
    scores["ligand_id"] = scores["ligand_id"].map(sanitize_id)
    # Coerce the score column to numbers; a non-numeric score would otherwise
    # make groupby().max() compare strings and silently corrupt the ranking.
    raw = scores[args.score_col]
    scores[args.score_col] = pd.to_numeric(raw, errors="coerce")
    n_bad = int(scores[args.score_col].isna().sum() - raw.isna().sum())
    if n_bad:
        print(f"[warn] {n_bad} row(s) had a non-numeric {args.score_col!r} and were ignored.")
    scores = scores.dropna(subset=[args.score_col])
    if scores.empty:
        raise InputError(f"No numeric {args.score_col!r} values in {args.scores}")
    # Best (max) score per ligand, in case several poses are present.
    best = scores.groupby("ligand_id")[args.score_col].max()

    merged = controls.merge(
        best.rename("score"), left_on="ligand_id", right_index=True, how="left")
    scored = merged[merged["score"].notna()]
    unscored = merged[merged["score"].isna()]

    pos = scored[scored["control_role"] == "positive"]
    neg = scored[scored["control_role"] == "negative"]

    print(f"\n=== Control benchmark: {args.target} (score = {args.score_col}) ===")
    print(f"controls defined: {len(controls)}  scored: {len(scored)}  "
          f"not found in results: {len(unscored)}")
    if len(unscored):
        print("  missing (never docked/scored): "
              + ", ".join(sorted(unscored["ligand_id"])[:12])
              + (" ..." if len(unscored) > 12 else ""))

    def show(group, label):
        print(f"\n  {label} controls ({len(group)}):")
        for _, r in group.sort_values("score", ascending=False).iterrows():
            print(f"    {r['score']:+7.3f}  {r['ligand_id']:32} "
                  f"{r.get('control_note', '') or ''}")

    verdict_code = 0
    if len(pos):
        show(pos, "POSITIVE")
    if len(neg):
        show(neg, "NEGATIVE")

    if len(pos) and len(neg):
        auc = roc_auc(list(pos["score"]), list(neg["score"]))
        print(f"\n  median positive score: {pos['score'].median():+.3f}")
        print(f"  median negative score: {neg['score'].median():+.3f}")
        print(f"  ROC-AUC (positives vs negatives): {auc:.3f}")
        if args.min_auc is not None:
            if auc >= args.min_auc:
                print(f"  VERDICT: PASS (AUC {auc:.3f} >= {args.min_auc})")
            else:
                print(f"  VERDICT: FAIL (AUC {auc:.3f} < {args.min_auc}) - "
                      f"the screen is not ranking known binders above decoys.")
                verdict_code = 2
        print("\n  Note: negatives are obvious-inactive sanity decoys, not "
              "property-matched DUD-E decoys; a high AUC rules out gross scoring\n"
              "  failure, it does not prove fine-grained discrimination.")
    else:
        print("\n  Not enough scored controls in both groups to compute AUC.")
        if args.min_auc is not None:
            verdict_code = 2

    if args.require_all and len(unscored):
        print(f"\n  VERDICT: FAIL ({len(unscored)} controls were never scored; "
              f"--require-all was set).")
        verdict_code = max(verdict_code, 2)
    return verdict_code


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("spike", help="merge controls into a filtered ligand CSV")
    sp.add_argument("--target", required=True)
    sp.add_argument("--filtered", required=True, help="CNS-filtered CSV (ligand_id, smiles)")
    sp.add_argument("--controls", default="data/controls.csv")
    sp.add_argument("--out", required=True, help="output spiked CSV path")
    sp.set_defaults(func=spike)

    ev = sub.add_parser("evaluate", help="score positives-vs-negatives separation")
    ev.add_argument("--target", required=True)
    ev.add_argument("--scores", required=True, help="results CSV (ligand_id + score column)")
    ev.add_argument("--controls", default="data/controls.csv")
    ev.add_argument("--score-col", default="CNNscore")
    ev.add_argument("--min-auc", type=float, default=None,
                    help="fail (exit 2) if ROC-AUC is below this value")
    ev.add_argument("--require-all", action="store_true",
                    help="fail if any control was not found in the results")
    ev.set_defaults(func=evaluate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
