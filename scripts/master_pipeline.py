#!/usr/bin/env python3
"""
master_pipeline.py - Master Orchestrator (Runs on Mac mini M2)
Coordinates ligand filtering (M2), P2Rank + DiffDock (Arch 4070 Ti Super),
Dockerized 3D CNN GNINA screening and DiffDock pose rescoring (Arch 2080 Super),
and ADMET-AI triage (Arch 2080 Super).

Each invocation creates a versioned run directory under
{SHARED_ROOT}/runs/{RUN_ID}/{target}/ so multiple campaigns can be run,
compared, and resumed without clobbering previous results.

Node addresses/users come from environment variables (see .env.example).
Remote hosts must already be in ~/.ssh/known_hosts; unknown hosts are rejected.
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import paramiko
from rdkit import Chem

from pd_common import (
    GNINA_IMAGE, MAMBA_BIN, SHARED_ROOT, node_config, sanitize_id,
    validate_pdb_filename,
)

P2RANK_BIN = f"{SHARED_ROOT}/tools/p2rank_2.4.2/prank"
DIFFDOCK_DIR = f"{SHARED_ROOT}/tools/DiffDock"
SSH_TIMEOUT = 30
REMOTE_JOB_TIMEOUT = 24 * 3600

MIN_CNN_SCORE = 0.65
ADMET_THRESHOLDS = {"BBB_Martins": (">=", 0.80), "hERG": ("<=", 0.30), "DILI": ("<=", 0.40)}

_nodes = {}


def get_nodes():
    if not _nodes:
        _nodes["arch_4070"] = node_config("ARCH4070")
        _nodes["arch_2080"] = node_config("ARCH2080")
    return _nodes


def q(value):
    """Shell-quote a value for the remote command line."""
    return shlex.quote(str(value))


def remote_path(path):
    """Quote a path but keep a leading ~ expandable on the remote side."""
    path = str(path)
    if path.startswith("~/"):
        return f'"$HOME"/{shlex.quote(path[2:])}'
    return shlex.quote(path)


def run_remote_ssh(node_key, command, stage_label, dry_run=False):
    """Executes a command on the specified Arch Linux GPU node over SSH (key auth, verified host).

    With dry_run=True the command is printed instead of executed, so the full
    command surface can be verified on the Mac mini before touching the GPU nodes.
    """
    if dry_run:
        cfg = get_nodes()[node_key]
        print(f"[dry-run:{stage_label}] would run on {node_key} ({cfg['host']}):")
        print(f"  {command}")
        return
    cfg = get_nodes()[node_key]
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(
            hostname=cfg["host"], username=cfg["user"],
            timeout=SSH_TIMEOUT, allow_agent=True, look_for_keys=True,
        )
        print(f"[{stage_label}] Connected to {node_key} ({cfg['host']})...")
        _, stdout, _ = client.exec_command(  # nosec B601 - callers shlex-quote every interpolated value
            command, get_pty=True, timeout=REMOTE_JOB_TIMEOUT)
        for line in iter(stdout.readline, ""):
            sys.stdout.write(f"  [{stage_label}] {line}")
        exit_status = stdout.channel.recv_exit_status()
    finally:
        client.close()
    if exit_status != 0:
        raise RuntimeError(f"[{stage_label}] Failed on {node_key} with exit code {exit_status}")
    print(f"[{stage_label}] Completed successfully.")


def now_str():
    """Current timestamp as a compact run-ID-friendly string."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_root(run_id):
    """Top-level directory for a run: {SHARED_ROOT}/runs/{run_id}/"""
    return os.path.join(SHARED_ROOT, "runs", run_id)


def target_run_dir(run_id, target):
    """Per-target run directory: {SHARED_ROOT}/runs/{run_id}/{target}/"""
    return os.path.join(run_root(run_id), target)


def stage_dir(run_id, target, stage_name):
    """Directory for a pipeline stage output: .../{run_id}/{target}/{stage_name}/"""
    d = os.path.join(target_run_dir(run_id, target), stage_name)
    os.makedirs(d, exist_ok=True)
    return d


def manifest_path(run_id, target):
    return os.path.join(target_run_dir(run_id, target), "manifest.json")


def load_manifest(run_id, target):
    p = manifest_path(run_id, target)
    if os.path.isfile(p):
        with open(p) as fh:
            return json.load(fh)
    return {"run_id": run_id, "target": target, "stages": {}, "status": "running"}


def save_manifest(run_id, target, manifest):
    p = manifest_path(run_id, target)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(manifest, fh, indent=2)


def stage_done(manifest, stage_name):
    return manifest.get("stages", {}).get(stage_name, {}).get("done", False)


def mark_stage_done(manifest, stage_name, detail=None):
    manifest.setdefault("stages", {})[stage_name] = {
        "done": True,
        "finished_at": datetime.datetime.now().isoformat(),
        "detail": detail or {},
    }
    save_manifest(manifest["run_id"], manifest["target"], manifest)


def select_pocket(df, target):
    """Choose which P2Rank pocket to dock into.

    LRRK2 -> rank 1 (the ATP-binding catalytic pocket).
    GBA1  -> rank 2 (the allosteric site used by chemical chaperones);
             falls back to rank 1 when fewer than two pockets were predicted.

    Returns (pocket_index, center_x, center_y, center_z).
    """
    if df is None or len(df) == 0:
        raise RuntimeError("P2Rank returned no pockets to select from")
    target = target.upper()
    if target == "LRRK2":
        pocket_idx = 0
    else:
        pocket_idx = 1 if len(df) > 1 else 0
    row = df.iloc[pocket_idx]
    return (
        pocket_idx,
        float(row["center_x"]),
        float(row["center_y"]),
        float(row["center_z"]),
    )


def get_top_pocket_coords(run_id, target, pdb_filename, dry_run=False):
    """Runs P2Rank on arch_4070 and extracts the predicted pocket center (X, Y, Z)."""
    pdb_path = f"{SHARED_ROOT}/00_targets/{target}/{pdb_filename}"
    out_dir = stage_dir(run_id, target, "03_pockets")
    cmd = f"{q(P2RANK_BIN)} predict -f {q(pdb_path)} -o {q(out_dir)}"
    run_remote_ssh("arch_4070", cmd, "P2Rank-PocketFinder", dry_run=dry_run)

    predictions_csv = f"{out_dir}/{pdb_filename}_predictions.csv"
    if dry_run:
        print(f"[dry-run] P2Rank output would be: {predictions_csv}")
        return 0.0, 0.0, 0.0, pdb_path
    df = pd.read_csv(predictions_csv, skipinitialspace=True)
    pocket_idx, cx, cy, cz = select_pocket(df, target)
    print(f"[Pocket Coords] Selected Pocket #{pocket_idx+1} for {target}: X={cx:.2f}, Y={cy:.2f}, Z={cz:.2f}")
    return cx, cy, cz, pdb_path


def build_diffdock_csv(pdb_path, filtered_csv, diffdock_csv, max_ligands=250):
    """Creates the batch input CSV required by DiffDock."""
    df = pd.read_csv(filtered_csv).head(max_ligands)
    records = [{
        "complex_name": f"complex_{sanitize_id(row['ligand_id'])}",
        "protein_path": pdb_path,
        "ligand_description": row["smiles"],
        "protein_sequence": "",
    } for _, row in df.iterrows()]
    pd.DataFrame(records).to_csv(diffdock_csv, index=False)


def combine_diffdock_poses(diffdock_out, combined_sdf):
    """Collects each complex's top-ranked DiffDock pose into one SDF for GNINA rescoring."""
    writer = Chem.SDWriter(combined_sdf)
    count = 0
    if os.path.isdir(diffdock_out):
        for name in sorted(os.listdir(diffdock_out)):
            pose = os.path.join(diffdock_out, name, "rank1.sdf")
            if not name.startswith("complex_") or not os.path.isfile(pose):
                continue
            mol = next(iter(Chem.SDMolSupplier(pose, removeHs=False)), None)
            if mol is None:
                continue
            mol.SetProp("_Name", name[len("complex_"):])
            writer.write(mol)
            count += 1
    writer.close()
    return count


def collect_poses(docked_sdf, source):
    """Reads GNINA-annotated poses; returns one dict per pose."""
    if not os.path.isfile(docked_sdf):
        return []
    poses = []
    for mol in Chem.SDMolSupplier(docked_sdf):
        if mol is None or not mol.HasProp("_Name"):
            continue
        poses.append({
            "ligand_id": mol.GetProp("_Name"),
            "CNNscore": float(mol.GetProp("CNNscore")) if mol.HasProp("CNNscore") else 0.0,
            "CNNaffinity": float(mol.GetProp("CNNaffinity")) if mol.HasProp("CNNaffinity") else 0.0,
            "pose_source": source,
        })
    return poses


def extract_top_hits(poses, filtered_csv, top_hits_csv, min_cnn_score=MIN_CNN_SCORE):
    """Keeps the single best pose per ligand above threshold; SMILES come from the filtered CSV."""
    columns = ["ligand_id", "smiles", "CNNscore", "CNNaffinity", "pose_source"]
    df = pd.DataFrame(poses, columns=["ligand_id", "CNNscore", "CNNaffinity", "pose_source"])
    df = df[df["CNNscore"] >= min_cnn_score]
    df = df.sort_values("CNNscore", ascending=False).drop_duplicates("ligand_id", keep="first")
    smiles = pd.read_csv(filtered_csv)[["ligand_id", "smiles"]]
    smiles["ligand_id"] = smiles["ligand_id"].map(sanitize_id)
    df = df.merge(smiles, on="ligand_id", how="left")[columns]
    df.to_csv(top_hits_csv, index=False)
    print(f"[Hit Triage] Extracted {len(df)} high-confidence binders (CNNscore >= {min_cnn_score}).")
    return len(df)


def gnina_docker_cmd(gpu_id, args, run_id, target):
    """Hardened GNINA container invocation: no extra privileges, non-root, read-only inputs."""
    run_target = target_run_dir(run_id, target)
    return (
        f"docker run --rm --gpus {q(chr(34) + 'device=' + gpu_id + chr(34))} "
        f"--cap-drop ALL --security-opt no-new-privileges "
        f"--user \"$(id -u):$(id -g)\" "
        f"-v {q(SHARED_ROOT)}:{q(SHARED_ROOT)}:ro "
        f"-v {q(run_target)}:{q(run_target)} "
        f"{q(GNINA_IMAGE)} gnina {args}"
    )


def apply_admet_filters(df_admet):
    missing = [c for c in ADMET_THRESHOLDS if c not in df_admet.columns]
    if missing:
        raise RuntimeError(f"ADMET-AI output is missing expected column(s): {missing}")
    mask = pd.Series(True, index=df_admet.index)
    for col, (op, limit) in ADMET_THRESHOLDS.items():
        mask &= (df_admet[col] >= limit) if op == ">=" else (df_admet[col] <= limit)
    return df_admet[mask]


def main():
    parser = argparse.ArgumentParser(description="Arch Linux + Mac mini M2 Parkinson's Pipeline Orchestrator")
    parser.add_argument("--target", required=True, choices=["LRRK2", "GBA1"], help="Target Parkinson's protein")
    parser.add_argument("--pdb", required=True, type=validate_pdb_filename,
                        help="Cleaned PDB filename (e.g., 7LHT_clean.pdb or 2V3D_clean.pdb)")
    parser.add_argument("--raw_smiles", default=f"{SHARED_ROOT}/01_ligands_raw/chembl_lrrk2_gba1_library.csv",
                        help="Path to raw SMILES CSV (default: curated ChEMBL LRRK2/GBA1 library)")
    parser.add_argument("--run-id", default=None,
                        help="Unique run identifier (auto-generated as YYYYMMDD_HHMMSS if omitted)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages whose outputs already exist in the run directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the remote commands that would run, without executing them")
    args = parser.parse_args()

    dry_run = args.dry_run

    run_id = args.run_id or now_str()
    raw_smiles = os.path.realpath(args.raw_smiles)
    if not raw_smiles.startswith(os.path.realpath(SHARED_ROOT) + os.sep):
        sys.exit(f"--raw_smiles must live under {SHARED_ROOT}")

    manifest = load_manifest(run_id, args.target)
    manifest["run_id"] = run_id
    manifest["target"] = args.target
    manifest["pdb"] = args.pdb
    manifest["raw_smiles"] = raw_smiles
    manifest.setdefault("started_at", datetime.datetime.now().isoformat())
    save_manifest(run_id, args.target, manifest)

    tr = target_run_dir(run_id, args.target)
    print(f"=== Pipeline run {run_id} | target={args.target} | run_dir={tr} ===")

    # ------------------------------------------------------------------
    # STAGE 1: BBB & CNS Ligand Filtering (Mac mini M2)
    # ------------------------------------------------------------------
    cns_dir = stage_dir(run_id, args.target, "02_cns")
    cns_sdf = os.path.join(cns_dir, f"{args.target}_cns.sdf")
    cns_csv = os.path.join(cns_dir, f"{args.target}_cns.csv")

    if args.resume and stage_done(manifest, "cns_filter") and os.path.isfile(cns_csv):
        print("=== STAGE 1: BBB & CNS Ligand Filtering (skipped, already done) ===")
    elif dry_run and not os.path.isfile(raw_smiles):
        print(f"[dry-run] STAGE 1 skipped: raw SMILES file not found at {raw_smiles}")
        print("[dry-run] build it first with: python scripts/build_chembl_library.py")
    else:
        print("=== STAGE 1: BBB & CNS Ligand Filtering (Mac mini M2) ===")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run([  # noqa: S603 - fixed interpreter and script, validated paths
            sys.executable, os.path.join(script_dir, "cns_filter.py"),
            raw_smiles, cns_sdf, cns_csv
        ], check=True)
        if len(pd.read_csv(cns_csv)) == 0:
            sys.exit("[PIPELINE COMPLETE] No ligands passed the CNS filter.")
        if not dry_run:
            mark_stage_done(manifest, "cns_filter", {"cns_csv": cns_csv, "ligands": len(pd.read_csv(cns_csv))})

    # ------------------------------------------------------------------
    # STAGE 2: Locate Binding Pocket via P2Rank on arch_4070
    # ------------------------------------------------------------------
    pocket_csv = f"{stage_dir(run_id, args.target, '03_pockets')}/{args.pdb}_predictions.csv"
    if args.resume and stage_done(manifest, "pocket_finder") and os.path.isfile(pocket_csv):
        print("=== STAGE 2: Binding Pocket Identification (skipped, already done) ===")
        # Still need coords; re-read from existing CSV.
        df = pd.read_csv(pocket_csv, skipinitialspace=True)
        pocket_idx, cx, cy, cz = select_pocket(df, args.target)
        pdb_path = f"{SHARED_ROOT}/00_targets/{args.target}/{args.pdb}"
    else:
        print("\n=== STAGE 2: Binding Pocket Identification (Arch 4070 Ti Super) ===")
        cx, cy, cz, pdb_path = get_top_pocket_coords(run_id, args.target, args.pdb, dry_run=dry_run)
        if not dry_run:
            mark_stage_done(manifest, "pocket_finder", {
                "pocket_csv": pocket_csv,
                "coords": {"x": cx, "y": cy, "z": cz},
            })

    # ------------------------------------------------------------------
    # STAGE 3: Parallel Dual-Node GPU Execution
    # ------------------------------------------------------------------
    diffdock_dir = stage_dir(run_id, args.target, "04_diffdock")
    gnina_dir = stage_dir(run_id, args.target, "05_gnina")
    diffdock_csv = os.path.join(gnina_dir, f"{args.target}_diffdock_input.csv")
    if dry_run and not os.path.isfile(cns_csv):
        print(f"[dry-run] DiffDock input CSV would be built from {cns_csv} -> {diffdock_csv}")
    else:
        build_diffdock_csv(pdb_path, cns_csv, diffdock_csv)

    diffdock_out = os.path.join(diffdock_dir, args.target)
    gnina_out_sdf = os.path.join(gnina_dir, f"{args.target}_docked.sdf")
    rescored_sdf = os.path.join(gnina_dir, f"{args.target}_diffdock_rescored.sdf")
    combined_sdf = os.path.join(gnina_dir, f"{args.target}_diffdock_top_poses.sdf")

    nodes = get_nodes()
    gpu_4070 = nodes["arch_4070"]["gpu_id"]
    gpu_2080 = nodes["arch_2080"]["gpu_id"]

    diffdock_done = (
        args.resume and stage_done(manifest, "diffdock")
        and os.path.isdir(diffdock_out)
        and any(f.endswith("rank1.sdf") for f in os.listdir(diffdock_out))
    ) if os.path.isdir(diffdock_out) else False
    gnina_dock_done = (
        args.resume and stage_done(manifest, "gnina_dock")
        and os.path.isfile(gnina_out_sdf)
    )

    print("\n=== STAGE 3: Parallel Multi-Node Docking (Arch 4070 Ti Super + Arch 2080 Super) ===")
    cmd_arch4070_diffdock = (
        f"cd {q(DIFFDOCK_DIR)} && "
        f"CUDA_VISIBLE_DEVICES={q(gpu_4070)} {remote_path(MAMBA_BIN)} run -n diffdock python -m inference "
        f"--config default_inference_args.yaml "
        f"--protein_ligand_csv {q(diffdock_csv)} --out_dir {q(diffdock_out)} "
        f"--inference_steps 20 --samples_per_complex 10 --batch_size 6"
    )

    cmd_arch2080_gnina = (
        f"mkdir -p {q(gnina_dir)} && " +
        gnina_docker_cmd(gpu_2080, (
            f"-r {q(pdb_path)} -l {q(cns_csv)} "
            f"--center_x {cx:.2f} --center_y {cy:.2f} --center_z {cz:.2f} "
            f"--size_x 22 --size_y 22 --size_z 22 "
            f"--cnn_scoring rescore --cpu 4 --seed 0 -o {q(gnina_out_sdf)}"
        ), run_id, args.target)
    )

    futures = []
    if not diffdock_done:
        print("[Dock] DiffDock on arch_4070 (not skipped)")
        futures.append(("arch_4070", cmd_arch4070_diffdock, "Arch4070-DiffDock"))
    else:
        print("[Dock] DiffDock on arch_4070 skipped (already done)")

    if not gnina_dock_done:
        print("[Dock] GNINA on arch_2080 (not skipped)")
        futures.append(("arch_2080", cmd_arch2080_gnina, "Arch2080-GNINA"))
    else:
        print("[Dock] GNINA on arch_2080 skipped (already done)")

    if futures:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pool_futures = [
                pool.submit(run_remote_ssh, node, cmd, label, dry_run=dry_run)
                for node, cmd, label in futures
            ]
            for fut in pool_futures:
                fut.result()

    if not dry_run:
        if not diffdock_done:
            mark_stage_done(manifest, "diffdock", {"diffdock_out": diffdock_out})
        if not gnina_dock_done:
            mark_stage_done(manifest, "gnina_dock", {"gnina_out_sdf": gnina_out_sdf})

    # ------------------------------------------------------------------
    # STAGE 3b: GNINA Rescoring of DiffDock Poses
    # ------------------------------------------------------------------
    print("\n=== STAGE 3b: GNINA Rescoring of DiffDock Poses (Arch 2080 Super) ===")
    n_poses = 0 if dry_run else combine_diffdock_poses(diffdock_out, combined_sdf)
    rescore_done = (
        args.resume and stage_done(manifest, "gnina_rescore")
        and os.path.isfile(rescored_sdf)
    )
    if rescore_done:
        print("[Rescore] GNINA rescoring skipped (already done).")
    elif dry_run:
        print(f"[dry-run] would combine DiffDock poses from {diffdock_out} into {combined_sdf}")
        print(f"[dry-run] would then rescore with GNINA on arch_2080 -> {rescored_sdf}")
    elif n_poses:
        run_remote_ssh("arch_2080", gnina_docker_cmd(gpu_2080, (
            f"-r {q(pdb_path)} -l {q(combined_sdf)} --score_only "
            f"--cnn_scoring rescore --cpu 4 -o {q(rescored_sdf)}"
        ), run_id, args.target), "Arch2080-GNINA-Rescore")
        mark_stage_done(manifest, "gnina_rescore", {
            "combined_sdf": combined_sdf,
            "rescored_sdf": rescored_sdf,
            "n_poses": n_poses,
        })
    else:
        print("[Rescore] No DiffDock poses found; continuing with GNINA docking results only.")
        mark_stage_done(manifest, "gnina_rescore", {"n_poses": 0, "skipped": True})

    # ------------------------------------------------------------------
    # STAGE 4: Hit Triage & Neuro-ADMET Validation
    # ------------------------------------------------------------------
    print("\n=== STAGE 4: Hit Triage & Neuro-ADMET Validation (Arch 2080 Super) ===")
    admet_dir = stage_dir(run_id, args.target, "06_admet")
    top_hits_csv = os.path.join(admet_dir, f"{args.target}_top_hits.csv")
    admet_out_csv = os.path.join(admet_dir, f"{args.target}_validated_candidates.csv")
    final_summary = os.path.join(admet_dir, f"{args.target}_FINAL_SAFE_LEADS.csv")

    triage_done = (
        args.resume and stage_done(manifest, "hit_triage")
        and os.path.isfile(top_hits_csv)
    )
    if triage_done:
        print("[Triage] Hit extraction skipped (already done).")
        poses = collect_poses(gnina_out_sdf, "gnina_dock") + collect_poses(rescored_sdf, "diffdock_rescored")
        num_hits = len(pd.read_csv(top_hits_csv)) if os.path.isfile(top_hits_csv) else 0
    elif dry_run:
        print(f"[dry-run] would extract top hits from {gnina_out_sdf} + {rescored_sdf}")
        print(f"[dry-run] top hits CSV would be: {top_hits_csv}")
        print(f"[dry-run] ADMET-AI would run on arch_2080 -> {admet_out_csv}")
        print(f"[dry-run] final safe leads would be: {final_summary}")
        print("[dry-run] stages 2-4 not executed; command surface verified only.")
        return
    else:
        poses = collect_poses(gnina_out_sdf, "gnina_dock") + collect_poses(rescored_sdf, "diffdock_rescored")
        num_hits = extract_top_hits(poses, cns_csv, top_hits_csv)
        mark_stage_done(manifest, "hit_triage", {
            "top_hits_csv": top_hits_csv,
            "n_hits": num_hits,
        })

    admet_done = (
        args.resume and stage_done(manifest, "admet")
        and os.path.isfile(admet_out_csv)
    )
    if num_hits > 0 and not admet_done:
        cmd_arch2080_admet = (
            f"mkdir -p {q(admet_dir)} && "
            f"CUDA_VISIBLE_DEVICES={q(gpu_2080)} {remote_path(MAMBA_BIN)} run -n admet admet_predict "
            f"--data_path {q(top_hits_csv)} --save_path {q(admet_out_csv)} --smiles_column smiles"
        )
        run_remote_ssh("arch_2080", cmd_arch2080_admet, "Arch2080-ADMET-AI")
        mark_stage_done(manifest, "admet", {"admet_out_csv": admet_out_csv})

    if num_hits > 0 and os.path.isfile(admet_out_csv):
        safe_bbb_leads = apply_admet_filters(pd.read_csv(admet_out_csv))
        safe_bbb_leads.to_csv(final_summary, index=False)
        manifest["final_leads_csv"] = final_summary
        manifest["n_final_leads"] = len(safe_bbb_leads)
        manifest["status"] = "completed"
        save_manifest(run_id, args.target, manifest)
        print(f"\n[PIPELINE COMPLETE] Found {len(safe_bbb_leads)} BBB-penetrant, low-predicted-toxicity leads.")
        print(f"Results saved to: {final_summary}")
        print("Note: these are in-silico predictions only; they require experimental validation.")
    elif num_hits == 0:
        manifest["status"] = "completed_no_hits"
        save_manifest(run_id, args.target, manifest)
        print("\n[PIPELINE COMPLETE] No ligands exceeded the CNNscore threshold in this batch.")
    else:
        manifest["status"] = "completed_partial"
        save_manifest(run_id, args.target, manifest)
        print("\n[PIPELINE COMPLETE] Pipeline finished but ADMET output was not found.")


if __name__ == "__main__":
    main()
