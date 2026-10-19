#!/usr/bin/env python3
"""
master_pipeline.py - Master Orchestrator (Runs on Mac mini M2)
Coordinates ligand filtering (M2), P2Rank + DiffDock (Arch 4070 Ti Super),
Dockerized 3D CNN GNINA screening and DiffDock pose rescoring (Arch 2080 Super),
and ADMET-AI triage (Arch 2080 Super).

Node addresses/users come from environment variables (see .env.example).
Remote hosts must already be in ~/.ssh/known_hosts; unknown hosts are rejected.
"""
import argparse
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


def run_remote_ssh(node_key, command, stage_label):
    """Executes a command on the specified Arch Linux GPU node over SSH (key auth, verified host)."""
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


def get_top_pocket_coords(target_name, pdb_filename, pocket_index=None):
    """Runs P2Rank on arch_4070 and extracts the predicted pocket center (X, Y, Z)."""
    pdb_path = f"{SHARED_ROOT}/00_targets/{target_name}/{pdb_filename}"
    out_dir = f"{SHARED_ROOT}/03_pockets/{target_name}"
    cmd = f"{q(P2RANK_BIN)} predict -f {q(pdb_path)} -o {q(out_dir)}"
    run_remote_ssh("arch_4070", cmd, "P2Rank-PocketFinder")

    predictions_csv = f"{out_dir}/{pdb_filename}_predictions.csv"
    df = pd.read_csv(predictions_csv, skipinitialspace=True)
    if df.empty:
        raise RuntimeError(f"P2Rank found no pockets for {pdb_filename}")
    # Default: rank 1 pocket for LRRK2, rank 2 for GBA1. The GBA1 choice is an unverified
    # assumption -- override with --pocket-index after inspecting P2Rank output / the structure.
    if pocket_index is None:
        pocket_idx = 0 if target_name.upper() == "LRRK2" else (1 if len(df) > 1 else 0)
    else:
        pocket_idx = min(max(pocket_index, 0), len(df) - 1)
    row = df.iloc[pocket_idx]
    cx, cy, cz = float(row["center_x"]), float(row["center_y"]), float(row["center_z"])
    print(f"[Pocket Coords] Selected Pocket #{pocket_idx+1} for {target_name}: X={cx:.2f}, Y={cy:.2f}, Z={cz:.2f}")
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


def gnina_docker_cmd(gpu_id, args, out_dir=None):
    """Hardened GNINA container invocation: no extra privileges, non-root, read-only inputs."""
    out_dir = out_dir or SHARED_ROOT + "/05_gnina_rescored"
    return (
        f"docker run --rm --gpus {q(chr(34) + 'device=' + gpu_id + chr(34))} "
        f"--cap-drop ALL --security-opt no-new-privileges "
        f"--user \"$(id -u):$(id -g)\" "
        f"-v {q(SHARED_ROOT)}:{q(SHARED_ROOT)}:ro "
        f"-v {q(out_dir)}:{q(out_dir)} "
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
                        help="Cleaned PDB filename (e.g., 7LHT_clean.pdb or 6N3F_clean.pdb)")
    parser.add_argument("--raw_smiles", default=f"{SHARED_ROOT}/01_ligands_raw/cns_starter.csv",
                        help="Path to raw SMILES CSV")
    parser.add_argument("--pocket-index", type=int, default=None,
                        help="0-based P2Rank pocket to dock into (default: 0 for LRRK2, 1 for GBA1)")
    args = parser.parse_args()

    raw_smiles = os.path.realpath(args.raw_smiles)
    if not raw_smiles.startswith(os.path.realpath(SHARED_ROOT) + os.sep):
        sys.exit(f"--raw_smiles must live under {SHARED_ROOT}")

    # Stage 1: Filter SMILES for Blood-Brain Barrier on Mac mini M2
    cns_sdf = f"{SHARED_ROOT}/02_ligands_cns_filtered/{args.target}_cns.sdf"
    cns_csv = f"{SHARED_ROOT}/02_ligands_cns_filtered/{args.target}_cns.csv"
    print("=== STAGE 1: BBB & CNS Ligand Filtering (Mac mini M2) ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([  # noqa: S603 - fixed interpreter and script, validated paths
        sys.executable, os.path.join(script_dir, "cns_filter.py"),
        raw_smiles, cns_sdf, cns_csv
    ], check=True)
    if len(pd.read_csv(cns_csv)) == 0:
        sys.exit("[PIPELINE COMPLETE] No ligands passed the CNS filter.")

    # Stage 2: Locate Binding Pocket via P2Rank on arch_4070
    print("\n=== STAGE 2: Binding Pocket Identification (Arch 4070 Ti Super) ===")
    cx, cy, cz, pdb_path = get_top_pocket_coords(args.target, args.pdb, args.pocket_index)

    diffdock_csv = f"{SHARED_ROOT}/scripts/{args.target}_diffdock_input.csv"
    build_diffdock_csv(pdb_path, cns_csv, diffdock_csv)

    # Stage 3: Parallel Dual-Node GPU Execution (Arch 4070 Ti Super + Arch 2080 Super)
    print("\n=== STAGE 3: Parallel Multi-Node Docking (Arch 4070 Ti Super + Arch 2080 Super) ===")
    diffdock_out = f"{SHARED_ROOT}/04_diffdock_poses/{args.target}"
    gnina_out_sdf = f"{SHARED_ROOT}/05_gnina_rescored/{args.target}_docked.sdf"
    rescored_sdf = f"{SHARED_ROOT}/05_gnina_rescored/{args.target}_diffdock_rescored.sdf"
    combined_sdf = f"{SHARED_ROOT}/05_gnina_rescored/{args.target}_diffdock_top_poses.sdf"

    nodes = get_nodes()
    gpu_4070 = nodes["arch_4070"]["gpu_id"]
    gpu_2080 = nodes["arch_2080"]["gpu_id"]

    cmd_arch4070_diffdock = (
        f"cd {q(DIFFDOCK_DIR)} && "
        f"CUDA_VISIBLE_DEVICES={q(gpu_4070)} {remote_path(MAMBA_BIN)} run -n diffdock python -m inference "
        f"--config default_inference_args.yaml "
        f"--protein_ligand_csv {q(diffdock_csv)} --out_dir {q(diffdock_out)} "
        f"--inference_steps 20 --samples_per_complex 10 --batch_size 6"
    )

    cmd_arch2080_gnina = (
        f"mkdir -p {q(SHARED_ROOT + '/05_gnina_rescored')} && " +
        gnina_docker_cmd(gpu_2080, (
            f"-r {q(pdb_path)} -l {q(cns_sdf)} "
            f"--center_x {cx:.2f} --center_y {cy:.2f} --center_z {cz:.2f} "
            f"--size_x 22 --size_y 22 --size_z 22 "
            f"--cnn_scoring rescore --cpu 4 --seed 0 -o {q(gnina_out_sdf)}"
        ))
    )

    # Failures in either node abort the pipeline instead of being silently dropped.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_remote_ssh, "arch_4070", cmd_arch4070_diffdock, "Arch4070-DiffDock"),
            pool.submit(run_remote_ssh, "arch_2080", cmd_arch2080_gnina, "Arch2080-GNINA"),
        ]
        for fut in futures:
            fut.result()

    # Stage 3b: rescore DiffDock poses with GNINA so both docking engines feed the triage.
    print("\n=== STAGE 3b: GNINA Rescoring of DiffDock Poses (Arch 2080 Super) ===")
    n_poses = combine_diffdock_poses(diffdock_out, combined_sdf)
    if n_poses:
        run_remote_ssh("arch_2080", gnina_docker_cmd(gpu_2080, (
            f"-r {q(pdb_path)} -l {q(combined_sdf)} --score_only "
            f"--cnn_scoring rescore --cpu 4 -o {q(rescored_sdf)}"
        )), "Arch2080-GNINA-Rescore")
    else:
        print("[Rescore] No DiffDock poses found; continuing with GNINA docking results only.")

    # Stage 4: Extract Top Hits on Mac mini M2 & Run ADMET-AI on Arch 2080 Super
    print("\n=== STAGE 4: Hit Triage & Neuro-ADMET Validation (Arch 2080 Super) ===")
    top_hits_csv = f"{SHARED_ROOT}/05_gnina_rescored/{args.target}_top_hits.csv"
    admet_out_csv = f"{SHARED_ROOT}/06_final_hits_admet/{args.target}_validated_candidates.csv"

    poses = collect_poses(gnina_out_sdf, "gnina_dock") + collect_poses(rescored_sdf, "diffdock_rescored")
    num_hits = extract_top_hits(poses, cns_csv, top_hits_csv)
    if num_hits > 0:
        cmd_arch2080_admet = (
            f"mkdir -p {q(SHARED_ROOT + '/06_final_hits_admet')} && "
            f"CUDA_VISIBLE_DEVICES={q(gpu_2080)} {remote_path(MAMBA_BIN)} run -n admet admet_predict "
            f"--data_path {q(top_hits_csv)} --save_path {q(admet_out_csv)} --smiles_column smiles"
        )
        run_remote_ssh("arch_2080", cmd_arch2080_admet, "Arch2080-ADMET-AI")

        safe_bbb_leads = apply_admet_filters(pd.read_csv(admet_out_csv))
        final_summary = f"{SHARED_ROOT}/06_final_hits_admet/{args.target}_FINAL_SAFE_LEADS.csv"
        safe_bbb_leads.to_csv(final_summary, index=False)
        print(f"\n[PIPELINE COMPLETE] Found {len(safe_bbb_leads)} BBB-penetrant, low-predicted-toxicity leads.")
        print(f"Results saved to: {final_summary}")
        print("Note: these are in-silico predictions only; they require experimental validation.")
    else:
        print("\n[PIPELINE COMPLETE] No ligands exceeded the CNNscore threshold in this batch.")


if __name__ == "__main__":
    main()
