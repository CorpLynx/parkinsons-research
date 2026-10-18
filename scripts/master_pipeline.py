#!/usr/bin/env python3
"""
master_pipeline.py - Master Orchestrator (Runs on Mac mini M2)
Coordinates ligand filtering (M2), P2Rank + DiffDock (Arch 4070 Ti Super),
Dockerized 3D CNN GNINA screening (Arch 2080 Super), and ADMET-AI triage (Arch 2080 Super).
"""
import os
import sys
import argparse
import subprocess
import threading
import pandas as pd
import paramiko
from rdkit import Chem

# ================= ARCH CLUSTER CONFIGURATION =================
SHARED_ROOT = "/mnt/pd_research"
MAMBA_BIN = "$HOME/.local/bin/micromamba"
P2RANK_BIN = f"{SHARED_ROOT}/tools/p2rank_2.4.2/prank"
DIFFDOCK_DIR = f"{SHARED_ROOT}/tools/DiffDock"

# Update host IPs and usernames to match your Arch Linux machines.
# If both GPUs are in the SAME Arch machine, use the same host IP and set gpu_id to "0" and "1".
NODES = {
    "arch_4070": {
        "host": "192.168.1.100",
        "user": "archuser",
        "gpu_id": "0"
    },
    "arch_2080": {
        "host": "192.168.1.101",
        "user": "archuser",
        "gpu_id": "0"
    }
}
# ==============================================================

def run_remote_ssh(node_key, command, stage_label):
    """Executes a command on the specified Arch Linux GPU node over SSH."""
    cfg = NODES[node_key]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=cfg["host"], username=cfg["user"])
    print(f"[{stage_label}] Connected to {node_key} ({cfg['host']})...")
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
    for line in iter(stdout.readline, ""):
        sys.stdout.write(f"  [{stage_label}] {line}")
    exit_status = stdout.channel.recv_exit_status()
    client.close()
    if exit_status != 0:
        raise RuntimeError(f"[{stage_label}] Failed on {node_key} with exit code {exit_status}")
    print(f"[{stage_label}] Completed successfully.")

def get_top_pocket_coords(target_name, pdb_filename):
    """Runs P2Rank on arch_4070 and extracts the predicted pocket center (X, Y, Z)."""
    pdb_path = f"{SHARED_ROOT}/00_targets/{target_name}/{pdb_filename}"
    out_dir = f"{SHARED_ROOT}/03_pockets/{target_name}"
    cmd = f"{P2RANK_BIN} predict -f {pdb_path} -o {out_dir}"
    run_remote_ssh("arch_4070", cmd, "P2Rank-PocketFinder")

    predictions_csv = f"{out_dir}/{pdb_filename}_predictions.csv"
    df = pd.read_csv(predictions_csv, skipinitialspace=True)
    # Select Rank 1 catalytic pocket for LRRK2, or Rank 2 allosteric chaperone pocket for GBA1
    pocket_idx = 0 if target_name.upper() == "LRRK2" else (1 if len(df) > 1 else 0)
    row = df.iloc[pocket_idx]
    cx, cy, cz = float(row["center_x"]), float(row["center_y"]), float(row["center_z"])
    print(f"[Pocket Coords] Selected Pocket #{pocket_idx+1} for {target_name}: X={cx:.2f}, Y={cy:.2f}, Z={cz:.2f}")
    return cx, cy, cz, pdb_path

def build_diffdock_csv(pdb_path, filtered_csv, diffdock_csv, max_ligands=250):
    """Creates the batch input CSV required by DiffDock."""
    df = pd.read_csv(filtered_csv).head(max_ligands)
    records = []
    for _, row in df.iterrows():
        records.append({
            "complex_name": f"complex_{row['ligand_id']}",
            "protein_path": pdb_path,
            "ligand_description": row["smiles"],
            "protein_sequence": ""
        })
    pd.DataFrame(records).to_csv(diffdock_csv, index=False)

def extract_top_gnina_hits(docked_sdf, top_hits_csv, min_cnn_score=0.65):
    """Parses GNINA output SDF on Mac mini M2 and extracts top binders for ADMET-AI."""
    suppl = Chem.SDMolSupplier(docked_sdf)
    hits = []
    for mol in suppl:
        if mol is None:
            continue
        cnn_score = float(mol.GetProp("CNNscore")) if mol.HasProp("CNNscore") else 0.0
        cnn_aff = float(mol.GetProp("CNNaffinity")) if mol.HasProp("CNNaffinity") else 0.0
        if cnn_score >= min_cnn_score:
            hits.append({
                "ligand_id": mol.GetProp("_Name"),
                "smiles": mol.GetProp("SMILES") if mol.HasProp("SMILES") else Chem.MolToSmiles(mol),
                "CNNscore": cnn_score,
                "CNNaffinity": cnn_aff
            })
    df_hits = pd.DataFrame(hits).sort_values(by="CNNscore", ascending=False)
    df_hits.to_csv(top_hits_csv, index=False)
    print(f"[Hit Triage] Extracted {len(df_hits)} high-confidence binders (CNNscore >= {min_cnn_score}).")
    return len(df_hits)

def main():
    parser = argparse.ArgumentParser(description="Arch Linux + Mac mini M2 Parkinson's Pipeline Orchestrator")
    parser.add_argument("--target", required=True, choices=["LRRK2", "GBA1"], help="Target Parkinson's protein")
    parser.add_argument("--pdb", required=True, help="Cleaned PDB filename (e.g., 7LHT_clean.pdb or 6N3F_clean.pdb)")
    parser.add_argument("--raw_smiles", default=f"{SHARED_ROOT}/01_ligands_raw/cns_starter.csv", help="Path to raw SMILES CSV")
    args = parser.parse_args()

    # Stage 1: Filter SMILES for Blood-Brain Barrier on Mac mini M2
    cns_sdf = f"{SHARED_ROOT}/02_ligands_cns_filtered/{args.target}_cns.sdf"
    cns_csv = f"{SHARED_ROOT}/02_ligands_cns_filtered/{args.target}_cns.csv"
    print("=== STAGE 1: BBB & CNS Ligand Filtering (Mac mini M2) ===")
    subprocess.run([
        sys.executable, f"{SHARED_ROOT}/scripts/cns_filter.py",
        args.raw_smiles, cns_sdf, cns_csv
    ], check=True)

    # Stage 2: Locate Binding Pocket via P2Rank on arch_4070
    print("\n=== STAGE 2: Binding Pocket Identification (Arch 4070 Ti Super) ===")
    cx, cy, cz, pdb_path = get_top_pocket_coords(args.target, args.pdb)

    diffdock_csv = f"{SHARED_ROOT}/scripts/{args.target}_diffdock_input.csv"
    build_diffdock_csv(pdb_path, cns_csv, diffdock_csv)

    # Stage 3: Parallel Dual-Node GPU Execution (Arch 4070 Ti Super + Arch 2080 Super)
    print("\n=== STAGE 3: Parallel Multi-Node Docking (Arch 4070 Ti Super + Arch 2080 Super) ===")
    diffdock_out = f"{SHARED_ROOT}/04_diffdock_poses/{args.target}"
    gnina_out_sdf = f"{SHARED_ROOT}/05_gnina_rescored/{args.target}_docked.sdf"

    gpu_4070 = NODES["arch_4070"]["gpu_id"]
    gpu_2080 = NODES["arch_2080"]["gpu_id"]

    cmd_arch4070_diffdock = (
        f"cd {DIFFDOCK_DIR} && "
        f"CUDA_VISIBLE_DEVICES={gpu_4070} {MAMBA_BIN} run -n diffdock python -m inference "
        f"--protein_ligand_csv {diffdock_csv} --out_dir {diffdock_out} "
        f"--inference_steps 20 --samples_per_complex 10 --batch_size 6"
    )

    cmd_arch2080_gnina = (
        f"mkdir -p {SHARED_ROOT}/05_gnina_rescored && "
        f"docker run --rm --gpus '\"device={gpu_2080}\"' "
        f"-v {SHARED_ROOT}:{SHARED_ROOT} gnina/gnina:latest "
        f"gnina -r {pdb_path} -l {cns_sdf} "
        f"--center_x {cx:.2f} --center_y {cy:.2f} --center_z {cz:.2f} "
        f"--size_x 22 --size_y 22 --size_z 22 "
        f"--cnn_scoring rescore --cpu 4 -o {gnina_out_sdf}"
    )

    t0 = threading.Thread(target=run_remote_ssh, args=("arch_4070", cmd_arch4070_diffdock, "Arch4070-DiffDock"))
    t1 = threading.Thread(target=run_remote_ssh, args=("arch_2080", cmd_arch2080_gnina, "Arch2080-GNINA"))

    t0.start()
    t1.start()
    t0.join()
    t1.join()

    # Stage 4: Extract Top Hits on Mac mini M2 & Run ADMET-AI on Arch 2080 Super
    print("\n=== STAGE 4: Hit Triage & Neuro-ADMET Validation (Arch 2080 Super) ===")
    top_hits_csv = f"{SHARED_ROOT}/05_gnina_rescored/{args.target}_top_hits.csv"
    admet_out_csv = f"{SHARED_ROOT}/06_final_hits_admet/{args.target}_validated_candidates.csv"

    num_hits = extract_top_gnina_hits(gnina_out_sdf, top_hits_csv, min_cnn_score=0.65)
    if num_hits > 0:
        cmd_arch2080_admet = (
            f"mkdir -p {SHARED_ROOT}/06_final_hits_admet && "
            f"CUDA_VISIBLE_DEVICES={gpu_2080} {MAMBA_BIN} run -n admet admet_predict "
            f"--data_path {top_hits_csv} --save_path {admet_out_csv} --smiles_column smiles"
        )
        run_remote_ssh("arch_2080", cmd_arch2080_admet, "Arch2080-ADMET-AI")

        df_admet = pd.read_csv(admet_out_csv)
        safe_bbb_leads = df_admet[
            (df_admet["BBB_Martins"] >= 0.80) &
            (df_admet["hERG"] <= 0.30) &
            (df_admet["DILI"] <= 0.40)
        ]
        final_summary = f"{SHARED_ROOT}/06_final_hits_admet/{args.target}_FINAL_SAFE_LEADS.csv"
        safe_bbb_leads.to_csv(final_summary, index=False)
        print(f"\n[PIPELINE COMPLETE] Found {len(safe_bbb_leads)} BBB-penetrant, non-toxic leads!")
        print(f"Results saved to: {final_summary}")
    else:
        print("\n[PIPELINE COMPLETE] No ligands exceeded the CNNscore threshold in this batch.")

if __name__ == "__main__":
    main()
