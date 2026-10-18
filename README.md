# Distributed Local AI Pipeline for Parkinson's Drug Discovery (Arch Linux + macOS Edition)

**Hardware Specification, Arch Linux Environment Configuration, Target Preparation & Automated Orchestration Guide**

This document outlines the complete system requirements, Arch Linux kernel/driver configuration, containerized and isolated CUDA dependencies, target preparation protocols, and automated orchestration scripts to transform a **Mac mini M2**, an **NVIDIA RTX 4070 Ti Super (16GB)**, and an **NVIDIA RTX 2080 Super (8GB)** into an integrated computational biology and drug-screening cluster targeting Parkinson's Disease (PD).

---

## 1. Cluster Architecture & Hardware Requirements

To maximize throughput without hitting VRAM bottlenecks or Arch Linux rolling-release library conflicts, the pipeline divides responsibilities across two compute paradigms: **Apple Silicon (`arm64`)** for orchestration, cheminformatics, and literature mining, and **Arch Linux (`x86_64` + NVIDIA CUDA)** for 3D structural folding, diffusion docking, and neural network scoring.

### 1.1 Hardware Role Matrix

| Node Identifier | Hardware | VRAM / Memory | Operating System | Primary Responsibilities | Assigned AI Models & Tools |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mac-mini`** *(Orchestrator)* | Mac mini M2 | 8GB–24GB Unified Memory | macOS Sonoma / Sequoia | Pipeline orchestration, CNS ligand curation, Blood-Brain Barrier (BBB) filtering, target protonation, literature mining | `RDKit`, `OpenMM`, `PDBFixer`, `PDB2PQR`, `Ollama` (`llama3.1:8b`), `Paramiko` |
| **`arch-4070`** *(Primary Compute)* | RTX 4070 Ti Super (Ada, NV190) | 16GB GDDR6X | Arch Linux | Protein structure prediction, mutant modeling, pocket detection, blind diffusion docking | `LocalColabFold` (AlphaFold2), `ESMFold`, `P2Rank`, `DiffDock-L` |
| **`arch-2080`** *(Secondary Screening)* | RTX 2080 Super (Turing, NV160) | 8GB GDDR6 | Arch Linux | High-throughput 3D CNN pocket docking/rescoring, de novo ligand generation, toxicity & BBB triage | `GNINA` (Dockerized), `REINVENT4`, `ADMET-AI` |

### 1.2 Physical & Network Prerequisites

1. **System RAM:**
   * **`arch-4070`:** Minimum 32GB DDR4/DDR5 (64GB recommended for LocalColabFold MSA generation).
   * **`arch-2080`:** Minimum 16GB–32GB DDR4/DDR5.
   * **`mac-mini`:** Minimum 8GB Unified Memory (16GB+ recommended).
2. **Storage (NVMe SSDs Required):**
   * Molecular docking reads and writes hundreds of thousands of small `.sdf`, `.pdb`, and `.csv` files.
   * Provision at least **1TB NVMe SSD** on `arch-4070` to host the shared NFS workspace (`/srv/nfs/pd_research`).
3. **Power Supply (PSU):**
   * If `arch-4070` (285W TDP) and `arch-2080` (250W TDP) are in separate cases, a 650W–750W Gold PSU per machine is sufficient.
   * If both GPUs are installed in a **single** Arch Linux host, a **1000W ATX 3.0 Gold/Platinum PSU** is required to handle combined transient spikes.
4. **Local Area Network (LAN):**
   * Connect all three machines via wired **1GbE or 2.5GbE Ethernet** to avoid Wi-Fi packet drop during NFS file transfers and SSH streaming.

---

## 2. Arch Linux Base Configuration (Run on Both GPU Nodes)

### Why Arch Linux Requires a Specific CUDA Strategy
Arch Linux's official `cuda` package in `[extra]` tracks the bleeding edge (installing to `/opt/cuda` alongside the latest `gcc` and `glibc`). However, scientific AI packages like `torch-scatter`, `torch-sparse`, `openfold`, and `gnina` rely on precompiled wheels built against older CUDA 12.1 / 12.4 toolkits and specific `libstdc++.so.6` symbols.

To guarantee zero compilation or symbol errors on Arch:
1. Install **only** the NVIDIA kernel drivers (`nvidia-open-dkms`), `nvidia-utils`, and `nvidia-container-toolkit` via `pacman`.
2. Use **`micromamba`** to manage isolated `cudatoolkit` and `pytorch-cuda` libraries in user space.
3. Run **`GNINA`** inside its official GPU-accelerated Docker container so Arch's rolling `boost-libs` never break it.

### 2.1 Install Kernel Drivers, Docker, and System Utilities

Run the following on both `arch-4070` and `arch-2080` (both Turing NV160 and Ada Lovelace NV190 support `nvidia-open-dkms`):

```bash
# 1. Full system sync and base package installation
sudo pacman -Syu --needed base-devel git wget curl aria2 cmake openssh nfs-utils \
  linux-headers nvidia-open-dkms nvidia-utils opencl-nvidia ocl-icd \
  docker nvidia-container-toolkit nvtop btop jdk17-openjdk

# 2. Enable kernel modesetting and services
sudo systemctl enable --now sshd docker nvidia-persistenced

# 3. Configure Docker to recognize NVIDIA GPUs
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker $USER
```

**Reboot both Arch machines**, then verify GPU access natively and inside Docker:

```bash
# Verify native driver
nvidia-smi

# Verify Docker GPU passthrough
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 2.2 Install `micromamba` on Both Arch Nodes

`micromamba` is a fast, statically-linked C++ reimplementation of Conda/Mamba that does not conflict with Arch's system `/usr/bin/python`:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc

# Configure strict conda-forge priority
micromamba config append channels pytorch
micromamba config append channels nvidia
micromamba config append channels conda-forge
micromamba config set channel_priority flexible
```

---

## 3. Shared NFS Workspace & SSH Mesh Setup

To allow the Mac mini M2 and both Arch Linux GPUs to read and write to the exact same file paths without manual `scp` transfers, configure an NFSv4 share exported by `arch-4070` and mounted at `/mnt/pd_research` on all nodes.

### 3.1 Configure NFS Server on `arch-4070`

```bash
# Create the physical backing directory on your NVMe drive
sudo mkdir -p /srv/nfs/pd_research
sudo chown -R $USER:$USER /srv/nfs/pd_research

# Create a local symlink so /mnt/pd_research works identically on arch-4070
sudo mkdir -p /mnt
sudo ln -s /srv/nfs/pd_research /mnt/pd_research

# Export to your local subnet (replace 192.168.1.0/24 with your LAN subnet if different)
echo "/srv/nfs/pd_research 192.168.1.0/24(rw,sync,no_subtree_check,insecure,all_squash,anonuid=$(id -u),anongid=$(id -g))" | sudo tee -a /etc/exports

sudo systemctl enable --now nfs-server
sudo exportfs -arv
```

### 3.2 Mount Shared Workspace on `arch-2080`

```bash
sudo mkdir -p /mnt/pd_research
# Replace 192.168.1.100 with the LAN IP of arch-4070
sudo mount -t nfs 192.168.1.100:/srv/nfs/pd_research /mnt/pd_research

# Optional: Add to /etc/fstab for automatic mounting on boot
echo "192.168.1.100:/srv/nfs/pd_research /mnt/pd_research nfs defaults,_netdev 0 0" | sudo tee -a /etc/fstab
```

### 3.3 Mount Shared Workspace & SSH Keys on `mac-mini` (macOS)

On the **Mac mini M2**, generate an SSH keypair, copy it to both Arch nodes, and mount the NFS share:

```bash
# 1. Generate SSH key and copy to both Arch nodes (replace IPs/usernames as needed)
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id archuser@192.168.1.100  # arch-4070
ssh-copy-id archuser@192.168.1.101  # arch-2080

# 2. Mount the NFS share on macOS
sudo mkdir -p /Volumes/pd_research
sudo mount -t nfs -o resvport,rw 192.168.1.100:/srv/nfs/pd_research /Volumes/pd_research

# Note: On macOS, create a synthetic.conf entry or use /Volumes/pd_research.
# To make /mnt/pd_research work identically across macOS and Arch Linux:
echo -e "mnt\tSystem/Volumes/Data/Volumes" | sudo tee -a /etc/synthetic.conf
# (Reboot macOS once for /etc/synthetic.conf to create /mnt/pd_research -> /Volumes/pd_research)
```

### 3.4 Initialize the Unified Directory Tree

Run once from any machine:

```bash
mkdir -p /mnt/pd_research/{00_targets,01_ligands_raw,02_ligands_cns_filtered,03_pockets,04_diffdock_poses,05_gnina_rescored,06_final_hits_admet,scripts,tools}
```

---

## 4. Installing AI Models by Node

### 4.1 Node 1: `arch-4070` (RTX 4070 Ti Super — 16GB VRAM)

Run the following installations on `arch-4070`.

#### A. P2Rank (Machine Learning Binding Pocket Detection)
Uses the `jdk17-openjdk` package installed earlier via `pacman`:

```bash
cd /mnt/pd_research/tools
wget https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz
tar -zxvf p2rank_2.4.2.tar.gz
rm p2rank_2.4.2.tar.gz
```

#### B. DiffDock (Generative Diffusion Blind Docking)
```bash
cd /mnt/pd_research/tools
git clone https://github.com/gcorso/DiffDock.git
cd DiffDock

# Create isolated Python 3.10 + PyTorch 2.2.2 + CUDA 12.1 environment
micromamba create -n diffdock python=3.10 pytorch=2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge -y
micromamba activate diffdock

# Install prebuilt PyTorch Geometric CUDA wheels matching PyTorch 2.2.2 + cu121
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install torch-geometric e3nn biopython rdkit-pypi spyrmsd pandas scipy pyyaml fair-esm

micromamba deactivate
```

#### C. LocalColabFold (AlphaFold2 Mutant Structure Prediction)
```bash
mkdir -p ~/localcolabfold && cd ~/localcolabfold
curl -L https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh -O
chmod +x install_colabbatch_linux.sh
./install_colabbatch_linux.sh
```

---

### 4.2 Node 2: `arch-2080` (RTX 2080 Super — 8GB VRAM)

Run the following installations on `arch-2080`.

#### A. GNINA (Dockerized 3D CNN Docking & Affinity Rescoring)
Pulling the official container avoids Arch `glibc` / `boost` library incompatibilities while running at native speed on the 2080 Super:

```bash
docker pull gnina/gnina:latest

# Test GNINA container against the shared NFS mount
docker run --rm --gpus all -v /mnt/pd_research:/mnt/pd_research gnina/gnina:latest gnina --version
```

#### B. ADMET-AI (Graph Neural Network BBB & Neuro-Toxicity Triage)
```bash
micromamba create -n admet python=3.10 pytorch=2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge -y
micromamba activate admet
pip install admet-ai rdkit pandas
micromamba deactivate
```

#### C. REINVENT4 (De Novo Small Molecule Generation)
```bash
cd /mnt/pd_research/tools
git clone https://github.com/MolecularAI/REINVENT4.git
cd REINVENT4
micromamba create -n reinvent4 python=3.10 pytorch=2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge -y
micromamba activate reinvent4
pip install -e .
micromamba deactivate
```

---

### 4.3 Node 3: `mac-mini` (Mac mini M2 — Control & Cheminformatics)

Run the following on your **Mac mini M2**:

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Micromamba and Ollama for local literature mining
brew install micromamba ollama

# 3. Initialize Micromamba and create the Apple Silicon cheminformatics environment
micromamba shell init -s zsh -r ~/micromamba
source ~/.zshrc

micromamba create -n chem_m2 python=3.11 rdkit openbabel openmm pdbfixer biopython pandas numpy scipy tqdm requests paramiko -c conda-forge -y
micromamba activate chem_m2
pip install meeko pubchempy chembl_webresource_client
```

---

## 5. Automated Target & CNS Ligand Preparation Scripts

### 5.1 Target Preparation Script (`prep_targets.py`)

Save this script to `/mnt/pd_research/scripts/prep_targets.py` and run it on the **Mac mini M2**. It downloads, strips crystallographic artifacts/waters, rebuilds missing side-chain atoms via `PDBFixer`, and protonates two major Parkinson's targets at their physiological $\text{pH}$ values:
* **LRRK2 Kinase (`7LHT`):** Protonated at cytosolic $\text{pH } 7.4$.
* **GBA1 Glucocerebrosidase (`6N3F`):** Protonated at acidic lysosomal $\text{pH } 5.5$.

```python
#!/usr/bin/env python3
"""
prep_targets.py - Runs on Mac mini M2 (chem_m2 environment)
Downloads LRRK2 (7LHT) and GBA1 (6N3F), strips waters/heterogens, fixes missing atoms,
and protonates at physiological pH (7.4 for cytosolic LRRK2, 5.5 for lysosomal GBA1).
"""
import os
import urllib.request
from pdbfixer import PDBFixer
from openmm.app import PDBFile

SHARED_ROOT = "/mnt/pd_research"

TARGETS = {
    "LRRK2": {
        "pdb_id": "7LHT",
        "ph": 7.4,
        "dir": f"{SHARED_ROOT}/00_targets/LRRK2"
    },
    "GBA1": {
        "pdb_id": "6N3F",
        "ph": 5.5,
        "dir": f"{SHARED_ROOT}/00_targets/GBA1"
    }
}

def prepare_targets():
    for name, cfg in TARGETS.items():
        os.makedirs(cfg["dir"], exist_ok=True)
        raw_pdb = os.path.join(cfg["dir"], f"{cfg['pdb_id']}_raw.pdb")
        clean_pdb = os.path.join(cfg["dir"], f"{cfg['pdb_id']}_clean.pdb")

        print(f"[{name}] Downloading PDB {cfg['pdb_id']} from RCSB...")
        url = f"https://files.rcsb.org/download/{cfg['pdb_id']}.pdb"
        urllib.request.urlretrieve(url, raw_pdb)

        print(f"[{name}] Cleaning heterogens, rebuilding missing atoms, and protonating at pH {cfg['ph']}...")
        fixer = PDBFixer(filename=raw_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(cfg["ph"])

        with open(clean_pdb, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        print(f"[{name}] Ready for docking at: {clean_pdb}\n")

if __name__ == "__main__":
    prepare_targets()
```

### 5.2 Blood-Brain Barrier (CNS) Ligand Filter (`cns_filter.py`)

Save this script to `/mnt/pd_research/scripts/cns_filter.py`. It runs on the **Mac mini M2** to filter raw SMILES strings (or generate a starter benchmark library if no input CSV is provided) using strict Central Nervous System (CNS) Blood-Brain Barrier rules:
* Molecular Weight ($\text{MW} \le 450\text{ Da}$)
* Topological Polar Surface Area ($\text{TPSA} < 90\text{ \AA}^2$)
* Lipophilicity ($1.0 \le \log P \le 4.5$)
* Hydrogen Bond Donors ($\text{HBD} \le 3$) and Acceptors ($\text{HBA} \le 7$)
* Rotatable Bonds ($\le 8$)

```python
#!/usr/bin/env python3
"""
cns_filter.py - Runs on Mac mini M2 (chem_m2 environment)
Filters SMILES strings for Blood-Brain Barrier permeability and embeds 3D conformers.
"""
import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

STARTER_CNS_LIBRARY = [
    {"ligand_id": "DNL201_LRRK2_Inhibitor", "smiles": "CC1=CC(=NN1C)NC2=NC=C(C(=N2)NC3=CC=C(C=C3)N4CCOCC4)C(F)(F)F"},
    {"ligand_id": "MLi2_LRRK2_Inhibitor", "smiles": "CC(C)OC1=CC(=NC(=N1)N2CCC(CC2)O)N3C=C(C4=CC=CC=C43)C#N"},
    {"ligand_id": "GNE7915_LRRK2_Inhibitor", "smiles": "CC1=CC(=C(C=C1NC2=NC=C(C(=N2)NC3=CC=CC=C3OC)C(F)(F)F)F)N4CCOCC4"},
    {"ligand_id": "Ambroxol_GBA1_Chaperone", "smiles": "C1CC(CCC1NCC2=CC(=C(C(=C2)Br)N)Br)O"},
    {"ligand_id": "LTI291_GBA1_Activator", "smiles": "CC1=CC=C(C=C1)S(=O)(=O)N2CCN(CC2)C(=O)C3=CC=CC=C3F"}
]

def passes_bbb_rules(mol):
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

    return (
        mw <= 450.0 and
        1.0 <= logp <= 4.5 and
        tpsa < 90.0 and
        hbd <= 3 and
        hba <= 7 and
        rot_bonds <= 8
    )

def process_library(input_csv, output_sdf, output_csv):
    if not os.path.exists(input_csv):
        print(f"[Mac mini M2] {input_csv} not found. Generating starter CNS benchmark CSV...")
        os.makedirs(os.path.dirname(input_csv), exist_ok=True)
        pd.DataFrame(STARTER_CNS_LIBRARY).to_csv(input_csv, index=False)

    df = pd.read_csv(input_csv)
    os.makedirs(os.path.dirname(output_sdf), exist_ok=True)
    writer = Chem.SDWriter(output_sdf)
    passed_records = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if passes_bbb_rules(mol):
            mol_3d = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
            if res == 0:
                AllChem.MMFFOptimizeMolecule(mol_3d)
                mol_3d.SetProp("_Name", str(row["ligand_id"]))
                mol_3d.SetProp("SMILES", str(row["smiles"]))
                writer.write(mol_3d)
                passed_records.append({"ligand_id": row["ligand_id"], "smiles": row["smiles"]})

    writer.close()
    pd.DataFrame(passed_records).to_csv(output_csv, index=False)
    print(f"[Mac mini M2] Filtered {len(passed_records)} CNS-penetrant 3D ligands out of {len(df)}.")

if __name__ == "__main__":
    process_library(sys.argv[1], sys.argv[2], sys.argv[3])
```

---

## 6. Master Multi-Node Orchestration Script (`master_pipeline.py`)

Save this script to `/mnt/pd_research/scripts/master_pipeline.py` and run it on your **Mac mini M2**.

* **Supports Two Separate Arch PCs or a Single Dual-GPU Arch PC:** Edit the `NODES` dictionary at the top of the script with the IP addresses and usernames of `arch_4070` and `arch_2080`. (If both GPUs are installed in the same Arch machine, set both `host` values to the same IP and set `gpu_id` to `"0"` and `"1"`).
* **End-to-End Automation:**
  1. Filters SMILES for Blood-Brain Barrier rules on the **Mac mini M2**.
  2. Runs **P2Rank** on `arch-4070` to automatically locate the catalytic or allosteric binding pocket coordinates $(x, y, z)$.
  3. Launches **DiffDock** on `arch-4070` (16GB VRAM) and Dockerized **GNINA** on `arch-2080` (8GB VRAM) simultaneously using parallel SSH threads.
  4. Parses the docked `.sdf` outputs and runs **ADMET-AI** on `arch-2080` to confirm BBB permeability and rule out hERG cardiotoxicity or DILI liver injury.

```python
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
```

---

## 7. Quick-Start Execution Commands

Once your NFS share is mounted at `/mnt/pd_research` across all three nodes, run your entire campaign from the **Mac mini M2**:

```bash
# 1. Activate the Mac mini M2 environment
micromamba activate chem_m2

# 2. Download and protonate LRRK2 (pH 7.4) and GBA1 (pH 5.5)
python3 /mnt/pd_research/scripts/prep_targets.py

# 3. Launch a multi-node screen against LRRK2 (G2019S kinase pocket)
python3 /mnt/pd_research/scripts/master_pipeline.py \
  --target LRRK2 \
  --pdb 7LHT_clean.pdb

# 4. Launch a multi-node screen for GBA1 allosteric lysosomal chaperones
python3 /mnt/pd_research/scripts/master_pipeline.py \
  --target GBA1 \
  --pdb 6N3F_clean.pdb
```

---

## 8. Pre-Flight Verification Checklist

Before launching large-scale screening batches, verify each item below:

- [ ] **Arch Driver Check:** Running `nvidia-smi` on `arch-4070` shows **RTX 4070 Ti Super (16376 MiB)** and on `arch-2080` shows **RTX 2080 Super (8192 MiB)** with `nvidia-open-dkms` active.
- [ ] **Arch Docker GPU Check:** Running `docker run --rm --gpus all gnina/gnina:latest gnina --version` on `arch-2080` prints the GNINA version without `libstdc++` or CUDA errors.
- [ ] **SSH Mesh Check:** Running `ssh archuser@192.168.1.100 "nvidia-smi"` and `ssh archuser@192.168.1.101 "nvidia-smi"` from the **Mac mini M2** succeeds without prompting for a password.
- [ ] **Shared NFS Storage Check:** Creating a test file `touch /mnt/pd_research/test.txt` on the Mac mini M2 is immediately visible on both `arch-4070` and `arch-2080`.
- [ ] **DiffDock PyG Check:** Running `~/.local/bin/micromamba run -n diffdock python -c "import torch, torch_scatter; print(torch.cuda.is_available())"` on `arch-4070` returns `True`.
