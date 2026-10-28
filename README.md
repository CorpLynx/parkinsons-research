# Distributed Local AI Pipeline for Parkinson's Drug Discovery (Arch Linux + macOS Edition)

**Hardware Specification, Arch Linux Environment Configuration, Target Preparation & Automated Orchestration Guide**

This document outlines the complete system requirements, Arch Linux kernel/driver configuration, containerized and isolated CUDA dependencies, target preparation protocols, and automated orchestration scripts to transform a **Mac mini M2**, an **NVIDIA RTX 4070 Ti Super (16GB)**, and an **NVIDIA RTX 2080 Super (8GB)** into an integrated computational biology and drug-screening cluster targeting Parkinson's Disease (PD).

> **Scope & limitations.** This is a *computational triage* workflow. Docking scores and ADMET predictions are noisy in-silico estimates that prioritise compounds for experiments; they are not evidence of efficacy or safety, and nothing here is medical advice. Target PDB IDs (`7LHT` LRRK2, `2V3D` GBA1) and the curated ChEMBL ligand library are unverified defaults: confirm construct, mutation status and compound identity before use. See [SECURITY.md](SECURITY.md) for the trusted-LAN threat model, [CITATIONS.md](CITATIONS.md) for tool/data citations, [LICENSE](LICENSE) for licensing, and `pytest` / `ruff` for local checks.

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
# Review the installer before executing it (supply-chain hygiene):
curl -L micro.mamba.pm/install.sh -o /tmp/micromamba_install.sh && less /tmp/micromamba_install.sh
"${SHELL}" /tmp/micromamba_install.sh
source ~/.bashrc

# Channel priority (pytorch/nvidia first, conda-forge fallback)
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

# Export ONLY to the two other machines (/32 each), never a whole subnet. NFSv4 here has no
# user authentication (AUTH_SYS): anyone who can spoof an allowed IP gets read/write as your user.
# Also restrict TCP 2049 to those IPs in your firewall, and keep the share off untrusted/guest networks.
# 192.168.1.101 = arch-2080, 192.168.1.102 = mac-mini (example IPs: use fixed/reserved addresses)
echo "/srv/nfs/pd_research 192.168.1.101(rw,sync,no_subtree_check,all_squash,anonuid=$(id -u),anongid=$(id -g)) 192.168.1.102(rw,sync,no_subtree_check,all_squash,anonuid=$(id -u),anongid=$(id -g))" | sudo tee -a /etc/exports

sudo systemctl enable --now nfs-server
sudo exportfs -arv
```

### 3.2 Mount Shared Workspace on `arch-2080`

```bash
sudo mkdir -p /mnt/pd_research
# Replace 192.168.1.100 with the LAN IP of arch-4070
sudo mount -t nfs -o nosuid,nodev 192.168.1.100:/srv/nfs/pd_research /mnt/pd_research

# Optional: Add to /etc/fstab for automatic mounting on boot
echo "192.168.1.100:/srv/nfs/pd_research /mnt/pd_research nfs defaults,_netdev,nofail,nosuid,nodev,noexec 0 0" | sudo tee -a /etc/fstab
```

### 3.3 Mount Shared Workspace & SSH Keys on `mac-mini` (macOS)

On the **Mac mini M2**, generate an SSH keypair, copy it to both Arch nodes, and mount the NFS share:

```bash
# 1. Generate SSH key and copy to both Arch nodes (replace IPs/usernames as needed)
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id archuser@192.168.1.100  # arch-4070
ssh-copy-id archuser@192.168.1.101  # arch-2080

# Verify each host key fingerprint out-of-band on first connect. The pipeline uses paramiko
# RejectPolicy, so hosts MUST already be in ~/.ssh/known_hosts (ssh to each node once manually).
# Recommended: disable password SSH auth on the Arch nodes (PasswordAuthentication no).

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
git checkout <commit-sha>   # pin the version you validated

# Create isolated Python 3.10 + PyTorch 2.2.2 + CUDA 12.1 environment
micromamba create -n diffdock python=3.10 pytorch=2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge -y
micromamba activate diffdock

# Install prebuilt PyTorch Geometric CUDA wheels matching PyTorch 2.2.2 + cu121
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install torch-geometric e3nn biopython rdkit spyrmsd pandas scipy pyyaml fair-esm

micromamba deactivate
```

#### C. LocalColabFold (AlphaFold2 Mutant Structure Prediction)
```bash
mkdir -p ~/localcolabfold && cd ~/localcolabfold
# Review the script (and prefer a pinned release tag/commit) before running it
curl -L https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh -O
less install_colabbatch_linux.sh
chmod +x install_colabbatch_linux.sh
./install_colabbatch_linux.sh
```

---

### 4.2 Node 2: `arch-2080` (RTX 2080 Super — 8GB VRAM)

Run the following installations on `arch-2080`.

#### A. GNINA (Dockerized 3D CNN Docking & Affinity Rescoring)
Pulling the official container avoids Arch `glibc` / `boost` library incompatibilities while running at native speed on the 2080 Super:

```bash
# Pin a specific tag or digest for reproducibility, and export it as PD_GNINA_IMAGE
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
git checkout <commit-sha>   # pin the version you validated
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

Run [`scripts/prep_targets.py`](scripts/prep_targets.py) on the **Mac mini M2**. It downloads, strips crystallographic artifacts/waters, rebuilds missing side-chain atoms via `PDBFixer`, and protonates two major Parkinson's targets at their physiological $\text{pH}$ values:
* **LRRK2 Kinase (`7LHT`):** Protonated at cytosolic $\text{pH } 7.4$.
* **GBA1 Glucocerebrosidase (`2V3D`):** Protonated at acidic lysosomal $\text{pH } 5.5$.

The script lives in [`scripts/prep_targets.py`](scripts/prep_targets.py) in this repository (clone the repo into `/mnt/pd_research/` or symlink `scripts/`).

### 5.2 Blood-Brain Barrier (CNS) Ligand Filter (`cns_filter.py`)

[`scripts/cns_filter.py`](scripts/cns_filter.py) runs on the **Mac mini M2** to filter raw SMILES strings (or generate a starter benchmark library if no input CSV is provided) using strict Central Nervous System (CNS) Blood-Brain Barrier rules:
* Molecular Weight ($\text{MW} \le 450\text{ Da}$)
* Topological Polar Surface Area ($\text{TPSA} < 90\text{ \AA}^2$)
* Lipophilicity ($1.0 \le \log P \le 4.5$)
* Hydrogen Bond Donors ($\text{HBD} \le 3$) and Acceptors ($\text{HBA} \le 7$)
* Rotatable Bonds ($\le 8$)

The script lives in [`scripts/cns_filter.py`](scripts/cns_filter.py) in this repository (clone the repo into `/mnt/pd_research/` or symlink `scripts/`).

---

## 6. Master Multi-Node Orchestration Script (`master_pipeline.py`)

Run [`scripts/master_pipeline.py`](scripts/master_pipeline.py) on your **Mac mini M2**.

* **Supports Two Separate Arch PCs or a Single Dual-GPU Arch PC:** Copy `.env.example` to `.env` and set the `PD_ARCH4070_*` / `PD_ARCH2080_*` host, user and GPU variables, then `set -a; source .env; set +a`. (If both GPUs are installed in the same Arch machine, set both hosts to the same IP and the GPU variables to `0` and `1`).
* **End-to-End Automation:**
  1. Filters SMILES for Blood-Brain Barrier rules on the **Mac mini M2**.
  2. Runs **P2Rank** on `arch-4070` to automatically locate the catalytic or allosteric binding pocket coordinates $(x, y, z)$.
  3. Launches **DiffDock** on `arch-4070` (16GB VRAM) and Dockerized **GNINA** on `arch-2080` (8GB VRAM) simultaneously; a failure on either node aborts the run. DiffDock's top poses are then rescored with GNINA (`--score_only`) so both engines feed triage.
  4. Keeps the best pose per ligand (CNNscore ≥ 0.65) and runs **ADMET-AI** on `arch-2080` to confirm BBB permeability and rule out hERG cardiotoxicity or DILI liver injury.

The script lives in [`scripts/master_pipeline.py`](scripts/master_pipeline.py) in this repository (clone the repo into `/mnt/pd_research/` or symlink `scripts/`).

---

## 7. Quick-Start Execution Commands

### 7.1 One-shot install (Mac mini M2)

```bash
cd /path/to/parkinsons-research
bash scripts/install.sh --shared-root /mnt/pd_research
# then edit .env with your Arch node host/user/GPU settings
set -a; source .env; set +a
```

### 7.2 Run the pipeline

Each invocation creates a versioned run directory under
`$PD_SHARED_ROOT/runs/{RUN_ID}/{target}/`, so repeated campaigns never clobber
each other and every run keeps a `manifest.json`.

```bash
# Verify the full remote command surface WITHOUT touching the GPU nodes
./scripts/run_pipeline.sh --target LRRK2 --pdb 7LHT_clean.pdb --dry-run

# Real run (Stages 2-4 need the Arch GPU nodes)
./scripts/run_pipeline.sh --target LRRK2 --pdb 7LHT_clean.pdb
./scripts/run_pipeline.sh --target GBA1   --pdb 2V3D_clean.pdb

# Resume a crashed run (skips stages whose outputs already exist)
./scripts/run_pipeline.sh --target LRRK2 --pdb 7LHT_clean.pdb \
  --run-id 20261024_093000 --resume

# List previous campaigns
./scripts/run_pipeline.sh --list-runs

# Stage 1 only (CNS filter; no Arch nodes required)
./scripts/run_pipeline.sh --local
```

### 7.3 Manual invocation

If you prefer to drive the steps directly:

```bash
micromamba activate chem_m2

# Download and protonate LRRK2 (pH 7.4) and GBA1 (pH 5.5)
python3 /mnt/pd_research/scripts/prep_targets.py

# Build the curated ChEMBL library (LRRK2 + GBA1 compounds)
python3 /mnt/pd_research/scripts/build_chembl_library.py \
  --output /mnt/pd_research/01_ligands_raw/chembl_lrrk2_gba1_library.csv

# Launch a multi-node screen against LRRK2 (WT full-length kinase pocket)
python3 /mnt/pd_research/scripts/master_pipeline.py \
  --target LRRK2 \
  --pdb 7LHT_clean.pdb

# Launch a multi-node screen for GBA1 allosteric lysosomal chaperones
python3 /mnt/pd_research/scripts/master_pipeline.py \
  --target GBA1 \
  --pdb 2V3D_clean.pdb
```

### 7.4 Where results land

```
$PD_SHARED_ROOT/runs/{RUN_ID}/{target}/
├── 02_cns/          CNS-filtered SDF + CSV
├── 03_pockets/      P2Rank pocket predictions
├── 04_diffdock/     DiffDock poses
├── 05_gnina/        GNINA docked + rescored SDFs, DiffDock input CSV
├── 06_admet/        Top hits, all_scores.csv, ADMET output, FINAL_SAFE_LEADS.csv
└── manifest.json    Run metadata, stage completion, lead counts
```

Summarise any run with:

```bash
python3 scripts/check_pipeline.py \
  $PD_SHARED_ROOT/runs/{RUN_ID}/{target}/06_admet/{target}_FINAL_SAFE_LEADS.csv
```

### 7.5 Validating the screen with known binders and decoys (controls)

Docking and CNN scores are noisy. Before trusting a ranking, confirm the screen
puts molecules that are *known* to bind the target above molecules that are
known *not* to. `data/controls.csv` ships a labelled set per target: positive
controls (e.g. **MLi-2** and staurosporine for LRRK2; isofagomine, ambroxol and
the native 2V3D chaperone miglustat for GBA1) and negative "decoy" controls
(caffeine, glucose, aspirin, …) that should score low.

1. **Spike** the controls into the CNS-filtered ligand set so they are docked
   alongside the library (controls bypass the BBB filter on purpose, so that
   negatives such as glucose are not dropped before docking):

   ```bash
   python3 scripts/check_controls.py spike --target LRRK2 \
     --filtered $PD_SHARED_ROOT/runs/{RUN_ID}/LRRK2/02_cns/LRRK2_cns.csv \
     --out      $PD_SHARED_ROOT/runs/{RUN_ID}/LRRK2/02_cns/LRRK2_cns_spiked.csv
   ```

2. Run the pipeline as usual. Stage 4 now also writes an unfiltered
   `{target}_all_scores.csv` (best pose per ligand, before the hit threshold),
   which is what the evaluator needs — negatives are expected to fall *below*
   the hit cutoff.

3. **Evaluate** how well positives separated from negatives:

   ```bash
   python3 scripts/check_controls.py evaluate --target LRRK2 \
     --scores $PD_SHARED_ROOT/runs/{RUN_ID}/LRRK2/06_admet/LRRK2_all_scores.csv \
     --score-col CNNscore --min-auc 0.7
   ```

   It prints each control's score, the median of each group, and the ROC-AUC
   (probability a random positive outranks a random negative; `1.0` perfect,
   `0.5` coin-flip), and exits non-zero if the AUC is below `--min-auc`. This is
   a **sanity** check: the negatives are obvious-inactive decoys, not
   property-matched DUD-E decoys, so a high AUC rules out gross scoring failure
   rather than proving fine-grained discrimination. If known binders do **not**
   rise to the top, treat that run's ranking as unreliable.

---

## 8. Pre-Flight Verification Checklist

Before launching large-scale screening batches, verify each item below:

- [ ] **Arch Driver Check:** Running `nvidia-smi` on `arch-4070` shows **RTX 4070 Ti Super (16376 MiB)** and on `arch-2080` shows **RTX 2080 Super (8192 MiB)** with `nvidia-open-dkms` active.
- [ ] **Arch Docker GPU Check:** Running `docker run --rm --gpus all gnina/gnina:latest gnina --version` on `arch-2080` prints the GNINA version without `libstdc++` or CUDA errors.
- [ ] **SSH Mesh Check:** Running `ssh archuser@192.168.1.100 "nvidia-smi"` and `ssh archuser@192.168.1.101 "nvidia-smi"` from the **Mac mini M2** succeeds without prompting for a password.
- [ ] **Dry-Run Check:** Running `./scripts/run_pipeline.sh --target LRRK2 --pdb 7LHT_clean.pdb --dry-run` prints every remote command (P2Rank, DiffDock, GNINA, ADMET-AI) with resolved paths and quoting, without opening an SSH connection. Do this **before** every first run on the GPU nodes.
- [ ] **Smoke Test:** `pip install -r requirements-dev.txt && pytest` passes on the Mac mini, and `python scripts/smoke_test.py` builds the curated ChEMBL library and filters it to CNS-penetrant 3D ligands (MLi-2 survives the BBB filter).
- [ ] **Shared NFS Storage Check:** Creating a test file `touch /mnt/pd_research/test.txt` on the Mac mini M2 is immediately visible on both `arch-4070` and `arch-2080`.
- [ ] **DiffDock PyG Check:** Running `~/.local/bin/micromamba run -n diffdock python -c "import torch, torch_scatter; print(torch.cuda.is_available())"` on `arch-4070` returns `True`.

---

## 9. Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request:

| Job | What it does |
|-----|--------------|
| `lint` | `ruff check .`, `bandit -r scripts -ll`, `pip-audit` |
| `test` | `pytest` (54 tests) + uploads the JUnit report |
| `smoke` | Live end-to-end Stage 1: builds the curated ChEMBL library from the ChEMBL API and CNS-filters it; uploads the artifacts |

The `smoke` job hits the live ChEMBL API, so it is the check that catches a broken
library build before it reaches your laptop.
