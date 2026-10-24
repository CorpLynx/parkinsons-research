#!/usr/bin/env bash
#
# install.sh - Unified install script for the Mac mini M2 orchestrator node
#
# Sets up the chem_m2 environment, installs pipeline dependencies, creates the
# working directory tree, and stages a .env file for the Parkinson's pipeline.
#
# Usage:
#   cd /path/to/parkinsons-research
#   bash scripts/install.sh [--shared-root PATH] [--skip-smoke-test]
#
# After install:
#   set -a; source .env; set +a
#   ./scripts/run_pipeline.sh          # Stage 1 locally (CNS filter only)
#   ./scripts/run_pipeline.sh --target LRRK2 --pdb 7LHT_clean.pdb   # full run (needs Arch GPU nodes)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
ENV_FILE="${REPO_ROOT}/.env"
ENV_EXAMPLE="${REPO_ROOT}/.env.example"
SHARED_ROOT_DEFAULT="${HOME}/pd_research"
SKIP_SMOKE_TEST=0

usage() {
  cat <<EOF
Usage: $0 [--shared-root PATH] [--skip-smoke-test]

Options:
  --shared-root PATH   Shared NFS root directory (default: ${SHARED_ROOT_DEFAULT})
                       On macOS this must be a local path; /mnt does not exist.
  --skip-smoke-test    Skip the end-to-end smoke test at the end of install.
  -h, --help           Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shared-root)
      SHARED_ROOT_DEFAULT="$2"; shift 2 ;;
    --skip-smoke-test)
      SKIP_SMOKE_TEST=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

SHARED_ROOT="${SHARED_ROOT_DEFAULT}"

echo "============================================================"
echo " Parkinson's Pipeline - Mac mini M2 Orchestrator Installer "
echo "============================================================"
echo "Repo root:  ${REPO_ROOT}"
echo "Shared root: ${SHARED_ROOT}"
echo

# ---------------------------------------------------------------------------
# 0. Prerequisites
# ---------------------------------------------------------------------------
echo "[0/6] Checking prerequisites..."

if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew is required but not installed. Install it first:"
  echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required but not found."
  exit 1
fi

echo "  Homebrew:   $(brew --version | head -1)"
echo "  Python:     $(python3 --version)"

# ---------------------------------------------------------------------------
# 1. Install micromamba
# ---------------------------------------------------------------------------
echo "[1/6] Installing micromamba..."

if ! command -v micromamba >/dev/null 2>&1; then
  echo "  micromamba not found. Installing via Homebrew..."
  brew install micromamba
else
  echo "  micromamba already installed: $(micromamba --version 2>/dev/null || micromamba --version)"
fi

# Initialize micromamba for the current shell (zsh/bash)
if ! micromamba shell init -s "$(basename "${SHELL}")" -r "${HOME}/micromamba" 2>/dev/null; then
  echo "  WARNING: micromamba shell init may not have completed. You may need to run:"
  echo "    micromamba shell init -s $(basename "${SHELL}") -r ~/micromamba"
fi

# ---------------------------------------------------------------------------
# 2. Create the chem_m2 conda environment
# ---------------------------------------------------------------------------
echo "[2/6] Creating chem_m2 environment from environment.yml..."

ENV_YML="${REPO_ROOT}/environment.yml"
if [[ ! -f "${ENV_YML}" ]]; then
  echo "ERROR: ${ENV_YML} not found."
  exit 1
fi

if micromamba env list 2>/dev/null | grep -q '^chem_m2'; then
  echo "  chem_m2 environment already exists. Updating..."
  micromamba env update -n chem_m2 -f "${ENV_YML}" || true
else
  micromamba create -n chem_m2 -f "${ENV_YML}" -y
fi

# Install pip-only extras not in environment.yml
echo "  Installing pip-only packages (meeko, pubchempy, chembl_webresource_client)..."
MICROMAMBA_BIN="${HOME}/micromamba/bin/micromamba"
if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
  # Try to locate micromamba binary
  MICROMAMBA_BIN="$(command -v micromamba)"
fi
"${MICROMAMBA_BIN}" run -n chem_m2 pip install meeko pubchempy chembl_webresource_client

# ---------------------------------------------------------------------------
# 3. Create the working directory tree
# ---------------------------------------------------------------------------
echo "[3/6] Creating working directory tree under ${SHARED_ROOT}..."

mkdir -p "${SHARED_ROOT}"/{00_targets,01_ligands_raw,02_ligands_cns_filtered,03_pockets,04_diffdock_poses,05_gnina_rescored,06_final_hits_admet,scripts,tools}

# Clone repo scripts into the shared root so all nodes see identical copies.
if [[ "${REPO_ROOT}" != "${SHARED_ROOT}" ]]; then
  if [[ ! -d "${SHARED_ROOT}/scripts" || "$(ls -A "${SHARED_ROOT}/scripts" 2>/dev/null | wc -l)" -eq 0 ]]; then
    echo "  Copying pipeline scripts into ${SHARED_ROOT}/scripts"
    cp "${SCRIPTS_DIR}"/*.py "${SHARED_ROOT}/scripts/"
  fi
fi

# ---------------------------------------------------------------------------
# 4. Stage the .env file
# ---------------------------------------------------------------------------
echo "[4/6] Staging .env file..."

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${ENV_EXAMPLE}" "${ENV_FILE}"
  # Point PD_SHARED_ROOT at the chosen location
  sed -i.bak "s|^PD_SHARED_ROOT=.*|PD_SHARED_ROOT=${SHARED_ROOT}|" "${ENV_FILE}"
  rm -f "${ENV_FILE}.bak"
  echo "  Created .env from .env.example (PD_SHARED_ROOT=${SHARED_ROOT})"
else
  echo "  .env already exists. Leaving unchanged."
fi

echo
echo "  IMPORTANT: Before running the pipeline, edit .env and set:"
echo "    PD_ARCH4070_HOST / PD_ARCH4070_USER / PD_ARCH4070_GPU"
echo "    PD_ARCH2080_HOST / PD_ARCH2080_USER / PD_ARCH2080_GPU"
echo "  Then:  set -a; source .env; set +a"
echo

# ---------------------------------------------------------------------------
# 5. Verify the install
# ---------------------------------------------------------------------------
echo "[5/6] Verifying install..."

"${MICROMAMBA_BIN}" run -n chem_m2 python3 -c "
import rdkit, pandas, paramiko
print('  rdkit:', rdkit.__version__)
print('  pandas:', pandas.__version__)
print('  paramiko: OK')
"

# ---------------------------------------------------------------------------
# 6. End-to-end smoke test
# ---------------------------------------------------------------------------
if [[ "${SKIP_SMOKE_TEST}" -eq 0 ]]; then
  echo "[6/6] Running end-to-end smoke test..."
  set +u
  # shellcheck disable=SC1090
  set -a; source "${ENV_FILE}"; set +a
  set -u
  export PD_SHARED_ROOT="${SHARED_ROOT}"

  "${MICROMAMBA_BIN}" run -n chem_m2 python3 "${SCRIPTS_DIR}/smoke_test.py" \
    --outdir "${SHARED_ROOT}/01_ligands_raw/smoke_test_output"
else
  echo "[6/6] Skipping smoke test (--skip-smoke-test)."
fi

echo
echo "============================================================"
echo " Install complete. Next steps:                            "
echo "============================================================"
echo "  1. Edit .env with your Arch node host/user/GPU settings."
echo "  2. set -a; source .env; set +a"
echo "  3. Download & prepare targets:"
echo "       micromamba activate chem_m2"
echo "       python3 scripts/prep_targets.py"
echo "  4. Run the pipeline (Stage 1 locally; Stages 2-4 need Arch GPU nodes):"
echo "       ./scripts/run_pipeline.sh --target LRRK2 --pdb 7LHT_clean.pdb"
echo "       ./scripts/run_pipeline.sh --target GBA1   --pdb 2V3D_clean.pdb"
echo "     Or Stage 1 only (CNS filter, no GPU needed):"
echo "       ./scripts/run_pipeline.sh --local"
echo "============================================================"
