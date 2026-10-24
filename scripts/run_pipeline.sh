#!/usr/bin/env bash
#
# run_pipeline.sh - Simplified entry point for the Parkinson's pipeline
#
# Runs the full pipeline on the Mac mini M2 orchestrator, orchestrating
# target prep (Stage 0), CNS filtering (Stage 1), and multi-node GPU
# docking/ADMET triage (Stages 2-4) via master_pipeline.py.
#
# Usage:
#   set -a; source .env; set +a        # configure PD_* env vars first
#   ./scripts/run_pipeline.sh              # run both targets (full pipeline; needs Arch GPU nodes)
#   ./scripts/run_pipeline.sh --local      # Stage 1 only (CNS filter; no GPU needed)
#   ./scripts/run_pipeline.sh --target LRRK2 --pdb 7LHT_clean.pdb
#   ./scripts/run_pipeline.sh --target GBA1   --pdb 2V3D_clean.pdb
#
# Environment (from .env):
#   PD_SHARED_ROOT   Shared NFS root (default: ~/pd_research)
#   PD_ARCH4070_* / PD_ARCH2080_*   GPU node connection settings (needed for full run)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
ENV_FILE="${REPO_ROOT}/.env"
MAMBA_ENV="chem_m2"
LOCAL_ONLY=0
TARGET=""
PDB=""
RAW_SMILES=""

usage() {
  cat <<EOF
Usage: $0 [--local] [--target TARGET] [--pdb PDB_FILE] [--raw-smiles CSV]

Options:
  --local            Run Stage 1 only (CNS filtering). No Arch GPU nodes required.
  --target TARGET    Target to run: LRRK2 or GBA1 (default: both).
  --pdb PDB_FILE     Cleaned PDB filename for the target (e.g. 7LHT_clean.pdb, 2V3D_clean.pdb).
  --raw-smiles CSV   Path to raw SMILES CSV (default: curated ChEMBL LRRK2/GBA1 library).
  -h, --help         Show this help message.

Before running:
  set -a; source .env; set +a   # set PD_SHARED_ROOT and Arch node settings
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      LOCAL_ONLY=1; shift ;;
    --target)
      TARGET="$2"; shift 2 ;;
    --pdb)
      PDB="$2"; shift 2 ;;
    --raw-smiles)
      RAW_SMILES="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Load .env if present
# ---------------------------------------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a; source "${ENV_FILE}"; set +a
else
  echo "WARNING: .env not found. Copy .env.example to .env and set PD_* variables."
  echo "  cp .env.example .env && edit .env"
  echo "  Then: set -a; source .env; set +a"
fi

export PD_SHARED_ROOT="${PD_SHARED_ROOT:-${HOME}/pd_research}"
export PD_MAMBA_BIN="${PD_MAMBA_BIN:-${HOME}/micromamba/bin/micromamba}"

if [[ -z "${PD_SHARED_ROOT}" || "${PD_SHARED_ROOT}" == "" ]]; then
  echo "ERROR: PD_SHARED_ROOT is not set. Edit .env and run: set -a; source .env; set +a"
  exit 1
fi

# ---------------------------------------------------------------------------
# Activate chem_m2
# ---------------------------------------------------------------------------
echo "Activating ${MAMBA_ENV} environment..."
eval "$("${PD_MAMBA_BIN}" shell hook -s "$(basename "${SHELL}")" 2>/dev/null || true)"
micromamba activate "${MAMBA_ENV}"

# ---------------------------------------------------------------------------
# Stage 0: Prepare targets (download + protonate PDBs)
# ---------------------------------------------------------------------------
run_target_prep() {
  echo "============================================================"
  echo " STAGE 0: Target Preparation (Mac mini M2)"
  echo "============================================================"
  python3 "${SCRIPTS_DIR}/prep_targets.py"
}

# ---------------------------------------------------------------------------
# Stage 1: CNS filter the curated library
# ---------------------------------------------------------------------------
run_cns_filter() {
  local target="$1"
  local pdb="$2"
  local raw_smiles="${RAW_SMILES:-${PD_SHARED_ROOT}/01_ligands_raw/chembl_lrrk2_gba1_library.csv}"

  echo "============================================================"
  echo " STAGE 1: CNS / BBB Filtering for ${target} (Mac mini M2)"
  echo "============================================================"
  echo "  Input SMILES: ${raw_smiles}"

  python3 "${SCRIPTS_DIR}/cns_filter.py" \
    "${raw_smiles}" \
    "${PD_SHARED_ROOT}/02_ligands_cns_filtered/${target}_cns.sdf" \
    "${PD_SHARED_ROOT}/02_ligands_cns_filtered/${target}_cns.csv"
}

# ---------------------------------------------------------------------------
# Stages 2-4: Full multi-node pipeline (requires Arch GPU nodes)
# ---------------------------------------------------------------------------
run_full_pipeline() {
  local target="$1"
  local pdb="$2"

  if [[ -z "${pdb}" ]]; then
    echo "ERROR: --pdb is required for a full pipeline run."
    echo "  Use: $0 --target ${target} --pdb <CLEAN_PDB>.pdb"
    exit 2
  fi

  # Validate PDB filename via the shared helper
  python3 -c "import sys; sys.path.insert(0,'${SCRIPTS_DIR}'); from pd_common import validate_pdb_filename; validate_pdb_filename('${pdb}')"

  echo "============================================================"
  echo " STAGES 2-4: Multi-node GPU Pipeline for ${target}"
  echo "============================================================"
  echo "  Target:  ${target}"
  echo "  PDB:     ${pdb}"
  echo "  Arch4070: ${PD_ARCH4070_HOST:-<not set>} (GPU ${PD_ARCH4070_GPU:-0})"
  echo "  Arch2080: ${PD_ARCH2080_HOST:-<not set>} (GPU ${PD_ARCH2080_GPU:-0})"

  python3 "${SCRIPTS_DIR}/master_pipeline.py" \
    --target "${target}" \
    --pdb "${pdb}" \
    --raw_smiles "${PD_SHARED_ROOT}/01_ligands_raw/chembl_lrrk2_gba1_library.csv"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ "${LOCAL_ONLY}" -eq 1 ]]; then
  echo "Running LOCAL mode (Stage 1 CNS filter only)..."
  run_target_prep
  run_cns_filter "LRRK2" "7LHT_clean.pdb"
  run_cns_filter "GBA1"   "2V3D_clean.pdb"
  echo
  echo "LOCAL mode complete. CNS-filtered ligands are in:"
  echo "  ${PD_SHARED_ROOT}/02_ligands_cns_filtered/"
  exit 0
fi

# Full pipeline mode
if [[ -n "${TARGET}" ]]; then
  # Single target
  case "${TARGET}" in
    LRRK2)
      run_target_prep
      run_full_pipeline "LRRK2" "${PDB:-7LHT_clean.pdb}"
      ;;
    GBA1)
      run_target_prep
      run_full_pipeline "GBA1" "${PDB:-2V3D_clean.pdb}"
      ;;
    *)
      echo "ERROR: Unknown target '${TARGET}'. Use LRRK2 or GBA1."
      exit 2
      ;;
  esac
else
  # Both targets
  run_target_prep
  run_full_pipeline "LRRK2" "${PDB:-7LHT_clean.pdb}"
  run_full_pipeline "GBA1"   "${PDB:-2V3D_clean.pdb}"
fi

echo
echo "============================================================"
echo " Pipeline complete. Final leads in:"
echo "  ${PD_SHARED_ROOT}/06_final_hits_admet/"
echo "============================================================"
