#!/usr/bin/env bash
#
# run_pipeline.sh - Simplified entry point for the Parkinson's pipeline
#
# Runs the full pipeline on the Mac mini M2 orchestrator, orchestrating
# target prep (Stage 0), CNS filtering (Stage 1), and multi-node GPU
# docking/ADMET triage (Stages 2-4) via master_pipeline.py.
#
# Every invocation creates a versioned run directory under
#   $PD_SHARED_ROOT/runs/{RUN_ID}/{target}/
# so repeated campaigns never clobber each other.
#
# Usage:
#   set -a; source .env; set +a        # configure PD_* env vars first
#   ./scripts/run_pipeline.sh              # run both targets (full pipeline; needs Arch GPU nodes)
#   ./scripts/run_pipeline.sh --local      # Stage 1 only (CNS filter; no GPU needed)
#   ./scripts/run_pipeline.sh --target LRRK2 --pdb 7LHT_clean.pdb --run-id myrun_20261024
#   ./scripts/run_pipeline.sh --target GBA1   --pdb 2V3D_clean.pdb --resume
#   ./scripts/run_pipeline.sh --list-runs
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
LIST_RUNS=0
TARGET=""
PDB=""
RAW_SMILES=""
RUN_ID=""
RESUME=0

usage() {
  cat <<EOF
Usage: $0 [--local] [--list-runs] [--target TARGET] [--pdb PDB_FILE]
          [--raw-smiles CSV] [--run-id ID] [--resume]

Options:
  --local            Run Stage 1 only (CNS filtering). No Arch GPU nodes required.
  --list-runs        List existing run directories under PD_SHARED_ROOT/runs/.
  --target TARGET    Target to run: LRRK2 or GBA1 (default: both).
  --pdb PDB_FILE     Cleaned PDB filename for the target (e.g. 7LHT_clean.pdb, 2V3D_clean.pdb).
  --raw-smiles CSV   Path to raw SMILES CSV (default: curated ChEMBL LRRK2/GBA1 library).
  --run-id ID        Unique run identifier (auto-generated as YYYYMMDD_HHMMSS if omitted).
  --resume           Skip stages whose outputs already exist in the run directory.
  -h, --help         Show this help message.

Before running:
  set -a; source .env; set +a   # set PD_SHARED_ROOT and Arch node settings
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      LOCAL_ONLY=1; shift ;;
    --list-runs)
      LIST_RUNS=1; shift ;;
    --target)
      TARGET="$2"; shift 2 ;;
    --pdb)
      PDB="$2"; shift 2 ;;
    --raw-smiles)
      RAW_SMILES="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --resume)
      RESUME=1; shift ;;
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
# List existing runs and exit
# ---------------------------------------------------------------------------
if [[ "${LIST_RUNS}" -eq 1 ]]; then
  echo "Existing run directories under ${PD_SHARED_ROOT}/runs/:"
  if [[ -d "${PD_SHARED_ROOT}/runs" ]]; then
    find "${PD_SHARED_ROOT}/runs" -maxdepth 2 -name manifest.json 2>/dev/null | sort | while read -r m; do
      rid="$(dirname "${m}" | xargs dirname | xargs basename)"
      tgt="$(basename "$(dirname "${m}")")"
      status="$(python3 -c "import json,sys; print(json.load(open('${m}')).get('status','?'))" 2>/dev/null || echo "?")"
      nleads="$(python3 -c "import json,sys; print(json.load(open('${m}')).get('n_final_leads','?'))" 2>/dev/null || echo "?")"
      echo "  run=${rid}  target=${tgt}  status=${status}  final_leads=${nleads}"
    done
  else
    echo "  (none yet)"
  fi
  exit 0
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
  local run_arg=()
  local resume_arg=()

  if [[ -z "${pdb}" ]]; then
    echo "ERROR: --pdb is required for a full pipeline run."
    echo "  Use: $0 --target ${target} --pdb <CLEAN_PDB>.pdb"
    exit 2
  fi

  # Validate PDB filename via the shared helper
  python3 -c "import sys; sys.path.insert(0,'${SCRIPTS_DIR}'); from pd_common import validate_pdb_filename; validate_pdb_filename('${pdb}')"

  if [[ -n "${RUN_ID}" ]]; then
    run_arg=(--run-id "${RUN_ID}")
  fi
  if [[ "${RESUME}" -eq 1 ]]; then
    resume_arg=(--resume)
  fi

  echo "============================================================"
  echo " STAGES 2-4: Multi-node GPU Pipeline for ${target}"
  echo "============================================================"
  echo "  Target:   ${target}"
  echo "  PDB:      ${pdb}"
  echo "  Run ID:   ${RUN_ID:-<auto-generated>}"
  echo "  Resume:   ${RESUME}"
  echo "  Arch4070: ${PD_ARCH4070_HOST:-<not set>} (GPU ${PD_ARCH4070_GPU:-0})"
  echo "  Arch2080: ${PD_ARCH2080_HOST:-<not set>} (GPU ${PD_ARCH2080_GPU:-0})"

  python3 "${SCRIPTS_DIR}/master_pipeline.py" \
    --target "${target}" \
    --pdb "${pdb}" \
    --raw_smiles "${PD_SHARED_ROOT}/01_ligands_raw/chembl_lrrk2_gba1_library.csv" \
    "${run_arg[@]}" "${resume_arg[@]}"
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
  # Both targets, sequentially
  run_target_prep
  run_full_pipeline "LRRK2" "${PDB:-7LHT_clean.pdb}"
  run_full_pipeline "GBA1"   "${PDB:-2V3D_clean.pdb}"
fi

echo
echo "============================================================"
echo " Pipeline complete. Final leads in:"
echo "  ${PD_SHARED_ROOT}/runs/<RUN_ID>/<target>/06_admet/"
echo "  List runs: ./scripts/run_pipeline.sh --list-runs"
echo "============================================================"
