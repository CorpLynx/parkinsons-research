"""Shared configuration and input validation for the PD pipeline.

All host names, users and paths come from environment variables (or a local,
git-ignored .env exported into the shell) so no site-specific values or
credentials live in the repository.
"""
import os
import re

SHARED_ROOT = os.environ.get("PD_SHARED_ROOT", "/mnt/pd_research")

# Pin by digest in production, e.g. gnina/gnina@sha256:<digest>
GNINA_IMAGE = os.environ.get("PD_GNINA_IMAGE", "gnina/gnina:latest")

MAMBA_BIN = os.environ.get("PD_MAMBA_BIN", "~/.local/bin/micromamba")

_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")
_PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
_PDB_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.pdb$")
_HOST_RE = re.compile(r"^[A-Za-z0-9.-]{1,253}$")
_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")

TARGETS = {
    # PDB IDs were taken from the original README; confirm construct/mutation
    # status (e.g. G2019S) on rcsb.org before drawing biological conclusions.
    "LRRK2": {"pdb_id": "7LHT", "ph": 7.4},
    "GBA1": {"pdb_id": "6N3F", "ph": 5.5},
}


def sanitize_id(value):
    """Make a ligand identifier safe to embed in file names and shell args."""
    cleaned = _ID_RE.sub("_", str(value)).strip(".")
    return cleaned[:64] or "unnamed"


def validate_pdb_id(pdb_id):
    if not _PDB_ID_RE.match(pdb_id):
        raise ValueError(f"Invalid PDB id: {pdb_id!r}")
    return pdb_id.upper()


def validate_pdb_filename(name):
    """Reject anything that is not a plain *.pdb file name (no paths)."""
    if not _PDB_FILE_RE.match(name) or ".." in name:
        raise ValueError(f"Invalid PDB filename: {name!r}")
    return name


def node_config(prefix):
    """Read one GPU node's connection settings from the environment."""
    host = os.environ.get(f"PD_{prefix}_HOST")
    user = os.environ.get(f"PD_{prefix}_USER")
    if not host or not user:
        raise RuntimeError(
            f"Set PD_{prefix}_HOST and PD_{prefix}_USER (see .env.example)."
        )
    if not _HOST_RE.match(host) or not _USER_RE.match(user):
        raise ValueError(f"Invalid host/user for {prefix}")
    gpu_id = os.environ.get(f"PD_{prefix}_GPU", "0")
    if not gpu_id.isdigit():
        raise ValueError(f"PD_{prefix}_GPU must be a number")
    return {"host": host, "user": user, "gpu_id": gpu_id}
