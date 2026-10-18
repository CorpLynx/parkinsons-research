# Security Notes

## Threat model
A trusted, private LAN with three machines owned by one researcher. The pipeline is **not** designed for multi-user or internet-exposed deployment.

## Controls in this repository
- **No secrets or site data in git:** hosts/users/paths come from environment variables (`.env.example`); `.env` is git-ignored.
- **SSH:** key auth only, `paramiko.RejectPolicy` (hosts must be pre-verified in `known_hosts`), connection and job timeouts.
- **Command injection:** every value interpolated into a remote shell command goes through `shlex.quote`; target names, PDB filenames/IDs, hosts and users are validated against strict patterns; ligand IDs are sanitised because they become file/directory names.
- **Path traversal:** `--raw_smiles` must resolve under the shared root; PDB filenames cannot contain path components.
- **Downloads:** PDB files are fetched over HTTPS from a fixed host with timeout, size cap and a format sanity check.
- **Containers:** GNINA runs with `--cap-drop ALL`, `no-new-privileges`, a non-root user, and the shared tree mounted read-only except the output directory. Pin `PD_GNINA_IMAGE` by digest.
- **Input limits:** row and heavy-atom caps on user-supplied libraries.
- **CI:** ruff (incl. flake8-bandit rules), bandit, pip-audit and pytest run on every push with read-only token permissions.

## Known residual risks (operator responsibility)
- NFS (AUTH_SYS) trusts client IPs. Export to specific hosts only, firewall port 2049, keep the LAN trusted.
- `pdbfixer`, DiffDock, GNINA, REINVENT4 and ADMET-AI are third-party code; pin versions/commits and review updates.
- Installers fetched with `curl` should be reviewed before execution.
- Pin GitHub Actions to commit SHAs if you fork this for a shared/organisational setting.

Report vulnerabilities privately to the repository owner.
