# README Review & Proposed Improvements

Status: implemented, except items marked **open**.

Implemented: A1–A9, A11 (DiffDock `--config`; verify against your pinned commit), B (config via env, known_hosts, scripts in repo, image/commit pinning guidance), C, most of D. **Open:** A10 (PDB IDs/G2019S could not be checked — RCSB was unreachable from the build environment), per-node env files for DiffDock/ADMET/REINVENT4, LICENSE and citations (owner's choice), Ollama/OpenMM role clarification. The DiffDock rescoring and GNINA/Docker commands are unit-tested only for quoting; they have **not** been run on GPU hardware.

## A. Correctness issues in the pipeline (highest priority)

1. **DiffDock output is never used.** Stage 3 runs DiffDock, but Stage 4 only parses GNINA's SDF. Either feed DiffDock poses to GNINA rescoring (`gnina --score_only`) or drop DiffDock from the diagram.
2. **Thread failures are swallowed.** `run_remote_ssh` raises inside `threading.Thread`, so the exception never reaches `main()` and Stage 4 runs on missing files. Use `concurrent.futures.ThreadPoolExecutor` and call `.result()`.
3. **Empty-hit crash.** `pd.DataFrame([]).sort_values("CNNscore")` raises `KeyError` when no ligand passes. Guard before sorting.
4. **Multiple poses per ligand.** GNINA writes several poses per input; hits are not deduplicated, so one ligand can appear repeatedly. Keep the best pose per `ligand_id`.
5. **`SMILES` property may be dropped** by GNINA; the fallback `MolToSmiles` includes explicit hydrogens' effects. Join back to the filtered CSV on `ligand_id` instead.
6. **PDBFixer `findMissingResidues()`** will model long missing loops with no confidence. Clear or restrict `fixer.missingResidues` for large gaps (LRRK2 is large).
7. **Comment/code mismatch:** section 2.2 says "strict" channel priority but sets `flexible`.
8. **`rdkit-pypi` is deprecated** — use `rdkit` from conda-forge or PyPI.
9. **Starter library SMILES/labels are unverified.** Several entries (e.g. "MLi2", "LTI291") do not look like the named compounds. Verify against ChEMBL/PubChem or rename to generic IDs, and check they pass `passes_bbb_rules`. **Addressed:** the curated `data/chembl_lrrk2_gba1_library.csv` now validates clean — all 265 SMILES parse, every stored property (mw/logp/tpsa/hbd/hba/rot_bonds/heavy_atoms) matches an RDKit recomputation, the `passes_bbb` column is fully consistent with `passes_bbb_rules`, and named compounds (MLi-2, Ambroxol, Afegostat/isofagomine, …) are structurally consistent with their labels.
10. **PDB IDs and "G2019S" claim:** confirm 7LHT/6N3F are the intended constructs and whether 7LHT carries G2019S; Quick-Start says "G2019S kinase pocket".
11. **DiffDock invocation:** `python -m inference` should pass the config (`--config default_inference_args.yaml`) per upstream docs; verify against the pinned commit.

## B. Reproducibility

- Pin versions: `gnina/gnina:latest`, DiffDock/REINVENT4 git clones (use commit SHAs), P2Rank is pinned already.
- Provide `environment.yml` per node (added: `environment.yml` for `chem_m2`; add `diffdock.yml`, `admet.yml`, `reinvent4.yml`).
- Replace hard-coded IPs/users with a `config.yaml` or env vars; remove `AutoAddPolicy` in favor of known_hosts.
- Store scripts in this repo and clone it into the share rather than copy-pasting from the README.

## C. Security / ops

- NFS export uses `insecure,all_squash` on a /24; document that this is LAN-only, trusted-network setup (no auth).
- `curl | sh` installers (Homebrew, micromamba): note to review before running.
- `sudo mount` / fstab examples lack `nofail`; add it so a down NFS server cannot hang boot.

## D. Documentation structure

- README is ~28 KB with full scripts embedded; reduce to overview + setup and link to `scripts/`.
- Add: pipeline diagram, expected runtimes/VRAM per stage, troubleshooting, results/interpretation section.
- **Added (control benchmark):** `data/controls.csv` (known binders + decoy negatives per target) plus `scripts/check_controls.py` (`spike` to co-dock the controls, `evaluate` to report positives-vs-negatives ROC-AUC with a pass/fail verdict). `master_pipeline.py` now also emits an unfiltered `{target}_all_scores.csv` so negatives can be scored. Lets a run prove it ranks known binders above obvious inactives before any hit is trusted. Note: negatives are obvious-inactive sanity decoys, not property-matched DUD-E decoys.
- Ollama/OpenMM are listed in the role matrix but unused by any script — implement or remove.
- 8 GB Mac mini cannot comfortably run `llama3.1:8b`; note the RAM requirement.
- Add disclaimer: computational predictions only, research use, not medical advice.
- Add a LICENSE (owner's choice) and citations for DiffDock, GNINA, P2Rank, ADMET-AI, REINVENT4, AlphaFold/ColabFold.
- Fix the 8-item checklist to include an end-to-end smoke test using the starter library.
