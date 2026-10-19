# Validating the pipeline: retrospective benchmark

Question answered: *does the docking/scoring stage rank known LRRK2 / GBA1 actives above decoys?*
If not, new "hits" from the pipeline are not meaningful.

```bash
# 0. Convert downloaded decoys (see "Getting decoys" below) to CSV, dropping anything that matches a known active
python scripts/convert_decoys.py decoys_final.ism --output decoys.csv --library data/chembl_lrrk2_gba1_library.csv

# 1. Build actives (from data/chembl_lrrk2_gba1_library.csv) + decoys
python scripts/benchmark.py prepare --target LRRK2 --decoy-pool decoys.csv --decoy-ratio 20

# 1b. No docking needed: do the decoys mimic the actives? (AUC of plain properties should be ~0.5)
python scripts/benchmark.py check --target LRRK2

# 2. Dock them on arch-2080 with the same P2Rank pocket + GNINA settings as the pipeline
python scripts/benchmark.py run --target LRRK2 --pdb 7LHT_clean.pdb      # add --pocket-index N to override

# 3. Metrics, bootstrap CIs, trivial baselines, plain-language verdict -> 07_benchmark/<target>/report.md
python scripts/benchmark.py evaluate --target LRRK2
```

## Reading the report
- ROC-AUC 0.5 is random. Trust a score only if its 95% CI excludes 0.5 **and** it clearly beats the
  `Baseline:` rows (molecular weight, heavy atoms, logP), which use no docking at all.
- EF1/5/10% = how many times more actives than chance appear in the top 1/5/10% of the ranking.
- A good result is necessary, not sufficient. It does not show any new compound works.

## Getting decoys
Correction to my earlier suggestion: as far as I know **DUD-E has no LRRK2 target** (it does list
glucocerebrosidase, `glcm`, for GBA1). I could not reach these sites from the build environment, so please check
each one yourself. Options for LRRK2, best first:
1. **DUDE-Z** (dudez.docking.org): upload your own active SMILES (e.g. the `act_` rows of `labels.csv`) and it
   generates property-matched, topologically dissimilar ZINC decoys for them. This fits any target.
2. **A large ZINC subset** of drug-like molecules as `--decoy-pool`; `prepare` does the property matching and
   dissimilarity filtering itself.
3. For GBA1, the DUD-E `glcm` decoys (`decoys_final.ism`) can go straight into `convert_decoys.py`.

Whichever you use, run `check` afterwards.

## Decoy pools (checked offline)
`--decoy-pool rdkit-nci` uses the 5,000 NCI structures bundled with RDKit so you can try the workflow with no
download. Offline testing showed it is **too small and not kinase-like**: for LRRK2 the decoys already mismatch on
H-bond donors at 3 decoys per active (property AUC ~0.68) and get worse at 10:1 (molecular weight AUC 0.66, heavy
atoms 0.71). For GBA1 (smaller, more polar actives) it matches acceptably. Use `check` to decide; for real
results use DUD-E or a large ZINC subset. NCI compounds are also not verified inactives.

## Tool details verified against upstream docs
DiffDock `--config default_inference_args.yaml` and its CSV columns, GNINA flags (`--score_only`, `--seed`,
`--num_modes`, `--cnn_scoring`, `--center_*`) and ADMET-AI's `admet_predict` CLI and `BBB_Martins` / `hERG` /
`DILI` output columns match the code. P2Rank has moved to 2.5.1 (Java 17-24); the pinned 2.4.2 still works.

## Known limits of the current data
- **Decoys:** without `--decoy-pool` the script falls back to the other target's actives, a weak
  cross-target control. Get real property-matched decoys before drawing conclusions.
- **LRRK2 actives are mostly kinome-panel data** (148 of 151 rows come from broad kinase panels), so many are
  promiscuous kinase inhibitors; only 3 have a dedicated-assay source.
- **GBA1 actives are mostly active-site inhibitors** (only 4 flagged activator/chaperone). The default GBA1
  pocket (P2Rank rank 2) is an unverified assumption; inspect the structure and set `--pocket-index`.
- Library cleaning removed 15 non-human-enzyme rows (Agrobacterium / almond beta-glucosidase) by keyword; other
  non-human rows could remain. `python scripts/build_chembl_library.py --clean IN --output OUT` re-applies it offline.
- Not yet run on real hardware: `run` and the GNINA/SSH path are untested; `prepare`/`evaluate` are unit-tested.
