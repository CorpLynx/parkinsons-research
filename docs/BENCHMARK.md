# Validating the pipeline: retrospective benchmark

Question answered: *does the docking/scoring stage rank known LRRK2 / GBA1 actives above decoys?*
If not, new "hits" from the pipeline are not meaningful.

```bash
# 1. Build actives (from data/chembl_lrrk2_gba1_library.csv) + decoys. Best: DUD-E / ZINC decoys.
python scripts/benchmark.py prepare --target LRRK2 --decoy-pool decoys.csv --decoy-ratio 20

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
