import os
import pandas as pd
import pytest
from rdkit import Chem

import cns_filter
import master_pipeline as mp
import pd_common
import prep_targets


def test_sanitize_id_blocks_path_and_shell_chars():
    for bad in ["a/b", "x;y", "$(id)", "a b", "..", ""]:
        out = pd_common.sanitize_id(bad)
        assert out and all(c.isalnum() or c in "_.-" for c in out)
        assert not out.startswith(".")


@pytest.mark.parametrize("name", ["../x.pdb", "a/b.pdb", "x.pdb; ls", "x.txt", ".hidden.pdb", "a..b.pdb"])
def test_pdb_filename_rejects_unsafe(name):
    with pytest.raises(ValueError):
        pd_common.validate_pdb_filename(name)


def test_pdb_filename_accepts_normal():
    assert pd_common.validate_pdb_filename("7LHT_clean.pdb") == "7LHT_clean.pdb"


@pytest.mark.parametrize("pid", ["", "ABCDE", "../x", "7LH;", "LHT7"])
def test_pdb_id_rejects_bad(pid):
    with pytest.raises(ValueError):
        pd_common.validate_pdb_id(pid)


def test_node_config_validation(monkeypatch):
    monkeypatch.setenv("PD_T_HOST", "10.0.0.1; reboot")
    monkeypatch.setenv("PD_T_USER", "archuser")
    with pytest.raises(ValueError):
        pd_common.node_config("T")
    monkeypatch.delenv("PD_T_HOST")
    with pytest.raises(RuntimeError):
        pd_common.node_config("T")


def test_remote_quoting():
    assert mp.q("a b; rm -rf /") == "'a b; rm -rf /'"
    assert mp.remote_path("~/.local/bin/micromamba").startswith('"$HOME"/')
    assert mp.remote_path("/opt/x y") == "'/opt/x y'"


def test_gnina_cmd_is_hardened():
    cmd = mp.gnina_docker_cmd("0", "--version", "run_20260101_000000", "LRRK2")
    assert "--cap-drop ALL" in cmd and "no-new-privileges" in cmd and ":ro" in cmd
    assert "runs/run_20260101_000000/LRRK2" in cmd


def test_bbb_rules():
    assert cns_filter.passes_bbb_rules(Chem.MolFromSmiles("CCCCc1ccccc1N"))  # small lipophilic
    assert not cns_filter.passes_bbb_rules(Chem.MolFromSmiles("OC(=O)C(O)C(O)C(O)C(O)CO"))  # polar
    assert not cns_filter.passes_bbb_rules(None)


def test_cns_filter_end_to_end(tmp_path):
    inp = tmp_path / "in.csv"
    pd.DataFrame({"ligand_id": ["../evil", "ok", "ok", "bad"],
                  "smiles": ["CCCCc1ccccc1N", "CCCCc1ccccc1C", "CCCCc1ccccc1C",
                             "not_a_smiles"]}).to_csv(inp, index=False)
    n = cns_filter.process_library(str(inp), str(tmp_path / "o.sdf"), str(tmp_path / "o.csv"))
    out = pd.read_csv(tmp_path / "o.csv")
    assert n == 2 and out["ligand_id"].is_unique
    assert not any("/" in i for i in out["ligand_id"])


def test_cns_filter_requires_columns(tmp_path):
    inp = tmp_path / "in.csv"
    pd.DataFrame({"foo": [1]}).to_csv(inp, index=False)
    with pytest.raises(ValueError):
        cns_filter.load_library(str(inp))


def test_extract_top_hits_dedupes_and_handles_empty(tmp_path):
    filt = tmp_path / "f.csv"
    pd.DataFrame({"ligand_id": ["a", "b"], "smiles": ["C", "CC"]}).to_csv(filt, index=False)
    poses = [
        {"ligand_id": "a", "CNNscore": 0.7, "CNNaffinity": 5.0, "pose_source": "gnina_dock"},
        {"ligand_id": "a", "CNNscore": 0.9, "CNNaffinity": 6.0, "pose_source": "diffdock_rescored"},
        {"ligand_id": "b", "CNNscore": 0.1, "CNNaffinity": 2.0, "pose_source": "gnina_dock"},
    ]
    out = tmp_path / "h.csv"
    assert mp.extract_top_hits(poses, str(filt), str(out)) == 1
    row = pd.read_csv(out).iloc[0]
    assert row["ligand_id"] == "a" and row["CNNscore"] == 0.9 and row["smiles"] == "C"
    assert mp.extract_top_hits([], str(filt), str(out)) == 0  # previously KeyError


def test_admet_filters():
    df = pd.DataFrame({"BBB_Martins": [0.9, 0.5], "hERG": [0.1, 0.1], "DILI": [0.2, 0.2]})
    assert len(mp.apply_admet_filters(df)) == 1
    with pytest.raises(RuntimeError):
        mp.apply_admet_filters(pd.DataFrame({"hERG": [0.1]}))


def test_drop_long_gaps():
    class F:
        missingResidues = {(0, 5): ["A"] * 3, (0, 50): ["A"] * 40}
    f = F()
    prep_targets.drop_long_gaps(f)
    assert list(f.missingResidues) == [(0, 5)]


def test_chembl_library_schema_and_bbb_consistency():
    """The curated ChEMBL library must be pipeline-ready:
    required columns, valid SMILES, and passes_bbb consistent with cns_filter
    BBB rules computed from SMILES via RDKit (mirrors cns_filter exactly)."""
    import csv
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    lib = repo_root / "data" / "chembl_lrrk2_gba1_library.csv"
    assert lib.is_file(), f"curated library not found: {lib}"
    rows = list(csv.DictReader(lib.open()))
    assert rows, "library is empty"
    required = {"ligand_id", "smiles", "target", "passes_bbb", "is_reference"}
    assert required <= set(rows[0].keys()), f"missing columns: {required - set(rows[0].keys())}"
    assert all(r["target"] in {"LRRK2", "GBA1"} for r in rows)
    assert all(r["smiles"] for r in rows), "every compound needs a SMILES"

    # BBB flag must agree with the same rules cns_filter applies to SMILES.
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        have_rdkit = True
    except ImportError:
        have_rdkit = False

    for r in rows:
        assert pd_common.sanitize_id(r["ligand_id"]) == r["ligand_id"], f"unsafe ligand_id: {r['ligand_id']!r}"
        if have_rdkit:
            mol = Chem.MolFromSmiles(r["smiles"])
            assert mol is not None, f"invalid SMILES for {r['ligand_id']}: {r['smiles']!r}"
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            rtb = rdMolDescriptors.CalcNumRotatableBonds(mol)
            expected = (
                mw <= 450.0
                and 1.0 <= logp <= 4.5
                and tpsa < 90.0
                and hbd <= 3
                and hba <= 7
                and rtb <= 8
                and mol.GetNumHeavyAtoms() <= 100
            )
            assert (r["passes_bbb"] == "True") == expected, (
                f"passes_bbb mismatch for {r['ligand_id']}: stored={r['passes_bbb']} computed={expected}"
            )

    n_bbb = sum(1 for r in rows if r["passes_bbb"] == "True")
    assert n_bbb > 0, "library should contain BBB-penetrant compounds"
    # Reference compounds expected to be present + BBB-passing.
    for chembl_id in ("CHEMBL4098877", "CHEMBL153479", "CHEMBL603469"):
        ref = [r for r in rows if r["chembl_id"] == chembl_id]
        assert ref, f"reference {chembl_id} absent from library"
        assert ref[0]["passes_bbb"] == "True", f"{chembl_id} must pass BBB filter"
    # Isofagomine/afegostat: present (GBA1 chaperone) but too polar for BBB.
    iso = [r for r in rows if r["chembl_id"] == "CHEMBL206468"]
    assert iso, "reference isofagomine/afegostat (CHEMBL206468) absent from library"
    assert iso[0]["passes_bbb"] == "False", "isofagomine should fail BBB filter (polar)"
    print(f"[check] library {len(rows)} compounds, {n_bbb} BBB-passing, "
          f"MLi-2/ambroxol/lestaurtinib present+BBB-pass, isofagomine present+BBB-fail")


def test_stage4_triage_known_compounds(tmp_path):
    """Simulate Stage 4 hit triage against known LRRK2/GBA1 compounds.

    Builds synthetic GNINA poses + ADMET-AI output for a handful of curated
    compounds, then runs extract_top_hits + apply_admet_filters and asserts:
      * best pose per ligand_id is kept (CNNscore dedup)
      * CNNscore >= 0.65 threshold is enforced
      * BBB >= 0.80, hERG <= 0.30, DILI <= 0.40 filters are applied
      * a known BBB-penetrant LRRK2 inhibitor (MLi-2) survives; a polar
        non-CNS compound is dropped.
    """
    # Filtered (CNS-passed) SMILES for two known compounds.
    filt = tmp_path / "f.csv"
    pd.DataFrame({
        "ligand_id": ["LRRK2_CHEMBL4098877", "GBA1_CHEMBL6329", "LRRK2_DUMMY_POLAR"],
        "smiles": [
            "C[C@@H]1CN(c2cc(-c3n[nH]c4ccc(OC5(C)CC5)cc34)ncn2)C[C@H](C)O1",  # MLi-2
            "Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl",  # ambroxol
            "OC(=O)C(O)C(O)C(O)C(O)CO",  # polar non-CNS (glucose-like)
        ],
    }).to_csv(filt, index=False)

    # Synthetic poses: MLi-2 has two poses (keep best=0.9), ambroxol one good pose,
    # polar compound one sub-threshold pose (0.40).
    poses = [
        {"ligand_id": "LRRK2_CHEMBL4098877", "CNNscore": 0.70, "CNNaffinity": -5.0, "pose_source": "gnina_dock"},
        {"ligand_id": "LRRK2_CHEMBL4098877", "CNNscore": 0.90, "CNNaffinity": -7.0, "pose_source": "diffdock_rescored"},
        {"ligand_id": "GBA1_CHEMBL6329", "CNNscore": 0.80, "CNNaffinity": -6.0, "pose_source": "gnina_dock"},
        {"ligand_id": "LRRK2_DUMMY_POLAR", "CNNscore": 0.40, "CNNaffinity": -2.0, "pose_source": "gnina_dock"},
    ]
    hits = tmp_path / "h.csv"
    n = mp.extract_top_hits(poses, str(filt), str(hits))
    assert n == 2, f"expected 2 hits above threshold, got {n}"
    h = pd.read_csv(hits)
    # Best pose per ligand kept: MLi-2 should be 0.90 not 0.70.
    ml2 = h[h["ligand_id"] == "LRRK2_CHEMBL4098877"].iloc[0]
    assert ml2["CNNscore"] == 0.90, f"MLi-2 best pose not kept: {ml2['CNNscore']}"
    # Polar compound dropped by threshold.
    assert "LRRK2_DUMMY_POLAR" not in set(h["ligand_id"])

    # Synthetic ADMET-AI output: MLi-2 passes neuro-ADMET; ambroxol fails hERG.
    admet = tmp_path / "admet.csv"
    pd.DataFrame({
        "BBB_Martins": [0.90, 0.85, 0.50],
        "hERG": [0.10, 0.50, 0.10],
        "DILI": [0.20, 0.20, 0.20],
    }, index=["LRRK2_CHEMBL4098877", "GBA1_CHEMBL6329", "LRRK2_DUMMY_POLAR"]).to_csv(admet)
    admet_df = pd.read_csv(admet)
    admet_df.insert(0, "ligand_id", ["LRRK2_CHEMBL4098877", "GBA1_CHEMBL6329", "LRRK2_DUMMY_POLAR"])
    safe = mp.apply_admet_filters(admet_df)
    assert len(safe) == 1, f"expected 1 safe lead, got {len(safe)}"
    assert safe.iloc[0]["ligand_id"] == "LRRK2_CHEMBL4098877", f"expected MLi-2 safe, got {safe.iloc[0]['ligand_id']}"
    print("[check] stage-4 triage: 2 hits above threshold, 1 safe lead (MLi-2), ambroxol dropped on hERG")


def test_stage4_admet_column_validation():
    """apply_admet_filters must require all three ADMET columns."""
    for missing in [
        {"hERG": [0.1], "DILI": [0.2]},  # missing BBB_Martins
        {"BBB_Martins": [0.9], "DILI": [0.2]},  # missing hERG
        {"BBB_Martins": [0.9], "hERG": [0.1]},  # missing DILI
    ]:
        with pytest.raises(RuntimeError):
            mp.apply_admet_filters(pd.DataFrame(missing))


def test_pdb_construct_verification():
    """PDB construct verification (README_REVIEW D / A10).

    Verified against RCSB core APIs:
      * 7LHT — full-length human LRRK2 dimer, cryo-EM 3.5 A (Cell 2021,
        DOI:10.1016/j.cell.2021.05.004). Wild-type sequence; G2019S
        specificity comes from ChEMBL assay data, not the PDB construct.
      * 2V3D — human acid beta-glucosidase (GBA1), X-ray 1.96 A
        (J Biol Chem 2007, DOI:10.1074/JBC.M705005200).
      * 6N3F is NOT GBA1 — it is the Tat-SF1/SF3b1 complex (J Biol Chem
        2019). The README's 6N3F mapping was replaced with 2V3D.
    """
    targets = pd_common.TARGETS
    assert targets["LRRK2"]["pdb_id"] == "7LHT"
    assert targets["GBA1"]["pdb_id"] == "2V3D"
    assert targets["LRRK2"]["ph"] == 7.4  # cytosolic
    assert targets["GBA1"]["ph"] == 5.5  # lysosomal
    # 6N3F must not be referenced as GBA1 anywhere in config.
    assert "6N3F" not in {c["pdb_id"] for c in targets.values()}
    for _name, cfg in targets.items():
        assert pd_common.validate_pdb_id(cfg["pdb_id"]) == cfg["pdb_id"].upper()
    print("[check] PDB constructs: LRRK2=7LHT (WT full-length), GBA1=2V3D "
          "(human acid beta-glucosidase); 6N3F rejected (not GBA1)")


def test_check_pipeline_report_flat(tmp_path, monkeypatch, capsys):
    """check_pipeline.report() on a validated-candidates CSV (no CNNscore column)."""
    import check_pipeline as cp

    df = pd.DataFrame({
        "ligand_id": ["LRRK2_CHEMBL4098877", "GBA1_CHEMBL153479", "LRRK2_X"],
        "target": ["LRRK2", "GBA1", "LRRK2"],
        "chembl_id": ["CHEMBL4098877", "CHEMBL153479", "CHEMBL_X"],
        "BBB_Martins": [0.90, 0.85, 0.50],
        "hERG": [0.10, 0.10, 0.10],
        "DILI": [0.20, 0.20, 0.20],
    })
    p = tmp_path / "candidates.csv"
    df.to_csv(p, index=False)
    safe = cp.report(df, p.name)
    assert len(safe) == 2  # LRRK2_X fails BBB (0.50 < 0.80)
    assert set(safe["ligand_id"]) == {"LRRK2_CHEMBL4098877", "GBA1_CHEMBL153479"}
    captured = capsys.readouterr()
    assert "Safe leads (all three pass): 2" in captured.out
    assert "MLi-2 (LRRK2) (CHEMBL4098877): present" in captured.out


def test_check_pipeline_report_with_cnnscore(tmp_path):
    """check_pipeline.report() ranks top leads by CNNscore when present."""
    import check_pipeline as cp

    df = pd.DataFrame({
        "ligand_id": ["LRRK2_CHEMBL4098877", "LRRK2_Y", "GBA1_CHEMBL153479"],
        "target": ["LRRK2", "LRRK2", "GBA1"],
        "chembl_id": ["CHEMBL4098877", "CHEMBL_Y", "CHEMBL153479"],
        "CNNscore": [0.90, 0.70, 0.80],
        "CNNaffinity": [-7.0, -5.0, -6.0],
        "BBB_Martins": [0.90, 0.90, 0.85],
        "hERG": [0.10, 0.10, 0.50],
        "DILI": [0.20, 0.20, 0.20],
        "smiles": ["C", "CC", "CCC"],
    })
    p = tmp_path / "safe_leads.csv"
    df.to_csv(p, index=False)
    safe = cp.report(df, p.name)
    assert len(safe) == 2  # GBA1_CHEMBL153479 fails hERG (0.50 > 0.30)
    top = safe.sort_values("CNNscore", ascending=False)
    assert top.iloc[0]["ligand_id"] == "LRRK2_CHEMBL4098877"
    assert top.iloc[0]["CNNscore"] == 0.90


def test_check_pipeline_empty(tmp_path):
    """check_pipeline.report() handles an empty CSV without crashing."""
    import check_pipeline as cp

    p = tmp_path / "empty.csv"
    pd.DataFrame(columns=["ligand_id", "target", "BBB_Martins", "hERG", "DILI"]).to_csv(p, index=False)
    df = cp.load_csv(p)
    assert df.empty


def test_run_id_paths_are_versioned(tmp_path, monkeypatch):
    """Run directories are versioned under {SHARED_ROOT}/runs/{run_id}/{target}/."""
    import master_pipeline as mp

    monkeypatch.setattr(mp, "SHARED_ROOT", str(tmp_path))
    rid = "20261024_093000"
    target = "LRRK2"

    assert mp.run_root(rid) == os.path.join(str(tmp_path), "runs", rid)
    assert mp.target_run_dir(rid, target) == os.path.join(str(tmp_path), "runs", rid, target)
    d = mp.stage_dir(rid, target, "02_cns")
    assert d == os.path.join(str(tmp_path), "runs", rid, target, "02_cns")
    assert os.path.isdir(d)
    assert mp.manifest_path(rid, target) == os.path.join(str(tmp_path), "runs", rid, target, "manifest.json")


def test_manifest_round_trip(tmp_path, monkeypatch):
    """Manifest save/load preserves run metadata and stage completion."""
    import master_pipeline as mp

    monkeypatch.setattr(mp, "SHARED_ROOT", str(tmp_path))
    rid = "20261024_093000"
    target = "GBA1"

    m = mp.load_manifest(rid, target)
    assert m["run_id"] == rid
    assert m["target"] == target
    assert m["status"] == "running"
    assert not mp.stage_done(m, "cns_filter")

    mp.mark_stage_done(m, "cns_filter", {"cns_csv": "x.csv", "ligands": 5})
    assert mp.stage_done(m, "cns_filter")
    assert m["stages"]["cns_filter"]["detail"]["ligands"] == 5

    # Reload from disk and verify persistence
    m2 = mp.load_manifest(rid, target)
    assert mp.stage_done(m2, "cns_filter")
    assert m2["stages"]["cns_filter"]["detail"]["ligands"] == 5


def test_now_str_format():
    """now_str() returns a compact timestamp suitable for run IDs."""
    import master_pipeline as mp

    s = mp.now_str()
    assert len(s) == 15  # YYYYMMDD_HHMMSS
    assert s[8] == "_"
    assert s[:8].isdigit()
    assert s[9:].isdigit()
