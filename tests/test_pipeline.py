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
    cmd = mp.gnina_docker_cmd("0", "--version")
    assert "--cap-drop ALL" in cmd and "no-new-privileges" in cmd and ":ro" in cmd


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
    # Reference compound MLi-2 must be present and BBB-passing.
    ml2 = [r for r in rows if r["chembl_id"] == "CHEMBL4098877"]
    assert ml2, "reference MLi-2 (CHEMBL4098877) absent from library"
    assert ml2[0]["passes_bbb"] == "True", "MLi-2 must pass BBB filter"
    print(f"[check] library {len(rows)} compounds, {n_bbb} BBB-passing, MLi-2 present+BBB-pass")


# ---------------------------------------------------------------- library cleaning / benchmark
import json  # noqa: E402

import numpy as np  # noqa: E402

import benchmark as bm  # noqa: E402
import build_chembl_library as bcl  # noqa: E402


def test_non_human_assays_detected():
    assert bcl.is_non_human_assay("Inhibitory activity against Agrobacterium sp. Beta-glucosidase")
    assert bcl.is_non_human_assay("sweet almond beta-glucosidase")
    assert not bcl.is_non_human_assay("Inhibition of recombinant human GBA1")


def test_shipped_library_is_clean():
    lib = pd.read_csv(bm.DEFAULT_LIBRARY)
    assert not lib["assay_description"].map(bcl.is_non_human_assay).any()
    assert {"mode_hint", "source_class"} <= set(lib.columns)
    assert not (lib["pref_name"] == lib["chembl_id"]).any()


def test_metrics_perfect_and_random():
    y = np.array([1] * 10 + [0] * 90)
    assert bm.roc_auc(y, np.r_[np.ones(10), np.zeros(90)]) == 1.0
    assert bm.enrichment_factor(y, np.r_[np.ones(10), np.zeros(90)], 0.10) == pytest.approx(10.0)
    rng = np.random.default_rng(1)
    assert abs(bm.roc_auc(y, rng.random(100)) - 0.5) < 0.2
    # ties are scored pessimistically for AP
    assert bm.average_precision(y, np.zeros(100)) < 0.2


def test_verdict_requires_ci_above_random():
    assert "No detectable" in bm.verdict(0.75, 0.48, 0.5)
    assert "trivial property baseline" in bm.verdict(0.75, 0.6, 0.74)
    assert "Moderate" in bm.verdict(0.75, 0.6, 0.5)


ACT = ["CCCCc1ccccc1N", "CCCCc1ccccc1C", "CCCCCc1ccccc1O", "CCCCc1ccc(F)cc1", "CCCCCc1ccccc1N"]
POOL = ["CCOC(=O)C1CCNCC1C", "CC(C)NC(=O)c1ccncc1", "OC1CCCCC1CC(=O)N", "CNC(=O)C1CCCN1C", "CCN(CC)C(=O)c1ccccn1",
        "CC1CCN(C(=O)C2CC2)CC1", "NC(=O)C1CCCCC1C", "COC(=O)c1ccncc1C"]


def test_decoys_dissimilar_and_unique():
    from rdkit import Chem as C
    active_mols = [C.MolFromSmiles(s) for s in ACT]
    chosen, pool_size = bm.select_decoys(active_mols, POOL, ratio=1, seed=0)
    assert pool_size > 0 and len(chosen) == len(set(chosen)) and len(chosen) <= len(ACT)
    assert not set(chosen) & set(bm.standardize(s)[0] for s in ACT)


def _write_scored_sdf(path, ids_scores):
    w = Chem.SDWriter(path)
    for name, cnn in ids_scores:
        m = Chem.MolFromSmiles("CC")
        m.SetProp("_Name", name)
        m.SetProp("CNNscore", str(cnn))
        m.SetProp("CNNaffinity", str(cnn * 6))
        m.SetProp("minimizedAffinity", str(-cnn * 8))
        w.write(m)
    w.close()


def test_prepare_and_evaluate_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "SHARED_ROOT", str(tmp_path))
    lib = pd.DataFrame({
        "ligand_id": [f"LRRK2_{i}" for i in range(5)] + ["GBA1_x"],
        "smiles": ACT + ["CCO"], "target": ["LRRK2"] * 5 + ["GBA1"], "pchembl": [8.0] * 6,
        "mode_hint": ["inhibitor_or_binder"] * 6, "source_class": ["dedicated_assay"] * 6})
    lib_path = tmp_path / "lib.csv"
    lib.to_csv(lib_path, index=False)
    pool_path = tmp_path / "pool.csv"
    pd.DataFrame({"smiles": POOL}).to_csv(pool_path, index=False)
    outdir = tmp_path / "bm"

    class A:
        target, library, decoy_pool = "LRRK2", str(lib_path), str(pool_path)
        decoy_ratio, min_pchembl, mode, source_class, seed = 1, 6.0, "any", "any", 0
        outdir_ = None
    A.outdir = str(outdir)
    bm.cmd_prepare(A)
    labels = pd.read_csv(outdir / "labels.csv")
    assert labels["label"].sum() == 5 and (labels["label"] == 0).sum() >= 2
    assert json.load(open(outdir / "manifest.json"))["library_sha256"]

    # Perfect fake docking: actives score high, decoys low.
    _write_scored_sdf(str(outdir / "scored.sdf"),
                      [(r.id, 0.9 if r.label else 0.1) for r in labels.itertuples()])

    class E:
        target, bootstrap, seed = "LRRK2", 50, 0
    E.outdir = str(outdir)
    bm.cmd_evaluate(E)
    res = json.load(open(outdir / "results.json"))["results"]
    assert res["CNNscore"]["roc_auc"] == 1.0
    assert (outdir / "report.md").read_text().startswith("# Benchmark report")


def test_outdir_must_be_under_shared_root(monkeypatch, tmp_path):
    monkeypatch.setattr(bm, "SHARED_ROOT", str(tmp_path))
    with pytest.raises(SystemExit):
        bm.check_under_root("/etc")


def test_bundled_nci_pool_and_decoy_quality(tmp_path):
    pool = bm.bundled_nci_pool()
    assert len(pool) > 1000
    from rdkit import Chem as C
    chosen, _ = bm.select_decoys([C.MolFromSmiles(s) for s in ACT], pool, ratio=3, seed=0)
    assert len(chosen) == 15
    df = pd.DataFrame({"smiles": ACT + chosen, "label": [1] * len(ACT) + [0] * len(chosen)})
    res = bm.decoy_quality(df, n_boot=20)
    assert "Baseline: logP" in res and all(0 <= r["roc_auc"] <= 1 for r in res.values())


def test_embed_survives_bad_molecule(tmp_path):
    failed = bm.embed_to_sdf([{"id": "ok", "smiles": "CCO"}, {"id": "odd", "smiles": "[H][H]"}],
                             str(tmp_path / "o.sdf"), 0)
    assert "ok" not in failed


def test_convert_decoys(tmp_path):
    import gzip

    import convert_decoys as cd
    src = tmp_path / "decoys_final.ism.gz"
    with gzip.open(src, "wt") as fh:
        fh.write("CCOC(=O)C1CCNCC1 ZINC1\nnot_a_smiles ZINC2\nOCC ZINC3\nOCC ZINC4\n"
                 "CCCCc1ccccc1N ZINC5\nCCN.Cl ZINC6\n")
    lib = tmp_path / "lib.csv"
    pd.DataFrame({"smiles": ["CCCCc1ccccc1N"]}).to_csv(lib, index=False)
    n, stats = cd.convert(str(src), str(tmp_path / "out.csv"), str(lib))
    out = pd.read_csv(tmp_path / "out.csv")
    assert n == 3 and list(out.columns) == ["smiles", "id"]
    assert stats == {"invalid": 1, "duplicate": 1, "matches_active": 1}
    assert "CCN" in out["smiles"].tolist()  # largest fragment kept
