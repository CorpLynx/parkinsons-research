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
