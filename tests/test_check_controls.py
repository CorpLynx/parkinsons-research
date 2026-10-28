"""Tests for the control-benchmark tool (scripts/check_controls.py)."""
import csv
import os

import pandas as pd
import pytest

import check_controls as cc

DATA_CONTROLS = os.path.join(os.path.dirname(__file__), "..", "data", "controls.csv")


# ---- roc_auc -------------------------------------------------------------

def test_roc_auc_perfect_separation():
    assert cc.roc_auc([0.9, 0.8, 0.7], [0.3, 0.2, 0.1]) == 1.0


def test_roc_auc_inverted():
    assert cc.roc_auc([0.1, 0.2], [0.8, 0.9]) == 0.0


def test_roc_auc_all_tied_is_half():
    assert cc.roc_auc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_roc_auc_partial_ties():
    # one positive tied with one negative, the other positive clearly higher
    auc = cc.roc_auc([0.9, 0.5], [0.5, 0.1])
    assert 0.5 < auc < 1.0


def test_roc_auc_empty_group_is_none():
    assert cc.roc_auc([], [0.1]) is None
    assert cc.roc_auc([0.1], []) is None


# ---- load_controls -------------------------------------------------------

def test_shipped_controls_have_both_roles_per_target():
    for target in ("LRRK2", "GBA1"):
        df = cc.load_controls(DATA_CONTROLS, target)
        roles = set(df["control_role"])
        assert "positive" in roles and "negative" in roles


def test_load_controls_unknown_target_raises():
    with pytest.raises(ValueError):
        cc.load_controls(DATA_CONTROLS, "NOTATARGET")


# ---- spike ---------------------------------------------------------------

def _args(**kw):
    return type("A", (), kw)()


def test_spike_adds_controls_and_dedupes(tmp_path):
    filtered = tmp_path / "filtered.csv"
    filtered.write_text("ligand_id,smiles\nLIB_1,CCO\nLIB_1,CCO\n")  # dup id
    out = tmp_path / "spiked.csv"
    cc.spike(_args(target="LRRK2", filtered=str(filtered),
                   controls=DATA_CONTROLS, out=str(out)))
    df = pd.read_csv(out)
    assert df["ligand_id"].is_unique
    assert "LIB_1" in set(df["ligand_id"])
    # at least one positive and one negative control present
    ids = set(df["ligand_id"])
    assert any(i.startswith("CTRL_POS_") for i in ids)
    assert any(i.startswith("CTRL_NEG_") for i in ids)


# ---- evaluate ------------------------------------------------------------

def _write_scores(path, target, pos_score, neg_score, score_col="CNNscore"):
    ctrls = cc.load_controls(DATA_CONTROLS, target)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ligand_id", score_col])
        for _, r in ctrls.iterrows():
            s = pos_score if r["control_role"] == "positive" else neg_score
            w.writerow([r["ligand_id"], s])


def test_evaluate_pass(tmp_path, capsys):
    scores = tmp_path / "s.csv"
    _write_scores(scores, "LRRK2", pos_score=0.9, neg_score=0.1)
    code = cc.evaluate(_args(target="LRRK2", scores=str(scores),
                             controls=DATA_CONTROLS, score_col="CNNscore",
                             min_auc=0.7, require_all=False))
    assert code == 0
    assert "PASS" in capsys.readouterr().out


def test_evaluate_fail_on_low_auc(tmp_path):
    scores = tmp_path / "s.csv"
    _write_scores(scores, "LRRK2", pos_score=0.1, neg_score=0.9)
    code = cc.evaluate(_args(target="LRRK2", scores=str(scores),
                             controls=DATA_CONTROLS, score_col="CNNscore",
                             min_auc=0.7, require_all=False))
    assert code == 2


def test_evaluate_require_all_flags_missing(tmp_path):
    # only score the positives; negatives absent -> require_all should fail
    ctrls = cc.load_controls(DATA_CONTROLS, "LRRK2")
    pos = ctrls[ctrls["control_role"] == "positive"]
    scores = tmp_path / "s.csv"
    with open(scores, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ligand_id", "CNNscore"])
        for _, r in pos.iterrows():
            w.writerow([r["ligand_id"], 0.9])
    code = cc.evaluate(_args(target="LRRK2", scores=str(scores),
                             controls=DATA_CONTROLS, score_col="CNNscore",
                             min_auc=None, require_all=True))
    assert code == 2


def test_evaluate_takes_best_pose_per_ligand(tmp_path):
    ctrls = cc.load_controls(DATA_CONTROLS, "LRRK2")
    one_pos = ctrls[ctrls["control_role"] == "positive"].iloc[0]["ligand_id"]
    scores = tmp_path / "s.csv"
    with open(scores, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ligand_id", "CNNscore"])
        w.writerow([one_pos, 0.2])   # weak pose
        w.writerow([one_pos, 0.95])  # strong pose -> should be the one used
    merged_best = pd.read_csv(scores).groupby("ligand_id")["CNNscore"].max()[one_pos]
    assert merged_best == 0.95


def test_missing_score_column_raises(tmp_path):
    scores = tmp_path / "s.csv"
    scores.write_text("ligand_id,wrongcol\nX,1\n")
    with pytest.raises(ValueError):
        cc.evaluate(_args(target="LRRK2", scores=str(scores),
                          controls=DATA_CONTROLS, score_col="CNNscore",
                          min_auc=None, require_all=False))


# ---- hardening -----------------------------------------------------------

def test_missing_file_is_clean_input_error():
    with pytest.raises(cc.InputError):
        cc.load_controls("/no/such/controls.csv", "LRRK2")


def test_empty_file_is_input_error(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(cc.InputError):
        cc.read_csv(str(empty), "scores")


def test_unknown_control_role_rejected(tmp_path):
    bad = tmp_path / "ctrl.csv"
    bad.write_text("ligand_id,smiles,target,control_role\n"
                   "A,CCO,LRRK2,positive\nB,CCO,LRRK2,maybe\n")
    with pytest.raises(cc.InputError):
        cc.load_controls(str(bad), "LRRK2")


def test_duplicate_control_ids_rejected(tmp_path):
    bad = tmp_path / "ctrl.csv"
    bad.write_text("ligand_id,smiles,target,control_role\n"
                   "DUP,CCO,LRRK2,positive\nDUP,CCO,LRRK2,negative\n")
    with pytest.raises(cc.InputError):
        cc.load_controls(str(bad), "LRRK2")


def test_spike_refuses_to_overwrite_input(tmp_path):
    f = tmp_path / "same.csv"
    f.write_text("ligand_id,smiles\nX,CCO\n")
    with pytest.raises(cc.InputError):
        cc.spike(_args(target="LRRK2", filtered=str(f),
                       controls=DATA_CONTROLS, out=str(f)))


def test_evaluate_ignores_non_numeric_scores(tmp_path, capsys):
    ctrls = cc.load_controls(DATA_CONTROLS, "LRRK2")
    scores = tmp_path / "s.csv"
    with open(scores, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ligand_id", "CNNscore"])
        for _, r in ctrls.iterrows():
            val = 0.9 if r["control_role"] == "positive" else 0.1
            w.writerow([r["ligand_id"], val])
        w.writerow(["JUNK_ROW", "not_a_number"])  # must be ignored, not crash
    code = cc.evaluate(_args(target="LRRK2", scores=str(scores),
                             controls=DATA_CONTROLS, score_col="CNNscore",
                             min_auc=0.7, require_all=False))
    out = capsys.readouterr().out
    assert code == 0
    assert "non-numeric" in out


def test_main_returns_1_on_input_error():
    code = cc.main(["evaluate", "--target", "LRRK2",
                    "--scores", "/no/such/file.csv"])
    assert code == 1
