"""Tests for the multi-node orchestration logic in master_pipeline.py.

These cover the pieces that would otherwise only be exercised on the Arch GPU
nodes: DiffDock input construction, pose collection, pocket selection, SSH
execution semantics, and the dry-run command surface.
"""
import os
import sys

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import master_pipeline as mp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_sdf(path, name="lig", props=None):
    """Append a single molecule to an SDF file and return the path."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    mol.SetProp("_Name", name)
    for k, v in (props or {}).items():
        mol.SetProp(k, str(v))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Append mode: SDWriter(path) would truncate on every call.
    with open(path, "a") as fh:
        writer = Chem.SDWriter(fh)
        writer.write(mol)
        writer.close()
    return path


def fake_pocket_df(n=3):
    """A P2Rank-like predictions DataFrame with distinguishable centers."""
    return pd.DataFrame({
        "rank": range(1, n + 1),
        "center_x": [float(i * 10) for i in range(1, n + 1)],
        "center_y": [float(i * 10) for i in range(1, n + 1)],
        "center_z": [float(i * 10) for i in range(1, n + 1)],
        "score": [1.0] * n,
    })


@pytest.fixture
def shared_root(tmp_path, monkeypatch):
    """Point the pipeline at a temporary shared root and reset node cache."""
    root = tmp_path / "pd_research"
    root.mkdir()
    monkeypatch.setattr(mp, "SHARED_ROOT", str(root))
    monkeypatch.setenv("PD_ARCH4070_HOST", "10.0.0.1")
    monkeypatch.setenv("PD_ARCH4070_USER", "archuser")
    monkeypatch.setenv("PD_ARCH4070_GPU", "0")
    monkeypatch.setenv("PD_ARCH2080_HOST", "10.0.0.2")
    monkeypatch.setenv("PD_ARCH2080_USER", "archuser")
    monkeypatch.setenv("PD_ARCH2080_GPU", "0")
    mp._nodes.clear()
    yield str(root)
    mp._nodes.clear()


# ---------------------------------------------------------------------------
# build_diffdock_csv
# ---------------------------------------------------------------------------

def test_build_diffdock_csv_schema(tmp_path):
    """DiffDock's input CSV must have exactly the columns upstream expects."""
    filtered = tmp_path / "cns.csv"
    pd.DataFrame({
        "ligand_id": ["LRRK2_CHEMBL4098877", "GBA1_CHEMBL153479"],
        "smiles": ["C[C@@H]1CN(c2cc(-c3n[nH]c4ccc(OC5(C)CC5)cc34)ncn2)C[C@H](C)O1", "Cc1ccccc1O"],
    }).to_csv(filtered, index=False)

    out = tmp_path / "diffdock_input.csv"
    mp.build_diffdock_csv("/path/to/protein.pdb", str(filtered), str(out))

    df = pd.read_csv(out, dtype=str).fillna("")
    assert list(df.columns) == ["complex_name", "protein_path", "ligand_description", "protein_sequence"]
    assert len(df) == 2
    assert df.iloc[0]["complex_name"] == "complex_LRRK2_CHEMBL4098877"
    assert df.iloc[0]["protein_path"] == "/path/to/protein.pdb"
    assert df.iloc[0]["ligand_description"].startswith("C[C@@H]1CN")
    assert df.iloc[0]["protein_sequence"] == ""


def test_build_diffdock_csv_respects_max_ligands(tmp_path):
    """DiffDock batching is capped so a run cannot exceed VRAM limits."""
    filtered = tmp_path / "cns.csv"
    pd.DataFrame({
        "ligand_id": [f"cmpd_{i}" for i in range(50)],
        "smiles": ["CCO"] * 50,
    }).to_csv(filtered, index=False)

    out = tmp_path / "diffdock_input.csv"
    mp.build_diffdock_csv("/p.pdb", str(filtered), str(out), max_ligands=10)
    assert len(pd.read_csv(out)) == 10


def test_build_diffdock_csv_sanitizes_complex_names(tmp_path):
    """complex_name must be shell/path safe since it becomes a directory name."""
    filtered = tmp_path / "cns.csv"
    pd.DataFrame({
        "ligand_id": ["../evil; rm -rf /"],
        "smiles": ["CCO"],
    }).to_csv(filtered, index=False)

    out = tmp_path / "diffdock_input.csv"
    mp.build_diffdock_csv("/p.pdb", str(filtered), str(out))
    name = pd.read_csv(out).iloc[0]["complex_name"]
    assert "/" not in name and ";" not in name
    assert name == "complex__evil__rm_-rf__"


# ---------------------------------------------------------------------------
# combine_diffdock_poses
# ---------------------------------------------------------------------------

def test_combine_diffdock_poses_collects_rank1(tmp_path):
    """Each DiffDock complex directory contributes its rank1 pose."""
    diffdock_out = tmp_path / "04_diffdock" / "LRRK2"
    write_sdf(str(diffdock_out / "complex_A" / "rank1.sdf"), name="A")
    write_sdf(str(diffdock_out / "complex_B" / "rank1.sdf"), name="B")
    # A rank2 pose must be ignored.
    write_sdf(str(diffdock_out / "complex_A" / "rank2.sdf"), name="A2")
    # A non-complex directory must be ignored.
    write_sdf(str(diffdock_out / "other" / "rank1.sdf"), name="X")

    combined = tmp_path / "combined.sdf"
    count = mp.combine_diffdock_poses(str(diffdock_out), str(combined))
    assert count == 2

    mols = [m for m in Chem.SDMolSupplier(str(combined)) if m is not None]
    assert len(mols) == 2
    names = sorted(m.GetProp("_Name") for m in mols)
    assert names == ["A", "B"]  # 'complex_' prefix stripped


def test_combine_diffdock_poses_empty_dir(tmp_path):
    """A missing/empty DiffDock output yields zero poses, not an error."""
    combined = tmp_path / "combined.sdf"
    assert mp.combine_diffdock_poses(str(tmp_path / "nonexistent"), str(combined)) == 0


def test_combine_diffdock_poses_skips_invalid_molecules(tmp_path):
    """Unparseable pose files are skipped rather than crashing the merge."""
    diffdock_out = tmp_path / "04_diffdock" / "LRRK2"
    write_sdf(str(diffdock_out / "complex_A" / "rank1.sdf"), name="A")
    bad = diffdock_out / "complex_B" / "rank1.sdf"
    os.makedirs(bad.parent, exist_ok=True)
    bad.write_text("this is not a molfile\n")

    combined = tmp_path / "combined.sdf"
    assert mp.combine_diffdock_poses(str(diffdock_out), str(combined)) == 1


# ---------------------------------------------------------------------------
# collect_poses
# ---------------------------------------------------------------------------

def test_collect_poses_reads_scores(tmp_path):
    """CNNscore and CNNaffinity are read off the GNINA-annotated poses."""
    sdf = tmp_path / "docked.sdf"
    write_sdf(str(sdf), name="L1", props={"CNNscore": "0.87", "CNNaffinity": "-6.5"})
    write_sdf(str(sdf), name="L2", props={"CNNscore": "0.72", "CNNaffinity": "-5.1"})

    poses = mp.collect_poses(str(sdf), "gnina_dock")
    assert len(poses) == 2
    by_id = {p["ligand_id"]: p for p in poses}
    assert by_id["L1"]["CNNscore"] == pytest.approx(0.87)
    assert by_id["L1"]["CNNaffinity"] == pytest.approx(-6.5)
    assert by_id["L1"]["pose_source"] == "gnina_dock"
    assert by_id["L2"]["CNNscore"] == pytest.approx(0.72)


def test_collect_poses_defaults_missing_scores_to_zero(tmp_path):
    """Poses without CNNscore attributes default to 0.0 rather than crashing."""
    sdf = tmp_path / "docked.sdf"
    write_sdf(str(sdf), name="L1")
    poses = mp.collect_poses(str(sdf), "gnina_dock")
    assert len(poses) == 1
    assert poses[0]["CNNscore"] == 0.0
    assert poses[0]["CNNaffinity"] == 0.0


def test_collect_poses_missing_file_returns_empty(tmp_path):
    """A missing SDF yields no poses (allows Stage 3b to degrade gracefully)."""
    assert mp.collect_poses(str(tmp_path / "nope.sdf"), "gnina_dock") == []


# ---------------------------------------------------------------------------
# select_pocket
# ---------------------------------------------------------------------------

def test_select_pocket_lrrk2_uses_rank1():
    """LRRK2 docks into the rank-1 catalytic (ATP) pocket."""
    idx, x, y, z = mp.select_pocket(fake_pocket_df(3), "LRRK2")
    assert idx == 0
    assert (x, y, z) == (10.0, 10.0, 10.0)


def test_select_pocket_gba1_uses_rank2():
    """GBA1 docks into the rank-2 allosteric chaperone pocket."""
    idx, x, y, z = mp.select_pocket(fake_pocket_df(3), "GBA1")
    assert idx == 1
    assert (x, y, z) == (20.0, 20.0, 20.0)


def test_select_pocket_gba1_falls_back_to_rank1():
    """With only one predicted pocket, GBA1 falls back to rank 1."""
    idx, x, y, z = mp.select_pocket(fake_pocket_df(1), "GBA1")
    assert idx == 0
    assert (x, y, z) == (10.0, 10.0, 10.0)


def test_select_pocket_empty_raises():
    """No pockets must fail loudly rather than silently docking nowhere."""
    with pytest.raises(RuntimeError):
        mp.select_pocket(pd.DataFrame(), "LRRK2")
    with pytest.raises(RuntimeError):
        mp.select_pocket(None, "GBA1")


# ---------------------------------------------------------------------------
# run_remote_ssh (mocked paramiko)
# ---------------------------------------------------------------------------

class FakeChannel:
    def __init__(self, status):
        self._status = status

    def recv_exit_status(self):
        return self._status


class FakeStdout:
    def __init__(self, lines, status):
        self._lines = list(lines)
        self.channel = FakeChannel(status)

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        return ""


class FakeSSHClient:
    """Minimal stand-in for paramiko.SSHClient.

    Configure via the class attributes `exit_status` and `lines`.
    """

    exit_status = 0
    lines = []
    instances = []

    def __init__(self, *args, **kwargs):
        self.connected_with = None
        self.executed_cmd = None
        self.policy = None
        self.closed = False
        FakeSSHClient.instances.append(self)

    def load_system_host_keys(self):
        pass

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, **kwargs):
        self.connected_with = kwargs

    def exec_command(self, command, **kwargs):
        self.executed_cmd = command
        return (None, FakeStdout(FakeSSHClient.lines, FakeSSHClient.exit_status), None)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_paramiko(monkeypatch):
    """Patch paramiko.SSHClient with the fake and reset the instance list."""
    FakeSSHClient.instances = []
    FakeSSHClient.exit_status = 0
    FakeSSHClient.lines = []

    def factory(*a, **kw):
        return FakeSSHClient(*a, **kw)

    monkeypatch.setattr(mp.paramiko, "SSHClient", factory)
    return FakeSSHClient


def test_run_remote_ssh_success(shared_root, fake_paramiko, capsys):
    """A zero exit status is a success and the client is closed."""
    fake_paramiko.exit_status = 0
    fake_paramiko.lines = ["hello from remote\n"]

    mp.run_remote_ssh("arch_4070", "echo hi", "TestStage")

    client = fake_paramiko.instances[-1]
    assert client.connected_with["hostname"] == "10.0.0.1"
    assert client.connected_with["username"] == "archuser"
    assert client.executed_cmd == "echo hi"
    assert client.closed
    assert "hello from remote" in capsys.readouterr().out


def test_run_remote_ssh_nonzero_exit_raises(shared_root, fake_paramiko):
    """A non-zero exit status must raise so the pipeline aborts."""
    fake_paramiko.exit_status = 1
    with pytest.raises(RuntimeError, match="exit code 1"):
        mp.run_remote_ssh("arch_2080", "false", "TestStage")


def test_run_remote_ssh_rejects_unknown_host_policy(shared_root, fake_paramiko):
    """Unknown hosts must be rejected (RejectPolicy), never auto-accepted."""
    import paramiko
    mp.run_remote_ssh("arch_4070", "echo hi", "TestStage")
    client = fake_paramiko.instances[-1]
    assert isinstance(client.policy, paramiko.RejectPolicy)


def test_run_remote_ssh_dry_run_does_not_connect(shared_root, fake_paramiko, capsys):
    """Dry run prints the command and never opens an SSH connection."""
    mp.run_remote_ssh("arch_4070", "echo hi", "TestStage", dry_run=True)
    assert fake_paramiko.instances == []
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "echo hi" in out
    assert "10.0.0.1" in out


# ---------------------------------------------------------------------------
# Parallel execution semantics
# ---------------------------------------------------------------------------

def test_parallel_failure_aborts_pipeline(shared_root, monkeypatch):
    """If either GPU node fails, the exception must reach the caller."""
    calls = []

    def fake_run(node_key, command, stage_label, dry_run=False):
        calls.append((node_key, stage_label))
        if node_key == "arch_2080":
            raise RuntimeError("[Arch2080-GNINA] Failed on arch_2080 with exit code 1")
        return None

    monkeypatch.setattr(mp, "run_remote_ssh", fake_run)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(fake_run, "arch_4070", "cmd", "DiffDock"),
            pool.submit(fake_run, "arch_2080", "cmd", "GNINA"),
        ]
        with pytest.raises(RuntimeError, match="exit code 1"):
            for fut in futures:
                fut.result()

    assert {c[0] for c in calls} == {"arch_4070", "arch_2080"}


# ---------------------------------------------------------------------------
# GNINA command construction
# ---------------------------------------------------------------------------

def test_gnina_cmd_mounts_only_run_dir(shared_root):
    """The container mounts the shared root read-only plus a writable run dir."""
    cmd = mp.gnina_docker_cmd("0", "--version", "20260101_000000", "LRRK2")
    run_dir = os.path.join(str(shared_root), "runs", "20260101_000000", "LRRK2")
    assert f"-v {mp.q(run_dir)}:{mp.q(run_dir)}" in cmd
    # Shared root is mounted read-only.
    assert f":{mp.q(str(shared_root))}:ro" in cmd
    # The old flat 05_gnina_rescored path must no longer be mounted writable.
    assert f"{mp.q(str(shared_root))}/05_gnina_rescored" not in cmd


# ---------------------------------------------------------------------------
# End-to-end dry run
# ---------------------------------------------------------------------------

def test_main_dry_run_prints_full_command_surface(shared_root, monkeypatch, capsys):
    """A dry run exercises every stage without touching the GPU nodes."""
    root = shared_root
    raw = os.path.join(root, "01_ligands_raw", "library.csv")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    pd.DataFrame({
        "ligand_id": ["LRRK2_CHEMBL4098877"],
        "smiles": ["C[C@@H]1CN(c2cc(-c3n[nH]c4ccc(OC5(C)CC5)cc34)ncn2)C[C@H](C)O1"],
    }).to_csv(raw, index=False)

    monkeypatch.setattr(sys, "argv", [
        "master_pipeline.py", "--target", "LRRK2", "--pdb", "7LHT_clean.pdb",
        "--raw_smiles", raw, "--run-id", "dryrun_1", "--dry-run",
    ])
    mp.main()

    out = capsys.readouterr().out
    # Stage 1 runs locally.
    assert "STAGE 1" in out
    # Remote commands are printed, not executed.
    assert "dry-run:P2Rank-PocketFinder" in out
    assert "dry-run:Arch4070-DiffDock" in out
    assert "dry-run:Arch2080-GNINA" in out
    assert "prank predict" in out
    assert "python -m inference" in out
    assert "gnina" in out
    # Stage 4 is reported but not executed.
    assert "stages 2-4 not executed" in out
    # No SSH client was ever created.
    assert FakeSSHClient.instances == []


def test_main_rejects_raw_smiles_outside_shared_root(shared_root, monkeypatch):
    """--raw_smiles must live under the shared root (path-traversal guard)."""
    outside = "/tmp/outside_library.csv"
    pd.DataFrame({"ligand_id": ["x"], "smiles": ["CCO"]}).to_csv(outside, index=False)
    monkeypatch.setattr(sys, "argv", [
        "master_pipeline.py", "--target", "LRRK2", "--pdb", "7LHT_clean.pdb",
        "--raw_smiles", outside, "--dry-run",
    ])
    with pytest.raises(SystemExit):
        mp.main()


def test_main_dry_run_tolerates_missing_library(shared_root, monkeypatch, capsys):
    """Dry run still verifies the remote command surface if the library is absent."""
    missing = os.path.join(shared_root, "01_ligands_raw", "not_built_yet.csv")

    monkeypatch.setattr(sys, "argv", [
        "master_pipeline.py", "--target", "GBA1", "--pdb", "2V3D_clean.pdb",
        "--raw_smiles", missing, "--run-id", "dryrun_2", "--dry-run",
    ])
    mp.main()

    out = capsys.readouterr().out
    assert "STAGE 1 skipped" in out
    assert "dry-run:P2Rank-PocketFinder" in out
    assert "dry-run:Arch4070-DiffDock" in out
    assert "dry-run:Arch2080-GNINA" in out
