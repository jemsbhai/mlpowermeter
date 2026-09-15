"""Tests for seeds, manifests, and environment snapshots."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tomlml.utils import (
    build_manifest, derive_seed, finalize_manifest, freeze_config, git_info,
    make_experiment_dir, platform_tag_from_gpu_name, read_json, seed_record_for_config,
    set_all_seeds, sha256_file, sha256_text, slugify, snapshot_environment, write_json,
)

# ----------------------------------------------------------------------------- seeds


def test_derive_seed_is_stable_and_in_range():
    a = derive_seed(42, "data_split")
    assert a == derive_seed(42, "data_split")
    assert a != derive_seed(42, "init")
    assert a != derive_seed(43, "data_split")
    assert 0 <= a < 2**31 - 1


def test_set_all_seeds_reproduces_numpy_and_random():
    import random
    np = pytest.importorskip("numpy")

    rec = set_all_seeds(123)
    x1 = np.random.rand(3)
    r1 = random.random()
    set_all_seeds(123)
    assert np.allclose(np.random.rand(3), x1)
    assert random.random() == r1
    assert rec["master_seed"] == 123 and rec["numpy"] == 123
    assert rec["deterministic_algorithms"] is False
    assert isinstance(rec["notes"], list)


def test_seed_record_for_config():
    assert seed_record_for_config({"seed": 7}) == 7
    assert seed_record_for_config({}) == 42
    assert seed_record_for_config({"seed": None}, default_seed=5) == 5


# ----------------------------------------------------------------------------- manifest


@pytest.mark.parametrize("name,tag", [
    ("NVIDIA GeForce RTX 4090 Laptop GPU", "rtx4090-laptop"),
    ("NVIDIA A100-SXM4-40GB", "a100-sxm4-40gb"),
    ("NVIDIA A100 80GB PCIe", "a100-pcie-80gb"),
    ("NVIDIA H100 80GB HBM3", "h100-80gb-hbm3"),
    ("Tesla V100-SXM2-16GB", "v100-sxm2-16gb"),
    (None, "unknown-gpu"),
])
def test_platform_tag(name, tag):
    assert platform_tag_from_gpu_name(name) == tag


def test_slugify():
    assert slugify("Instrumentation Gate (G0)") == "instrumentation-gate-g0"
    assert slugify("  a__b  ") == "a-b"


def test_sha256_helpers(tmp_path):
    assert sha256_text("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_write_and_read_json_roundtrip(tmp_path):
    p = tmp_path / "nested" / "x.json"
    write_json(p, {"b": Path("/tmp/z"), "a": {1, 2}, "c": [1, 2.5, None]})
    data = read_json(p)
    assert data == {"a": [1, 2], "b": "/tmp/z", "c": [1, 2.5, None]}
    assert list(json.loads(p.read_text()).keys()) == ["a", "b", "c"]  # sorted keys


def test_make_experiment_dir_never_overwrites(tmp_path):
    out = make_experiment_dir(tmp_path, "EXP-001", "Instrumentation Gate", "rtx4090-laptop",
                              run_id="20260915T120000")
    assert out == tmp_path / "experiments" / "exp_001_instrumentation-gate" / "rtx4090-laptop" / "20260915T120000"
    for sub in ("results", "samples", "logs", "figures"):
        assert (out / sub).is_dir()
    with pytest.raises(FileExistsError):
        make_experiment_dir(tmp_path, "EXP-001", "Instrumentation Gate", "rtx4090-laptop",
                            run_id="20260915T120000")
    out2 = make_experiment_dir(tmp_path, "exp-7", "x", "a100-sxm4-40gb")
    assert out2.parent.parent.name == "exp_007_x"


def test_freeze_config_and_manifest(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("experiment_id: EXP-001\nseed: 42\n")
    out = make_experiment_dir(tmp_path, "EXP-001", "x", "rtx4090-laptop", run_id="r1")
    frozen = freeze_config(cfg, out)
    assert Path(frozen["frozen"]).read_text() == cfg.read_text()
    assert frozen["sha256"] == sha256_file(cfg)

    m = build_manifest("EXP-001", "x", "rtx4090-laptop", "r1", out, {"seed": 42},
                       {"commit": "abc", "dirty": False}, device={"nvml_index": 0})
    assert m["status"] == "running" and "monotonic_start" in m
    m = finalize_manifest(m, "completed", summary={"ok": True})
    assert m["status"] == "completed"
    assert "monotonic_start" not in m and m["wall_time_s"] >= 0
    assert m["summary"] == {"ok": True}
    write_json(out / "manifest.json", m)
    assert read_json(out / "manifest.json")["experiment_id"] == "EXP-001"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_git_info_on_temporary_repo(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("1")
    git("add", "a.txt")
    git("commit", "-q", "-m", "init")
    info = git_info(tmp_path)
    assert info["error"] is None
    assert len(info["commit"]) == 40
    assert info["dirty"] is False
    (tmp_path / "a.txt").write_text("2")
    assert git_info(tmp_path)["dirty"] is True
    (tmp_path / "untracked.txt").write_text("u")  # untracked files do not count as dirty


def test_git_info_outside_repo(tmp_path):
    info = git_info(tmp_path)
    assert info["commit"] is None
    assert info["error"] is not None


# ----------------------------------------------------------------------------- envsnap


def test_snapshot_environment_has_expected_shape(fake_nvml, tmp_path):
    snap = snapshot_environment(device_state={"probe": 1}, repo_root=str(tmp_path))
    for key in ("timestamp_utc", "hostname", "os", "python", "cpu", "ram_gb",
                "torch", "nvml", "packages", "environment_variables", "git"):
        assert key in snap
    assert snap["nvml"]["available"] is True
    assert len(snap["nvml"]["devices"]) == 2
    assert snap["nvml"]["target_device_state"] == {"probe": 1}
    assert snap["nvml"]["driver_version"] == "595.79"
    assert "numpy" in snap["packages"]
    assert snap["git"]["error"] is not None  # tmp_path is not a repo
    assert fake_nvml.init_calls == fake_nvml.shutdown_calls
    json.dumps(snap)  # must be serializable as is
