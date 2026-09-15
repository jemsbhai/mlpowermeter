"""EXP-002 measurement pipeline against the fake device with sleep-based shapes."""

from __future__ import annotations

import gzip
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

from tomlml.experiments.exp_002_lowrank_crossover import build_plan, config_key
from tomlml.workloads.lowrank import rank_for

REPO_ROOT = Path(__file__).resolve().parents[1]


class SleepShape:
    """Stand-in for LowRankShape: each call sleeps a time that depends on the
    cell, so calibration and window call counts differ across cells."""

    def __init__(self, d, fracs, batches, n_sets, exact_fail=None):
        self.d = int(d)
        self.rank_fractions = [float(f) for f in fracs]
        self.batches = [int(b) for b in batches]
        self.n_sets = int(n_sets)
        self.ranks = {f: rank_for(self.d, f) for f in self.rank_fractions}
        self.exact_fail = exact_fail or set()
        self.setup_calls = 0
        self.torn_down = False

    def setup(self):
        self.setup_calls += 1

    def teardown(self):
        self.torn_down = True

    def runner(self, f, B, realization):
        base = 0.0005 if realization == "dense" else 0.0008

        def run_once():
            time.sleep(base)
        return run_once

    def exactness(self, f, B):
        bad = (f, B) in self.exact_fail
        return {"d": self.d, "f": f, "r": self.ranks[f], "B": B, "sets_checked": 2,
                "rel_frobenius": [1e-2 if bad else 1e-6] * 2, "max_rel_frobenius": 1e-2 if bad else 1e-6}

    def describe(self):
        return {"d": self.d, "ranks": {f"{f:g}": r for f, r in self.ranks.items()}, "kind": "sleep"}


def fake_census(run_once):
    return {"method": "fake", "commands_per_call": 2.0, "kernels_per_call": 2.0,
            "memcpy_per_call": 0.0, "memset_per_call": 0.0}


CONFIG: Dict[str, Any] = {
    "experiment_id": "EXP-002", "name": "lowrank-crossover", "experiment": "exp_002_lowrank_crossover",
    "phase": "pilot", "seed": 42, "deterministic": False, "torch_device": 0, "sample_interval_s": 0.02,
    "allow_dirty": True,
    "grid": {"shapes": [512, 1024], "rank_fractions": [0.0625, 0.5], "batches": [1, 64],
             "realizations": ["dense", "factorized"], "reps": 2,
             "calibration_shapes": [512], "heldout_shapes": [1024]},
    "quick_grid": {"shapes": [512], "rank_fractions": [0.5], "batches": [1],
                   "realizations": ["dense", "factorized"], "reps": 1,
                   "calibration_shapes": [512], "heldout_shapes": []},
    "workload": {"dtype": "float32", "allow_tf32": False, "data_seed": 1234,
                 "weight_cycling_l2_multiple": 2, "exactness_tolerance_rel_frobenius": 1.0e-4},
    "measurement": {"window_s": 0.15, "warmup_s": 0.02,
                    "settle_load": {"rank_fraction": 0.5, "batch": 64, "realization": "dense"},
                    "thermal_settle_threshold_c": 1.0, "thermal_settle_window_s": 0.12,
                    "thermal_settle_poll_s": 0.04, "thermal_settle_timeout_s": 2.0,
                    "census_warmup_calls": 1, "census_calls": 2},
    "analysis": {"models": ["P0", "M1", "M2"], "selection": "leave_one_cell_out", "bootstrap_resamples": 10},
    "criteria": {"heldout_median_ape_max": 0.2, "crossover_coverage_min_cells": 1,
                 "b1_rank_fraction_max": 0.25, "monotonicity_spearman_min": 0.8,
                 "exactness_max_excluded_fraction": 0.05, "regret_max": 0.05,
                 "contaminated_windows_max": 0, "implausible_power_fraction_max": 0.01},
}


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_experiment", REPO_ROOT / "scripts" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_plan_covers_every_cell_once_and_is_seeded():
    grid = CONFIG["grid"]
    plan_a = build_plan(grid, np.random.default_rng(7))
    plan_b = build_plan(grid, np.random.default_rng(7))
    plan_c = build_plan(grid, np.random.default_rng(8))
    assert set(plan_a) == {512, 1024}
    for d, cells in plan_a.items():
        assert len(cells) == 2 * 2 * 2 * 2
        assert len({(c["f"], c["B"], c["realization"], c["rep"]) for c in cells}) == len(cells)
    assert plan_a == plan_b and plan_a != plan_c
    assert config_key(512, 0.0625, 1, "dense") == "d512_f0.0625_B1_dense"


def test_exp002_pipeline_end_to_end(fake_nvml, tmp_path, monkeypatch):
    import tomlml.experiments.exp_002_lowrank_crossover as exp

    shapes: List[SleepShape] = []

    def factory(d, fracs, batches, K):
        s = SleepShape(d, fracs, batches, K, exact_fail={(0.0625, 64)} if d == 1024 else None)
        shapes.append(s)
        return s

    original_run = exp.run
    monkeypatch.setattr(exp, "run", lambda ctx: original_run(ctx, shape_factory=factory, census_fn=fake_census))
    monkeypatch.setattr(exp, "l2_cache_bytes",
                        lambda device_index=0, fallback_bytes=0: {"bytes": 64 * 2**20, "source": "fallback (test)"})
    cfg_path = tmp_path / "exp002.yaml"
    cfg_path.write_text(yaml.safe_dump(CONFIG))
    runner = _load_runner()
    rc = runner.main(["--config", str(cfg_path), "--out-root", str(tmp_path), "--run-id", "e2"])
    assert rc == 0

    out = tmp_path / "experiments" / "exp_002_lowrank-crossover" / "a100-sxm4-40gb" / "e2"
    for f in ("results/windows.json", "results/census.json", "results/exactness.json",
              "results/descriptors.json", "results/calibration.json", "results/summary.json",
              "results/capabilities.json", "samples/d512.csv.gz", "samples/d1024.csv.gz"):
        assert (out / f).exists(), f

    summary = json.loads((out / "results" / "summary.json").read_text())
    assert summary["analysis_pending"] is True and summary["gate_valid"] is True
    assert set(summary["shapes"]) == {"512", "1024"}
    assert summary["shapes"]["512"]["n_sets"] == 128          # 64 MiB fallback L2, 1 MiB per set
    assert summary["l2_cache"]["source"].startswith("fallback")
    assert set(summary["settles"]) == {"512", "1024"}
    assert summary["excluded"] == [{"key": "d1024_f0.0625_B64", "reason": "exactness",
                                    "max_rel_frobenius": 1e-2, "tolerance": 1e-4}]
    # 2 shapes x 2 fractions x 2 batches x 2 realizations x 2 reps = 32 cells, minus one excluded config (4 windows)
    assert summary["n_windows"] == 28
    assert summary["n_configurations"] == 14
    assert summary["contaminated_windows"] == 0
    assert summary["tf32"]["torch"] in (True, False)
    if summary["tf32"]["torch"]:
        assert summary["tf32"]["matmul_allow_tf32"] is False

    windows = json.loads((out / "results" / "windows.json").read_text())
    assert len(windows) == 28
    keys = {w["key"] for w in windows}
    assert "d1024_f0.0625_B64_dense" not in keys and "d512_f0.0625_B64_dense" in keys
    w0 = windows[0]
    for k in ("d", "f", "r", "B", "realization", "rep", "per_call_s", "energy_per_call_j",
              "energy_per_call_integral_j", "regime", "n_calls", "duration_s"):
        assert k in w0, k
    assert w0["regime"] == "uncapped"
    assert all(w["energy_per_call_j"] > 0 for w in windows)
    dense = [w for w in windows if w["realization"] == "dense"]
    fact = [w for w in windows if w["realization"] == "factorized"]
    assert np.mean([w["per_call_s"] for w in fact]) > np.mean([w["per_call_s"] for w in dense])

    desc = json.loads((out / "results" / "descriptors.json").read_text())
    d = desc["d512_f0.5_B64_factorized"]
    assert d["m1"]["n_mac"] == 64 * 256 * 1024 and d["commands_per_call"] == 2.0
    assert d["factorized_weights_fit_l2"] is False               # 128 sets x 1 MB of factorized weights > 64 MiB
    assert desc["d512_f0.0625_B1_factorized"]["factorized_weights_fit_l2"] is True
    assert desc["d512_f0.0625_B1_factorized"]["m2"]["weights_in_l2"] is True
    assert desc["d512_f0.0625_B1_factorized"]["m1"]["weights_in_l2"] is False

    with gzip.open(out / "samples" / "d512.csv.gz", "rt") as f:
        header = f.readline().strip().split(",")
    assert header[0] == "window" and "power_w" in header
    assert all(s.torn_down and s.setup_calls == 1 for s in shapes)
    order = summary["plan_order"]["512"]
    assert len(order) == 16 and len({tuple(o) for o in order}) == 16


def test_exp002_quick_grid_is_used_under_quick(fake_nvml, tmp_path, monkeypatch):
    import tomlml.experiments.exp_002_lowrank_crossover as exp

    def factory(d, fracs, batches, K):
        return SleepShape(d, fracs, batches, K)

    original_run = exp.run
    monkeypatch.setattr(exp, "run", lambda ctx: original_run(ctx, shape_factory=factory, census_fn=fake_census))
    cfg_path = tmp_path / "exp002q.yaml"
    cfg_path.write_text(yaml.safe_dump(CONFIG))
    runner = _load_runner()
    assert runner.main(["--config", str(cfg_path), "--out-root", str(tmp_path), "--run-id", "q1",
                        "--quick", "--quick-factor", "0.5"]) == 0
    out = tmp_path / "experiments" / "exp_002_lowrank-crossover" / "a100-sxm4-40gb" / "q1"
    summary = json.loads((out / "results" / "summary.json").read_text())
    assert summary["quick"] is True and summary["gate_valid"] is False
    assert summary["grid"]["shapes"] == [512] and summary["n_windows"] == 2
    assert summary["settings"]["window_s"] == pytest.approx(0.5)   # floor of 0.5 s after scaling


def test_exp002_crash_and_resume(fake_nvml, tmp_path, monkeypatch):
    """Crash after five measured windows, resume in the same directory, finish
    with every window measured exactly once and the same seeded order."""
    import tomlml.experiments.exp_002_lowrank_crossover as exp

    def factory(d, fracs, batches, K):
        return SleepShape(d, fracs, batches, K, exact_fail={(0.0625, 64)} if d == 1024 else None)

    original_run = exp.run
    monkeypatch.setattr(exp, "run", lambda ctx: original_run(ctx, shape_factory=factory, census_fn=fake_census))
    monkeypatch.setattr(exp, "l2_cache_bytes",
                        lambda device_index=0, fallback_bytes=0: {"bytes": 64 * 2**20, "source": "fallback (test)"})
    real_measure = exp.measure_window
    state = {"n": 0, "crash_at": 6}

    def crashing_measure(*a, **kw):
        state["n"] += 1
        if state["n"] == state["crash_at"]:
            raise RuntimeError("simulated power loss")
        return real_measure(*a, **kw)

    monkeypatch.setattr(exp, "measure_window", crashing_measure)
    cfg_path = tmp_path / "exp002.yaml"
    cfg_path.write_text(yaml.safe_dump(CONFIG))
    runner = _load_runner()
    with pytest.raises(RuntimeError, match="simulated power loss"):
        runner.main(["--config", str(cfg_path), "--out-root", str(tmp_path), "--run-id", "cr"])

    out = tmp_path / "experiments" / "exp_002_lowrank-crossover" / "a100-sxm4-40gb" / "cr"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "simulated power loss" in manifest["error"]
    ledger = (out / "results" / "windows.jsonl").read_text().splitlines()
    assert len(ledger) == 5
    first_labels = [json.loads(line)["label"] for line in ledger]
    assert (out / "results" / "shape_d512.json").exists()
    assert not (out / "results" / "shape_d1024.json").exists()
    assert not (out / "results" / "windows.json").exists()
    with gzip.open(out / "samples" / "d512.csv.gz", "rt") as f:
        lines = f.read().splitlines()
    assert lines[0].startswith("window,") and len({ln.split(",")[0] for ln in lines[1:]}) == 5

    # a fresh (non-resume) run must never reuse the directory
    with pytest.raises(FileExistsError):
        runner.main(["--config", str(cfg_path), "--out-root", str(tmp_path), "--run-id", "cr"])

    # resume refuses a code change unless overridden
    real_git = runner.git_info
    monkeypatch.setattr(runner, "git_info", lambda root=None: dict(real_git(root), commit="0" * 40))
    assert runner.main(["--config", str(cfg_path), "--out-root", str(tmp_path), "--resume", "cr"]) == 3
    monkeypatch.setattr(runner, "git_info", real_git)

    # resume proper: no more crashes, nothing re-measured
    state["crash_at"] = -1
    measured_before = state["n"]
    rc = runner.main(["--config", str(cfg_path), "--out-root", str(tmp_path), "--resume", "cr"])
    assert rc == 0
    summary = json.loads((out / "results" / "summary.json").read_text())
    assert summary["resumed"] is True
    assert summary["n_windows"] == 28
    assert summary["n_windows_skipped_on_resume"] == 5
    assert summary["n_windows_measured_this_session"] == 23
    assert state["n"] - measured_before == 23
    windows = json.loads((out / "results" / "windows.json").read_text())
    labels = [w["label"] for w in windows]
    assert len(labels) == len(set(labels)) == 28
    assert labels[:5] == first_labels                       # ledger order preserved
    assert sum(1 for w in windows if w["measured_in_resume"]) == 23
    assert set(summary["shapes"]) == {"512", "1024"} and set(summary["settles"]) == {"512", "1024"}
    assert summary["n_configurations"] == 14 and len(summary["excluded"]) == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed" and len(manifest["resumes"]) == 1
    assert manifest["resumes"][0]["previous_status"] == "failed"
    assert manifest["resumes"][0]["commit_differs"] is False
    assert (out / "environment.resume1.json").exists()
    ledger = (out / "results" / "windows.jsonl").read_text().splitlines()
    assert len(ledger) == 28
    with gzip.open(out / "samples" / "d512.csv.gz", "rt") as f:
        lines = f.read().splitlines()
    assert lines.count(lines[0]) == 1                         # one header even across gzip members
    assert len({ln.split(",")[0] for ln in lines[1:]}) == 16

    # a completed run cannot be resumed
    assert runner.main(["--config", str(cfg_path), "--out-root", str(tmp_path), "--resume", "cr"]) == 3


def test_ledger_drops_partial_last_line(tmp_path):
    from tomlml.experiments.exp_002_lowrank_crossover import append_ledger, load_ledger

    path = tmp_path / "windows.jsonl"
    append_ledger(path, {"label": "a", "x": 1})
    append_ledger(path, {"label": "b", "x": 2})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"label": "c", "x": ')          # crash mid-write
    recs = load_ledger(path)
    assert [r["label"] for r in recs] == ["a", "b"]
    assert load_ledger(tmp_path / "missing.jsonl") == []
