"""Tests for EXP-001: pure analysis functions and an end-to-end pipeline run
against the fake NVML binding with host-side sleep workloads."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from tomlml.experiments.exp_001_instrumentation import (
    cv, evaluate_criteria, gate_passes, linear_drift, min_sufficient_window, select_energy_source,
    update_period_stats, window_stats,
)
from tomlml.workloads import SleepWorkload

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- pure functions


def test_update_period_stats_detects_change_intervals():
    times = [i * 0.001 for i in range(1000)]           # 1 kHz polling for 1 s
    values = [i // 100 for i in range(1000)]            # value changes every 100 ms
    st = update_period_stats(times, values)
    assert st["n_polls"] == 1000
    assert st["n_changes"] == 9
    assert st["distinct_values"] == 10
    assert st["period_ms_median"] == pytest.approx(100.0)
    assert st["period_ms_min"] == pytest.approx(100.0)
    assert st["poll_rate_hz"] == pytest.approx(1000.0, rel=0.01)
    empty = update_period_stats([0.0], [1])
    assert empty["period_ms_median"] is None and empty["n_changes"] == 0
    none_vals = update_period_stats([0.0, 0.1, 0.2], [None, None, None])
    assert none_vals["n_changes"] == 0 and none_vals["distinct_values"] == 0


def test_linear_drift_recovers_slope():
    times = [float(t) for t in range(60)]
    values = [20.0 + 0.01 * t for t in times]           # 0.01 W/s = 0.6 W/min
    d = linear_drift(times, values)
    assert d["slope_per_s"] == pytest.approx(0.01)
    assert d["slope_per_min"] == pytest.approx(0.6)
    assert d["r2"] == pytest.approx(1.0)
    assert d["half_diff"] > 0
    assert linear_drift([0.0, 1.0], [1.0, 2.0])["slope_per_s"] is None
    assert linear_drift([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])["slope_per_s"] is None


def test_cv_and_window_stats():
    assert cv([10.0, 10.0, 10.0]) == 0.0
    assert cv([1.0]) is None
    assert cv([0.0, 0.0]) is None
    st = window_stats([1.0, 2.0, 3.0, None])
    assert st["n"] == 3 and st["mean"] == 2.0 and st["min"] == 1.0 and st["max"] == 3.0
    assert st["cv"] == pytest.approx(0.5)
    assert window_stats([])["mean"] is None


def test_min_sufficient_window_requires_all_longer_lengths():
    per = {"0.5": {"cv": 0.05}, "1": {"cv": 0.03}, "2": {"cv": 0.015},
           "5": {"cv": 0.01}, "10": {"cv": 0.008}, "20": {"cv": 0.005}}
    assert min_sufficient_window(per, 0.02) == 2.0
    per["10"]["cv"] = 0.03                               # a longer length fails: not sufficient below it
    assert min_sufficient_window(per, 0.02) == 20.0
    per["20"]["cv"] = None
    assert min_sufficient_window(per, 0.02) is None


def _passing_summary():
    return {
        "device": {"uuid_verified": True},
        "counter": {"supported": True, "monotonic_all_windows": True},
        "resolution": {"load": {"energy": {"period_ms_median": 100.0}}},
        "repeatability": {
            "matmul_large": {"stats_counter": {"cv": 0.01}, "stats_integral": {"cv": 0.012},
                             "counter_vs_integral_max_abs_rel": 0.02,
                             "throttle_reasons_union": ["SwPowerCap"]},
            "matmul_tiny_loop": {"stats_counter": {"cv": 0.03}, "stats_integral": {"cv": 0.035},
                                 "counter_vs_integral_max_abs_rel": 0.04,
                                 "throttle_reasons_union": []},
        },
        "thermal": {"settle_post_heat": {"timed_out": False, "wait_time_s": 40.0},
                    "settle_timeout_s": 300, "heat": {"throttle_reasons": ["SwPowerCap"]}},
        "isolation": {"contaminated_windows": 0, "samples_total": 20000,
                      "power_implausible_samples_total": 1},
        "window_sufficiency": {"matmul_large": {"min_sufficient_window_counter_s": 5.0,
                                                "min_sufficient_window_integral_s": 10.0}},
    }


CRITERIA = {
    "counter_update_period_ms_max": 200, "counter_vs_integral_rel_max": 0.05,
    "repeatability_cv_max": {"matmul_large": 0.03, "matmul_tiny_loop": 0.05},
    "min_window_cv_target": 0.02, "min_window_s_max": 10, "min_window_workloads": ["matmul_large"],
    "forbidden_throttle_reasons": ["HwSlowdown", "HwThermalSlowdown", "SwThermalSlowdown",
                                   "HwPowerBrakeSlowdown"],
    "implausible_power_fraction_max": 0.01,
}

ALL_KEYS = {
    "C1_counter_supported_and_monotonic", "C2_counter_update_period_ms",
    "C3_counter_vs_power_integral_rel", "C4_repeatability_cv_matmul_large",
    "C4_repeatability_cv_matmul_tiny_loop", "C5_thermal_settle_before_timeout",
    "C6_no_foreign_compute_processes", "C7_min_sufficient_window_s_matmul_large",
    "C8_no_forbidden_throttle_reasons", "C9_device_uuid_verified",
    "C10_implausible_power_sample_fraction",
}


def test_evaluate_criteria_all_pass_with_counter_selected():
    s = _passing_summary()
    sel = select_energy_source(s, CRITERIA)
    assert sel["selected"] == "counter" and sel["counter_rejected_reasons"] == []
    res = evaluate_criteria(s, CRITERIA)
    assert set(res) == ALL_KEYS
    assert all(c["pass"] for c in res.values()), {k: c for k, c in res.items() if not c["pass"]}
    assert {k for k, c in res.items() if not c["gating"]} == {
        "C1_counter_supported_and_monotonic", "C2_counter_update_period_ms",
        "C3_counter_vs_power_integral_rel"}
    assert res["C4_repeatability_cv_matmul_large"]["value"] == 0.01      # counter stats
    assert res["C7_min_sufficient_window_s_matmul_large"]["value"] == 5.0
    assert gate_passes(res)


def test_counter_rejected_falls_back_to_power_integral_and_gate_can_still_pass():
    """The rtx4090-laptop situation from the 2026-09-15 quick run: counter
    disagrees with the integral by far more than 5 percent."""
    s = _passing_summary()
    s["repeatability"]["matmul_large"]["counter_vs_integral_max_abs_rel"] = 0.57
    s["repeatability"]["matmul_tiny_loop"]["counter_vs_integral_max_abs_rel"] = 2.25
    sel = select_energy_source(s, CRITERIA)
    assert sel["selected"] == "power_integral" and sel["counter_rejected_reasons"] == ["C3"]
    res = evaluate_criteria(s, CRITERIA)
    assert res["C3_counter_vs_power_integral_rel"]["pass"] is False
    assert res["C3_counter_vs_power_integral_rel"]["gating"] is False
    assert res["C4_repeatability_cv_matmul_large"]["value"] == 0.012     # integral stats
    assert res["C7_min_sufficient_window_s_matmul_large"]["value"] == 10.0
    assert gate_passes(res)
    # but the integral must itself be fit for purpose
    s["window_sufficiency"]["matmul_large"]["min_sufficient_window_integral_s"] = 20.0
    assert not gate_passes(evaluate_criteria(s, CRITERIA))


def test_counter_rejected_for_non_monotonic_or_slow_updates():
    s = _passing_summary()
    s["counter"]["monotonic_all_windows"] = False
    s["resolution"]["load"]["energy"]["period_ms_median"] = 250.0
    sel = select_energy_source(s, CRITERIA)
    assert sel["selected"] == "power_integral"
    assert sel["counter_rejected_reasons"] == ["C1", "C2"]


@pytest.mark.parametrize("mutate,failing", [
    (lambda s: s["repeatability"]["matmul_large"]["stats_counter"].update(cv=0.04), "C4_repeatability_cv_matmul_large"),
    (lambda s: s["thermal"]["settle_post_heat"].update(timed_out=True), "C5_thermal_settle_before_timeout"),
    (lambda s: s["isolation"].update(contaminated_windows=2), "C6_no_foreign_compute_processes"),
    (lambda s: s["window_sufficiency"]["matmul_large"].update(min_sufficient_window_counter_s=20.0),
     "C7_min_sufficient_window_s_matmul_large"),
    (lambda s: s["thermal"]["heat"].update(throttle_reasons=["HwThermalSlowdown"]),
     "C8_no_forbidden_throttle_reasons"),
    (lambda s: s["device"].update(uuid_verified=None), "C9_device_uuid_verified"),
    (lambda s: s["isolation"].update(power_implausible_samples_total=500), "C10_implausible_power_sample_fraction"),
])
def test_evaluate_criteria_single_gating_failures(mutate, failing):
    s = _passing_summary()
    mutate(s)
    res = evaluate_criteria(s, CRITERIA)
    failed = sorted(k for k, c in res.items() if not c["pass"])
    assert failed == [failing]
    assert not gate_passes(res)


# ----------------------------------------------------------------------------- end to end


TEST_CONFIG = {
    "experiment_id": "EXP-001", "name": "instrumentation-gate",
    "experiment": "exp_001_instrumentation", "phase": "pilot",
    "seed": 42, "deterministic": False, "torch_device": 0, "sample_interval_s": 0.02,
    "probe_control_permissions": True, "allow_dirty": True,
    "workloads": {"matmul_large": {}, "matmul_tiny_loop": {}},
    "resolution": {"duration_s": 0.2},
    "idle": {"duration_s": 0.3, "discard_s": 0.05},
    "thermal": {"heat_duration_s": 0.2, "threshold_c": 1.0, "window_s": 0.15, "poll_s": 0.05,
                "timeout_s": 3.0},
    "repeatability": {"warmup_s": 0.05, "block_duration_s": 0.1, "n_blocks": 3},
    "window_sufficiency": {"lengths_s": [0.05, 0.1], "reps": 2, "warmup_s": 0.02},
    "criteria": dict(CRITERIA, repeatability_cv_max={"matmul_large": 0.5, "matmul_tiny_loop": 0.5},
                     min_window_cv_target=0.5, min_window_s_max=1.0,
                     counter_update_period_ms_max=1000),
}


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_experiment", REPO_ROOT / "scripts" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exp001_end_to_end_with_fake_device(fake_nvml, tmp_path, monkeypatch):
    import tomlml.experiments.exp_001_instrumentation as exp

    monkeypatch.setattr(exp, "make_workload",
                        lambda name, params, device_index=0: SleepWorkload(name, per_call_s=0.002))
    cfg_path = tmp_path / "exp_test.yaml"
    cfg_path.write_text(yaml.safe_dump(TEST_CONFIG))
    runner = _load_runner()

    rc = runner.main(["--config", str(cfg_path), "--out-root", str(tmp_path), "--run-id", "r1"])
    assert rc in (0, 1)  # 1 only if a physics criterion failed on the fake; structure is what we check

    out = tmp_path / "experiments" / "exp_001_instrumentation-gate" / "a100-sxm4-40gb" / "r1"
    for f in ("config.yaml", "seed.json", "environment.json", "manifest.json",
              "results/capabilities.json", "results/windows.json", "results/summary.json",
              "logs/run.log", "samples/resolution_idle.csv", "samples/resolution_load.csv",
              "samples/idle_pre.csv", "samples/heat.csv", "samples/idle_post.csv",
              "samples/repeatability_matmul_large.csv", "samples/window_sufficiency_matmul_tiny_loop.csv"):
        assert (out / f).exists(), f

    import json
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["device"]["nvml_index"] == 1 and manifest["device"]["uuid_verified"] is True
    assert manifest["summary"]["gate_valid"] is True

    summary = json.loads((out / "results" / "summary.json").read_text())
    assert summary["device"]["uuid_verified"] is True
    assert summary["capabilities_brief"]["energy_counter_supported"] is True
    assert summary["capabilities_brief"]["power_max_plausible_w"] == pytest.approx(350.0)
    assert summary["counter"]["supported"] is True and summary["counter"]["monotonic_all_windows"] is True
    assert summary["resolution"]["load"]["energy"]["n_polls"] > 10
    assert summary["idle"]["pre_heat"]["mean_power_w"] == pytest.approx(60.0)
    assert summary["thermal"]["settle_post_heat"]["timed_out"] is False
    assert summary["thermal"]["hot_idle_bias_w"] == pytest.approx(0.0, abs=1e-6)
    rep = summary["repeatability"]["matmul_large"]
    assert rep["n_blocks"] == 3 and len(rep["energy_per_call_counter_j"]) == 3
    assert rep["stats_counter"]["mean"] == pytest.approx(60.0 * 0.002, rel=0.5)
    assert rep["stats_integral"]["mean"] == pytest.approx(60.0 * 0.002, rel=0.5)
    assert rep["counter_vs_integral_max_abs_rel"] is not None and rep["counter_vs_integral_max_abs_rel"] < 0.2
    ws = summary["window_sufficiency"]["matmul_tiny_loop"]
    assert set(ws["per_length_counter"]) == {"0.05", "0.1"} and len(ws["order"]) == 4
    assert ws["per_length_integral"]["0.1"]["target_s"] == pytest.approx(0.1)
    assert summary["isolation"]["contaminated_windows"] == 0
    assert summary["isolation"]["samples_total"] > 0
    assert summary["energy_source"]["selected"] in ("counter", "power_integral")
    assert set(summary["criteria"]) == ALL_KEYS
    assert summary["criteria"]["C9_device_uuid_verified"]["pass"] is True
    assert summary["criteria"]["C6_no_foreign_compute_processes"]["pass"] is True
    assert summary["criteria"]["C10_implausible_power_sample_fraction"]["pass"] is True
    cap = json.loads((out / "results" / "capabilities.json").read_text())
    assert cap["power_limit_write_probe"]["probed"] is False  # NotSupported on the fake

    windows = json.loads((out / "results" / "windows.json").read_text())
    sections = {w["section"] for w in windows}
    assert sections == {"idle_pre", "heat", "idle_post", "repeatability", "window_sufficiency"}
    assert all("samples" not in w for w in windows)

    # a second run with the same id must be refused (never overwrite)
    with pytest.raises(FileExistsError):
        runner.main(["--config", str(cfg_path), "--out-root", str(tmp_path), "--run-id", "r1"])


def test_runner_refuses_dirty_tree_for_full_runs(fake_nvml, tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner, "git_info",
                        lambda root=None: {"commit": "abc", "branch": "main", "dirty": True, "error": None})
    cfg = dict(TEST_CONFIG, allow_dirty=False)
    cfg_path = tmp_path / "exp_dirty.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert runner.main(["--config", str(cfg_path), "--out-root", str(tmp_path)]) == 2
    assert not (tmp_path / "experiments").exists()


def test_runner_deep_merge_and_config_sources(tmp_path):
    runner = _load_runner()
    merged = runner.deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}, "c": 4})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}
    base = tmp_path / "base.yaml"
    base.write_text("seed: 1\nsample_interval_s: 0.05\n")
    exp = tmp_path / "exp.yaml"
    exp.write_text("seed: 7\nname: x\n")
    cfg = runner.load_config(exp, base)
    assert cfg["seed"] == 7 and cfg["sample_interval_s"] == 0.05 and cfg["name"] == "x"
    assert [s["path"].endswith(n) for s, n in zip(cfg["_sources"], ("base.yaml", "exp.yaml"))] == [True, True]


def test_quick_mode_scaling():
    from tomlml.experiments import RunContext

    ctx = RunContext(exp_id="EXP-000", name="x", platform_tag="t", run_id="r", out_dir=Path("."),
                     config={}, dev=None, sync_fn=lambda: None, logger=logging.getLogger("t"),
                     quick=True, quick_factor=0.05)
    assert ctx.scaled(60.0, 1.0) == pytest.approx(3.0)
    assert ctx.scaled(10.0, 1.0) == 1.0          # floor applies
    assert ctx.scaled_int(10, 3) == 3
    assert ctx.scaled_int(100, 3) == 5
    full = RunContext(exp_id="EXP-000", name="x", platform_tag="t", run_id="r", out_dir=Path("."),
                      config={}, dev=None, sync_fn=lambda: None, logger=logging.getLogger("t"))
    assert full.scaled(60.0) == 60.0 and full.scaled_int(10) == 10
    assert isinstance(np.random.default_rng(1).permutation(3).tolist(), list)
