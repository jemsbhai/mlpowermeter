"""Tests for the EXP-002 analysis on synthetic runs generated from a known model."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

from tomlml.analysis.exp_002 import (
    analyze, crossover_from_ratios, crossovers_from_energies, encode, evaluate_criteria,
    fit_relative_nnls, gate_passes, interval_hit, intervals_overlap, load_run, m3_energy, point_factor_hit,
    regret, spearman_monotone,
)
from tomlml.to_model.gemm import dense_descriptor, factorized_descriptor
from tomlml.workloads.lowrank import rank_for

REPO_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------- crossovers


def test_crossover_interpolates_first_upward_crossing():
    c = crossover_from_ratios([8, 16, 32], [0.5, 0.8, 1.25])
    assert c["kind"] == "numeric" and c["monotone"] is True and c["n_crossings"] == 1
    assert c["r_star"] == pytest.approx(math.sqrt(16 * 32))     # log(0.8) and log(1.25) are symmetric
    c = crossover_from_ratios([32, 8, 16], [1.25, 0.5, 1.0])       # unsorted input; ratio exactly 1 at r=16
    assert c["r_star"] == pytest.approx(16.0)
    assert crossover_from_ratios([8, 16], [1.1, 1.5])["kind"] == "none_dense"
    assert crossover_from_ratios([8, 16], [0.5, 1.0])["kind"] == "none_factorized"
    c = crossover_from_ratios([8, 16, 32, 64], [0.9, 1.1, 0.95, 1.2])   # noisy double crossing
    assert c["kind"] == "numeric" and c["n_crossings"] == 3 and c["monotone"] is False
    assert c["r_star"] < 16
    c = crossover_from_ratios([8, 16], [1.2, 0.8])                  # starts above, ends below: no rule
    assert c["kind"] == "none_dense" and c["monotone"] is False


def test_encode_and_interval_hit():
    num = {"kind": "numeric", "r_star": 100.0}
    nd = {"kind": "none_dense", "r_star": None}
    nf = {"kind": "none_factorized", "r_star": None}
    assert encode(num) == 100.0 and encode(nd) == 0.0 and math.isinf(encode(nf))
    assert interval_hit(num, 50.0, 200.0) and not interval_hit(num, 120.0, 200.0)
    assert interval_hit(nd, 0.0, 300.0) and not interval_hit(nd, 10.0, 300.0)
    assert interval_hit(nf, 10.0, math.inf) and not interval_hit(nf, 10.0, 500.0)
    assert intervals_overlap(0.0, 5.0, 5.0, 9.0) and not intervals_overlap(0.0, 4.0, 5.0, 9.0)
    assert intervals_overlap(10.0, math.inf, 500.0, math.inf) and not intervals_overlap(0.0, 0.0, 1.0, 2.0)
    assert point_factor_hit(num, {"kind": "numeric", "r_star": 140.0}, 1.5)
    assert not point_factor_hit(num, {"kind": "numeric", "r_star": 160.0}, 1.5)
    assert point_factor_hit(nd, nd, 1.5) and not point_factor_hit(nd, num, 1.5)


def test_spearman_monotone_handles_verdicts():
    m = spearman_monotone([1, 4, 16, 64], [0.0, 10.0, 40.0, math.inf])
    assert m["rho"] == pytest.approx(1.0) and m["n_decreases"] == 0 and m["constant"] is False
    m = spearman_monotone([1, 4, 16, 64], [0.0, 0.0, 0.0, 0.0])
    assert m["constant"] is True and m["rho"] == 1.0
    m = spearman_monotone([1, 4, 16, 64], [40.0, 10.0, 20.0, 5.0])
    assert m["rho"] < 0 and m["n_decreases"] == 2


# ----------------------------------------------------------------------------- fitting


def test_relative_nnls_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(1e9, 1e12, 200), rng.uniform(1e8, 1e11, 200), rng.integers(1, 5, 200)])
    true = np.array([1e-15, 5e-13, 2e-4])
    y = X @ true
    assert np.allclose(fit_relative_nnls(X, y), true, rtol=1e-6)
    y_noisy = y * np.exp(rng.normal(0, 0.01, 200))
    assert np.allclose(fit_relative_nnls(X, y_noisy), true, rtol=0.05)
    # a negative-looking contribution is clipped at zero, never negative
    y_neg = X[:, :2] @ true[:2]
    coef = fit_relative_nnls(X, y_neg)
    assert coef.min() >= 0 and coef[2] < 1e-9


# ----------------------------------------------------------------------------- synthetic run


TRUE = {"a_c": 1e-15, "a_m": 5e-13, "a_o": 2e-4}   # J per TO, J per TO, J per command
GRID = {"shapes": [256, 512, 1024], "rank_fractions": [1 / 32, 1 / 8, 1 / 4, 1 / 2, 3 / 4],
        "batches": [1, 16, 256, 4096], "realizations": ["dense", "factorized"], "reps": 3,
        "calibration_shapes": [256, 1024], "heldout_shapes": [512]}
CRITERIA = {"heldout_median_ape_max": 0.20, "crossover_coverage_min_cells": 3, "b1_rank_fraction_max": 0.25,
            "monotonicity_spearman_min": 0.8, "exactness_max_excluded_fraction": 0.05, "regret_max": 0.05,
            "contaminated_windows_max": 0, "implausible_power_fraction_max": 0.01}


def true_energy(desc: Dict[str, Any], variant: str = "m1") -> float:
    m = desc[variant]
    return TRUE["a_c"] * m["to_compute"] + TRUE["a_m"] * m["to_memory"] + TRUE["a_o"] * desc["commands_per_call"]


def make_synthetic_run(tmp_path: Path, noise: float = 0.01, generating_variant: str = "m1",
                       contaminate: int = 0, l2_bytes: int = 64 * 2**20, seed: int = 3) -> Path:
    rng = np.random.default_rng(seed)
    run_dir = tmp_path / "run"
    (run_dir / "results").mkdir(parents=True)
    descriptors: Dict[str, Any] = {}
    windows: List[Dict[str, Any]] = []
    K = 2
    for d in GRID["shapes"]:
        for f in GRID["rank_fractions"]:
            r = rank_for(d, f)
            fit_l2 = K * 2 * d * r * 4 <= l2_bytes
            for B in GRID["batches"]:
                for rz in GRID["realizations"]:
                    key = f"d{d}_f{f:g}_B{B}_{rz}"
                    if rz == "dense":
                        m1, m2, cmds = dense_descriptor(B, d, d), dense_descriptor(B, d, d), 1.0
                    else:
                        m1 = factorized_descriptor(B, d, d, r)
                        m2 = factorized_descriptor(B, d, d, r, weights_in_l2=fit_l2)
                        cmds = 2.0
                    descriptors[key] = {"m1": m1, "m2": m2, "factorized_weights_fit_l2": fit_l2,
                                        "commands_per_call": cmds, "per_call_s_calibration": 1e-4}
                    e = true_energy(descriptors[key], generating_variant)
                    for rep in range(GRID["reps"]):
                        val = e * math.exp(rng.normal(0, noise))
                        windows.append({"key": key, "label": f"{key}_rep{rep}", "d": d, "f": f, "r": r, "B": B,
                                        "realization": rz, "rep": rep, "energy_per_call_j": val,
                                        "energy_per_call_integral_j": val, "energy_per_call_counter_j": None,
                                        "regime": "uncapped", "contaminated": False, "n_samples": 400,
                                        "power_implausible_samples": 0, "counter_monotonic": None,
                                        "mean_temp_c": 60.0, "mean_sm_clock_mhz": 2000.0, "mean_power_w": 100.0,
                                        "duration_s": 20.0})
    for w in windows[:contaminate]:
        w["contaminated"] = True
    summary = {"run_id": "synthetic", "platform_tag": "fake", "quick": False, "grid": GRID,
               "settings": {"energy_source": "power_integral"}, "excluded": [],
               "contaminated_windows": contaminate, "samples_total": 400 * len(windows),
               "power_implausible_samples_total": 0, "l2_cache": {"bytes": l2_bytes}}
    (run_dir / "results" / "windows.json").write_text(json.dumps(windows))
    (run_dir / "results" / "descriptors.json").write_text(json.dumps(descriptors))
    (run_dir / "results" / "summary.json").write_text(json.dumps(summary))
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"seed": 42, "criteria": CRITERIA,
                                                         "analysis": {"bootstrap_resamples": 40}}))
    return run_dir


def test_analysis_recovers_generating_model_and_crossovers(tmp_path):
    run_dir = make_synthetic_run(tmp_path)
    run = load_run(run_dir)
    a = analyze(run)
    assert a["selected_model"] == "M1"                      # data were generated from M1 features
    c = a["coefficients"]["M1"]
    assert c["to_compute"] == pytest.approx(TRUE["a_c"], rel=0.1)
    assert c["to_memory"] == pytest.approx(TRUE["a_m"], rel=0.1)
    assert c["commands"] == pytest.approx(TRUE["a_o"], rel=0.1)
    assert a["heldout_prediction"]["M1"]["median_ape"] < 0.03
    assert a["heldout_prediction"]["P0"]["median_ape"] > a["heldout_prediction"]["M1"]["median_ape"]
    assert a["n_windows"] == {"total": 360, "used": 360, "calibration": 240, "heldout": 120}
    # crossovers of the fitted model agree with the generating model's own crossovers
    truth = crossovers_from_energies({k: true_energy(v) for k, v in run["descriptors"].items()},
                                     {k: {"d": v["m1"]["d_in"], "f": None, "r": v["m1"]["r"] or 0, "B": v["m1"]["B"],
                                          "realization": v["m1"]["realization"]} for k, v in run["descriptors"].items()},
                                     GRID)
    for cell, t in truth.items():
        p = a["predicted_crossovers"]["M1"][cell]
        assert p["kind"] == t["kind"]
        if t["kind"] == "numeric":
            assert p["r_star"] == pytest.approx(t["r_star"], rel=0.15)
    assert set(a["coverage"]["cells"]) == {"d512_B1", "d512_B16", "d512_B256", "d512_B4096"}
    assert a["coverage"]["hits"] >= 3
    assert a["coverage"]["point_factor_hits"] == 4
    cell = a["coverage"]["cells"]["d512_B1"]
    assert cell["measured_interval"][0] <= cell["measured_r_star"] <= cell["measured_interval"][1]
    assert cell["predicted_interval"][0] <= cell["predicted_r_star"] <= cell["predicted_interval"][1]
    assert cell["predicted_r_star"] == pytest.approx(cell["measured_r_star"], rel=0.05)
    assert set(a["criteria"]) >= {"E1_heldout_median_ape", "E2_crossover_coverage", "E3_b1_gap",
                                  "E4_monotonicity", "E5_exactness_exclusions", "E6_model_regret",
                                  "E7_isolation_and_plausibility"}
    assert a["criteria"]["E8_m3_crossover_point_hits"]["gating"] is False   # criteria.m3 absent here
    assert a["criteria"]["E1_heldout_median_ape"]["pass"] and a["criteria"]["E7_isolation_and_plausibility"]["pass"]
    assert a["criteria"]["E6_model_regret"]["gating"] is False
    assert a["regret"]["model_regret_rel"] < 0.02 and a["regret"]["n_cells"] == 20
    assert "energy_per_mac_j" in a["derived_costs"]["M1"]
    assert a["derived_costs"]["M1"]["energy_per_command_j"] == pytest.approx(TRUE["a_o"], rel=0.1)
    assert a["valid"] is True
    json.dumps(a)   # fully serializable


def test_analysis_prefers_m2_when_data_come_from_m2(tmp_path):
    run_dir = make_synthetic_run(tmp_path, generating_variant="m2", l2_bytes=64 * 2**20, seed=5)
    a = analyze(load_run(run_dir))
    # M2 differs from M1 only where factorized weights fit L2; with a 64 MiB L2 and d up to 1024
    # every factorized set fits, so the M2 rule is exercised on every factorized configuration
    assert any(v["m2"]["weights_in_l2"] for v in load_run(run_dir)["descriptors"].values())
    assert a["selected_model"] == "M2"
    assert a["selection"]["cv"]["M2"]["mean_median_ape"] < a["selection"]["cv"]["M1"]["mean_median_ape"]


def test_contaminated_windows_are_dropped_and_fail_e7(tmp_path):
    run_dir = make_synthetic_run(tmp_path, contaminate=3)
    a = analyze(load_run(run_dir))
    assert a["n_windows"]["used"] == 357 and a["quality"]["dropped"]["contaminated"] == 3
    assert a["criteria"]["E7_isolation_and_plausibility"]["pass"] is False
    assert not gate_passes(a["criteria"])


def test_regret_counts_wrong_choices():
    meas = {"d8_f0.25_B1_dense": 1.0, "d8_f0.25_B1_factorized": 2.0,
            "d8_f0.5_B1_dense": 1.0, "d8_f0.5_B1_factorized": 0.5}
    pred_good = dict(meas)
    pred_bad = {"d8_f0.25_B1_dense": 3.0, "d8_f0.25_B1_factorized": 2.0,
                "d8_f0.5_B1_dense": 1.0, "d8_f0.5_B1_factorized": 0.5}
    grid = {"batches": [1], "rank_fractions": [0.25, 0.5]}
    r = regret(pred_good, meas, grid, [8])
    assert r["n_cells"] == 2 and r["model_regret_rel"] == 0.0 and r["model_wrong_choices"] == 0
    assert r["flops_regret_rel"] == pytest.approx((2.0 + 1.0 - 1.5) / 1.5)   # FLOPs picks factorized at f=0.25, dense at f=0.5
    r = regret(pred_bad, meas, grid, [8])
    assert r["model_wrong_choices"] == 1 and r["model_regret_rel"] == pytest.approx(1.0 / 1.5)


def test_evaluate_criteria_gating_and_reporting():
    a = {"selected_model": "M1",
         "heldout_prediction": {"M1": {"median_ape": 0.1}, "P0": {"median_ape": 0.6}},
         "coverage": {"hits": 3, "n_cells": 4, "point_factor_hits": 4, "point_factor": 1.5},
         "b1_gap": {"512": {"pass": True}, "1024": {"pass": False}},
         "monotonicity": {"512": {"rho": 0.9, "constant": False}, "1024": {"rho": 0.1, "constant": False}},
         "exclusions": {"fraction": 0.0},
         "regret": {"model_regret_rel": 0.2, "flops_regret_rel": 0.5},
         "quality": {"contaminated_windows": 0, "implausible_fraction": 0.0}}
    c = evaluate_criteria(a, CRITERIA)
    assert c["E6_model_regret"]["pass"] is False and c["E6_model_regret"]["gating"] is False
    assert c["E3_b1_gap"]["pass"] is False and c["E4_monotonicity"]["pass"] is False   # all shapes by default
    assert not gate_passes(c)
    # v3 switches: point rule for E2, and the B=1 and monotonicity clauses restricted to named shapes
    v3 = dict(CRITERIA, crossover_coverage_rule="point", crossover_coverage_min_cells=4,
              b1_gap_shapes=[512], monotonicity_shapes=[512])
    c3 = evaluate_criteria(a, v3)
    assert c3["E2_crossover_coverage"]["pass"] is True and c3["E2_crossover_coverage"]["value"] == "4 of 4"
    assert c3["E3_b1_gap"]["pass"] is True and c3["E4_monotonicity"]["pass"] is True
    assert gate_passes(c3)                                   # E6 does not gate
    a["monotonicity"]["512"] = {"rho": 0.3, "constant": False}
    assert not gate_passes(evaluate_criteria(a, v3))


M3_TRUE = {"t_launch_s": 12e-6, "bandwidth_Bps": 4e11, "throughput_MACps": 1e13,
           "p_floor_w": 40.0, "p_memory_w": 140.0, "p_compute_w": 175.0}


def make_m3_run(tmp_path: Path, noise: float = 0.02, register_m3: bool = True, seed: int = 11) -> Path:
    """Synthetic run generated from the max-of-times, regime-power form with
    laptop-like parameters, on the full pre-registered grid."""
    rng = np.random.default_rng(seed)
    grid = {"shapes": [512, 1024, 2048, 4096], "rank_fractions": [1/64, 1/32, 1/16, 1/8, 1/4, 3/8, 1/2, 3/4],
            "batches": [1, 4, 16, 64, 256, 1024, 4096], "realizations": ["dense", "factorized"], "reps": 3,
            "calibration_shapes": [512, 2048], "heldout_shapes": [1024, 4096]}
    run_dir = tmp_path / "m3run"
    (run_dir / "results").mkdir(parents=True)
    descriptors, windows = {}, []
    theta = [M3_TRUE[k] for k in ("t_launch_s", "bandwidth_Bps", "throughput_MACps", "p_floor_w", "p_memory_w", "p_compute_w")]
    for d in grid["shapes"]:
        for f in grid["rank_fractions"]:
            r = rank_for(d, f)
            for B in grid["batches"]:
                for rz in grid["realizations"]:
                    key = f"d{d}_f{f:g}_B{B}_{rz}"
                    if rz == "dense":
                        m1, cmds = dense_descriptor(B, d, d), 1.0
                    else:
                        m1, cmds = factorized_descriptor(B, d, d, r), 2.0
                    descriptors[key] = {"m1": m1, "m2": m1, "factorized_weights_fit_l2": False,
                                        "commands_per_call": cmds, "per_call_s_calibration": 1e-4}
                    e, t = m3_energy(theta, np.array([m1["n_mac"]], float),
                                     np.array([4 * sum(m1["words"].values())], float), np.array([cmds]))
                    for rep in range(grid["reps"]):
                        jitter = math.exp(rng.normal(0, noise))
                        val = float(e[0]) * jitter
                        windows.append({"key": key, "label": f"{key}_rep{rep}", "d": d, "f": f, "r": r, "B": B,
                                        "realization": rz, "rep": rep, "energy_per_call_j": val,
                                        "energy_per_call_integral_j": val, "energy_per_call_counter_j": None,
                                        "per_call_s": float(t[0]) * jitter,
                                        "regime": "uncapped", "contaminated": False, "n_samples": 400,
                                        "power_implausible_samples": 0, "counter_monotonic": None,
                                        "mean_temp_c": 60.0, "mean_sm_clock_mhz": 2000.0, "mean_power_w": 100.0,
                                        "duration_s": 20.0})
    summary = {"run_id": "m3synthetic", "platform_tag": "fake", "quick": False, "grid": grid,
               "settings": {"energy_source": "power_integral", "power_limit_enforced_w": 175.0}, "excluded": [],
               "contaminated_windows": 0, "samples_total": 400 * len(windows),
               "power_implausible_samples_total": 0, "l2_cache": {"bytes": 64 * 2**20}}
    crit = dict(CRITERIA, crossover_coverage_min_cells=12)
    if register_m3:
        crit["m3"] = {"crossover_point_hits_min": 12, "point_factor": 1.5,
                      "b1_small_d_fraction_max": 0.30, "b1_large_d_fraction_min": 0.45}
    (run_dir / "results" / "windows.json").write_text(json.dumps(windows))
    (run_dir / "results" / "descriptors.json").write_text(json.dumps(descriptors))
    (run_dir / "results" / "summary.json").write_text(json.dumps(summary))
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"seed": 42, "criteria": crit,
                                                         "analysis": {"bootstrap_resamples": 20}}))
    return run_dir


def test_m3_recovers_max_of_times_world_and_registered_criteria(tmp_path):
    a = analyze(load_run(make_m3_run(tmp_path)))
    th = a["m3"]["fit"]["theta"]
    assert a["m3"]["fit"]["fitted_on_time"] is True
    for k, v in M3_TRUE.items():
        assert th[k] == pytest.approx(v, rel=0.15), k
    assert a["m3"]["fit"]["calibration_median_ape_time"] < 0.05
    assert a["m3"]["heldout"]["median_ape"] < 0.06
    assert a["m3"]["registered"] is True
    assert a["m3"]["point_hits"] >= 12 and a["m3"]["n_cells"] == 14
    # the transition size follows from the fitted floor and bandwidth
    assert a["m3"]["transition"]["d_transition"] == pytest.approx(math.sqrt(12e-6 * 4e11 / 4), rel=0.3)
    c = a["criteria"]
    assert c["E8_m3_crossover_point_hits"]["gating"] is True and c["E8_m3_crossover_point_hits"]["pass"]
    assert c["E9_m3_heldout_median_ape"]["gating"] is False
    assert c["E10_b1_crossover_rises_with_d"]["gating"] is True
    # in this world the additive model cannot follow the floor: M3 beats it on held-out energy
    assert a["m3"]["heldout"]["median_ape"] < a["heldout_prediction"][a["selected_model"]]["median_ape"]
    assert c["E9_m3_heldout_median_ape"]["pass"] is True


def test_m3_is_exploratory_without_registration(tmp_path):
    a = analyze(load_run(make_m3_run(tmp_path, register_m3=False)))
    assert a["m3"]["registered"] is False
    c = a["criteria"]
    assert c["E8_m3_crossover_point_hits"]["gating"] is False
    assert "exploratory" in c["E8_m3_crossover_point_hits"]["note"]
    assert c["E10_b1_crossover_rises_with_d"]["gating"] is False
    # E1 to E7 still decide the gate on their own
    assert gate_passes({k: v for k, v in c.items() if not k.startswith(("E8", "E9", "E10"))}) == gate_passes(c)


def test_cli_writes_analysis_and_figures(tmp_path):
    pytest.importorskip("matplotlib")
    run_dir = make_synthetic_run(tmp_path)
    spec = importlib.util.spec_from_file_location("analyze_exp_002", REPO_ROOT / "scripts" / "analyze_exp_002.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main(["--run-dir", str(run_dir), "--resamples", "20"]) == 0
    a = json.loads((run_dir / "results" / "analysis.json").read_text())
    assert a["bootstrap"]["n_resamples"] == 20
    assert len(a["figures"]) == 4
    for name in ("energy_pred_vs_meas.png", "ratio_vs_rank.png", "crossover_vs_batch.png", "diagnostics.png"):
        assert (run_dir / "figures" / name).stat().st_size > 1000
    assert "analysis_code" in a
