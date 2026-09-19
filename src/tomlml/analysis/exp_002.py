"""EXP-002 analysis (pre-registered; LOGBOOK.md EXP-002, criteria E1 to E7).

Inputs are the frozen artifacts of a measurement run: ``results/windows.json``,
``results/descriptors.json``, ``results/summary.json`` and ``config.yaml``.
Outputs are ``results/analysis.json`` and figures. Nothing here reads the
held-out shapes before the model is selected on calibration cells only.

Fitting convention (declared here, before any full-run data was inspected):
non-negative least squares on calibration windows with relative weighting,
i.e. every row is scaled by 1/energy so the fit minimizes the sum of squared
relative errors. This matches the percentage-error metrics of the criteria
and keeps the milli-joule cells from being drowned by the joule cells.

Crossover convention: for one (d, B) cell, the ratio E_factorized / E_dense
is evaluated at the tested ranks (repetition means); r* is the rank at the
first upward crossing of 1 (log-linear interpolation in rank); ``none_dense``
if the ratio exceeds 1 at every tested rank, ``none_factorized`` if it is at
most 1 everywhere. For interval and monotonicity arithmetic the verdicts are
encoded on one scale: numeric r*, ``none_dense`` as 0, ``none_factorized``
as +inf.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.stats import spearmanr

from ..to_model.costs import TO

MODELS = ("P0", "M1", "M2")
BYTES_PER_WORD = 4


# --------------------------------------------------------------------------- #
# Loading and observations
# --------------------------------------------------------------------------- #

def load_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    results = run_dir / "results"
    with open(results / "windows.json", "r", encoding="utf-8") as f:
        windows = json.load(f)
    with open(results / "descriptors.json", "r", encoding="utf-8") as f:
        descriptors = json.load(f)
    with open(results / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    import yaml
    with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {"run_dir": run_dir, "windows": windows, "descriptors": descriptors,
            "summary": summary, "config": config}


def energy_field(source: str) -> str:
    if source == "counter":
        return "energy_per_call_counter_j"
    if source == "power_integral":
        return "energy_per_call_integral_j"
    return "energy_per_call_j"


def observations(windows: List[Dict[str, Any]], source: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Usable per-window observations (contaminated or energy-less windows
    dropped, counted) with the energy from the platform's source."""
    field = energy_field(source)
    obs: List[Dict[str, Any]] = []
    dropped = {"contaminated": 0, "no_energy": 0}
    for w in windows:
        if w.get("contaminated"):
            dropped["contaminated"] += 1
            continue
        e = w.get(field)
        if e is None or not (e > 0):
            dropped["no_energy"] += 1
            continue
        obs.append({"key": w["key"], "label": w["label"], "d": int(w["d"]), "f": float(w["f"]),
                    "r": int(w["r"]), "B": int(w["B"]), "realization": w["realization"],
                    "rep": int(w["rep"]), "energy": float(e), "regime": w.get("regime"),
                    "per_call_s": w.get("per_call_s"),
                    "mean_temp_c": w.get("mean_temp_c"), "mean_sm_clock_mhz": w.get("mean_sm_clock_mhz"),
                    "mean_power_w": w.get("mean_power_w"), "duration_s": w.get("duration_s")})
    return obs, dropped


def rep_means(obs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, List[float]] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for o in obs:
        by_key.setdefault(o["key"], []).append(o["energy"])
        meta.setdefault(o["key"], {k: o[k] for k in ("d", "f", "r", "B", "realization")})
    out: Dict[str, Dict[str, Any]] = {}
    for key, vals in by_key.items():
        arr = np.array(vals, float)
        out[key] = dict(meta[key], n=len(arr), mean=float(arr.mean()),
                        std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                        cv=float(arr.std(ddof=1) / arr.mean()) if len(arr) > 1 else None,
                        values=[float(v) for v in arr])
    return out


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def features(desc: Dict[str, Any], model: str) -> Tuple[np.ndarray, bool]:
    """Feature vector for one configuration. Returns (x, census_fallback).

    A command count that is missing or not positive is treated as unknown
    (every call launches at least one kernel; a zero means the profiler saw
    no device events) and replaced by the declared GEMM count."""
    m1 = desc["m1"]
    if model == "P0":
        return np.array([float(m1["n_mac"])]), False
    m = m1 if model == "M1" else desc["m2"]
    s = desc.get("commands_per_call")
    fallback = s is None or not (float(s) > 0)
    if fallback:
        s = float(m["n_gemm"])  # profiler unavailable: one command per GEMM
    return np.array([float(m["to_compute"]), float(m["to_memory"]), float(s)]), fallback


FEATURE_NAMES = {"P0": ["n_mac"], "M1": ["to_compute", "to_memory", "commands"],
                 "M2": ["to_compute", "to_memory", "commands"]}


def design(obs: List[Dict[str, Any]], descriptors: Dict[str, Any], model: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    rows = []
    fallback_any = False
    for o in obs:
        x, fb = features(descriptors[o["key"]], model)
        fallback_any |= fb
        rows.append(x)
    return np.vstack(rows), np.array([o["energy"] for o in obs], float), fallback_any


def fit_relative_nnls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares minimizing sum((X c - y) / y)^2."""
    Xs = X / y[:, None]
    coef, _ = nnls(Xs, np.ones_like(y))
    return coef


def predict(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return X @ coef


def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_pred - y_true) / y_true


def fit_and_predict(obs_fit: List[Dict[str, Any]], obs_pred: List[Dict[str, Any]],
                    descriptors: Dict[str, Any], model: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    X, y, fb1 = design(obs_fit, descriptors, model)
    coef = fit_relative_nnls(X, y)
    Xp, _, fb2 = design(obs_pred, descriptors, model)
    return coef, predict(Xp, coef), fb1 or fb2


def leave_one_cell_out(obs_cal: List[Dict[str, Any]], descriptors: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Cross-validated median APE over calibration (d, B) cells."""
    cells = sorted({(o["d"], o["B"]) for o in obs_cal})
    scores = []
    for cell in cells:
        train = [o for o in obs_cal if (o["d"], o["B"]) != cell]
        test = [o for o in obs_cal if (o["d"], o["B"]) == cell]
        if not train or not test:
            continue
        _, pred, _ = fit_and_predict(train, test, descriptors, model)
        scores.append(float(np.median(ape(np.array([o["energy"] for o in test]), pred))))
    return {"cells": [list(c) for c in cells], "fold_median_ape": scores,
            "mean_median_ape": float(np.mean(scores)) if scores else None}


def select_model(obs_cal: List[Dict[str, Any]], descriptors: Dict[str, Any],
                 candidates: Sequence[str] = ("M1", "M2")) -> Dict[str, Any]:
    cv = {m: leave_one_cell_out(obs_cal, descriptors, m) for m in candidates}
    scored = [(cv[m]["mean_median_ape"], i, m) for i, m in enumerate(candidates)
              if cv[m]["mean_median_ape"] is not None]
    selected = min(scored)[2] if scored else candidates[0]   # ties: earlier candidate (M1, simpler rule)
    return {"selected": selected, "cv": cv}


# --------------------------------------------------------------------------- #
# M3: max-of-times with regime power (registered after the laptop run, D-012)
# --------------------------------------------------------------------------- #

M3_PARAMS = ("t_launch_s", "bandwidth_Bps", "throughput_MACps", "p_floor_w", "p_memory_w", "p_compute_w")


def m3_inputs(desc: Dict[str, Any]) -> Tuple[float, float, float]:
    """(MACs, bytes moved, commands) for one configuration; a missing or
    non-positive command count falls back to the declared GEMM count."""
    m = desc["m1"]
    macs = float(m["n_mac"])
    byt = float(sum(m["words"].values())) * BYTES_PER_WORD
    s = desc.get("commands_per_call")
    cmds = float(s) if (s is not None and float(s) > 0) else float(m["n_gemm"])
    return macs, byt, cmds


def m3_energy(theta: Sequence[float], macs: np.ndarray, byt: np.ndarray, cmds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """E = p_floor * t_call + (p_active - p_floor) * t_active, with
    t_active = max(bytes/BW, MACs/R), t_call = max(cmds * t_launch, t_active),
    p_active = p_compute where compute time dominates, else p_memory."""
    t_l, bw, r, p0, pm, pc = theta
    t_mem = byt / bw
    t_comp = macs / r
    t_act = np.maximum(t_mem, t_comp)
    t_call = np.maximum(cmds * t_l, t_act)
    p_act = np.where(t_comp >= t_mem, pc, pm)
    return p0 * t_call + (p_act - p0) * t_act, t_call


def fit_m3(obs_fit: List[Dict[str, Any]], descriptors: Dict[str, Any],
           p_cap_w: Optional[float] = None) -> Dict[str, Any]:
    """Nonlinear least squares on log relative error of energy and, where the
    windows carry it, of per-call time (equal weight). The model predicts
    both, and the time residuals pin the launch floor and the floor power
    separately, which energy alone cannot (a long floor at low power and a
    short one at high power give the same energy).

    Powers are parameterized as ordered fractions within the cap, p_floor <=
    p_memory <= p_compute <= cap, and the fit is multi-started from a grid of
    physically spread initial values, keeping the lowest cost.
    """
    X = np.array([m3_inputs(descriptors[o["key"]]) for o in obs_fit], float)
    y = np.array([o["energy"] for o in obs_fit], float)
    t_obs = np.array([o.get("per_call_s") or np.nan for o in obs_fit], float)
    have_t = np.isfinite(t_obs) & (t_obs > 0)
    macs, byt, cmds = X[:, 0], X[:, 1], X[:, 2]
    p_hi = float(p_cap_w) * 1.05 if p_cap_w else 1000.0

    def unpack(z: np.ndarray) -> List[float]:
        # ordered powers within the cap: p0 <= p_mem <= p_comp <= p_hi
        t_l, bw, r, p0, s1, s2 = z
        p_mem = p0 + (p_hi - p0) * s1
        p_comp = p_mem + (p_hi - p_mem) * s2
        return [t_l, bw, r, p0, p_mem, p_comp]

    def resid(z: np.ndarray) -> np.ndarray:
        e, t = m3_energy(unpack(z), macs, byt, cmds)
        r_e = np.log(np.maximum(e, 1e-12) / y)
        if have_t.any():
            r_t = np.log(np.maximum(t[have_t], 1e-12) / t_obs[have_t])
            return np.concatenate([r_e, r_t])
        return r_e

    lb = [1e-7, 1e9, 1e10, 1.0, 0.0, 0.0]
    ub = [1e-3, 1e13, 1e15, p_hi, 1.0, 1.0]
    starts = []
    for t_l in (5e-6, 12e-6, 25e-6):
        for bw in (2e11, 5e11, 1.5e12):
            for r in (2e12, 1e13, 3e13):
                for p0 in (0.15 * p_hi, 0.35 * p_hi):
                    starts.append([t_l, bw, r, p0, 0.6, 0.6])
    best = None
    for z0 in starts:
        z0 = [min(max(v, lo), hi) for v, lo, hi in zip(z0, lb, ub)]
        try:
            fit = least_squares(resid, z0, bounds=(lb, ub))
        except ValueError:
            continue
        if best is None or fit.cost < best.cost:
            best = fit
    theta = unpack(best.x)
    e_fit, t_fit = m3_energy(theta, macs, byt, cmds)
    out = {"theta": dict(zip(M3_PARAMS, [float(v) for v in theta])), "cost": float(best.cost),
           "success": bool(best.success), "n_obs": int(len(y)), "n_starts": len(starts),
           "fitted_on_time": bool(have_t.any()),
           "calibration_median_ape_energy": float(np.median(np.abs(e_fit - y) / y))}
    if have_t.any():
        out["calibration_median_ape_time"] = float(np.median(np.abs(t_fit[have_t] - t_obs[have_t]) / t_obs[have_t]))
    return out


def m3_predict(theta: Dict[str, float], descriptors: Dict[str, Any], keys: Sequence[str]) -> Dict[str, float]:
    X = np.array([m3_inputs(descriptors[k]) for k in keys], float)
    e, _ = m3_energy([theta[p] for p in M3_PARAMS], X[:, 0], X[:, 1], X[:, 2])
    return dict(zip(keys, [float(v) for v in e]))


def m3_transition_size(theta: Dict[str, float], d_range: Sequence[int] = (256, 8192)) -> Dict[str, Any]:
    """Square map size at which the dense B=1 weight read time equals the
    single-launch floor: d_t = sqrt(t_launch * BW / 4 bytes). Below it the
    B=1 crossover sits far below d/2; above it, near d/2."""
    d_t = math.sqrt(theta["t_launch_s"] * theta["bandwidth_Bps"] / BYTES_PER_WORD)
    return {"d_transition": d_t, "within_range": bool(d_range[0] <= d_t <= d_range[1])}


# --------------------------------------------------------------------------- #
# Crossovers
# --------------------------------------------------------------------------- #

def crossover_from_ratios(ranks: Sequence[int], ratios: Sequence[float]) -> Dict[str, Any]:
    order = np.argsort(ranks)
    r = np.array(ranks, float)[order]
    q = np.log(np.array(ratios, float)[order])
    if np.all(q > 0):
        return {"kind": "none_dense", "r_star": None, "n_crossings": 0, "monotone": bool(np.all(np.diff(q) >= 0))}
    if np.all(q <= 0):
        return {"kind": "none_factorized", "r_star": None, "n_crossings": 0,
                "monotone": bool(np.all(np.diff(q) >= 0))}
    crossings = []
    for i in range(len(r) - 1):
        if q[i] <= 0 < q[i + 1]:
            # log-linear interpolation in rank where log ratio = 0
            t = (0 - q[i]) / (q[i + 1] - q[i])
            crossings.append(("up", float(np.exp(np.log(r[i]) + t * (np.log(r[i + 1]) - np.log(r[i]))))))
        elif q[i] > 0 >= q[i + 1]:
            crossings.append(("down", None))
    ups = [c for kind, c in crossings if kind == "up"]
    if not ups:
        # starts above 1, ends at or below 1: dense wins at small rank, factorized at large;
        # no "factorize below r*" rule exists; report as none_dense at the low end
        return {"kind": "none_dense", "r_star": None, "n_crossings": len(crossings), "monotone": False}
    return {"kind": "numeric", "r_star": ups[0], "n_crossings": len(crossings),
            "monotone": bool(np.all(np.diff(q) >= 0))}


def encode(cross: Dict[str, Any]) -> float:
    if cross["kind"] == "numeric":
        return float(cross["r_star"])
    return 0.0 if cross["kind"] == "none_dense" else math.inf


def crossovers_from_energies(energy_by_key: Dict[str, float], meta_by_key: Dict[str, Dict[str, Any]],
                             grid: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per (d, B) cell: crossover from per-configuration energies (measured
    repetition means or model predictions). Cell keys are 'd{d}_B{B}'."""
    out: Dict[str, Dict[str, Any]] = {}
    for d in [int(x) for x in grid["shapes"]]:
        for B in [int(x) for x in grid["batches"]]:
            ranks, ratios = [], []
            for f in [float(x) for x in grid["rank_fractions"]]:
                kd = f"d{d}_f{f:g}_B{B}_dense"
                kf = f"d{d}_f{f:g}_B{B}_factorized"
                if kd in energy_by_key and kf in energy_by_key:
                    ranks.append(meta_by_key[kf]["r"])
                    ratios.append(energy_by_key[kf] / energy_by_key[kd])
            if len(ranks) >= 2:
                c = crossover_from_ratios(ranks, ratios)
                c.update(d=d, B=B, ranks=ranks, ratios=[float(x) for x in ratios])
                out[f"d{d}_B{B}"] = c
    return out


def flops_crossovers(grid: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for d in [int(x) for x in grid["shapes"]]:
        for B in [int(x) for x in grid["batches"]]:
            out[f"d{d}_B{B}"] = {"kind": "numeric", "r_star": d / 2.0, "d": d, "B": B, "n_crossings": 1,
                                 "monotone": True}
    return out


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_crossovers(obs_cal: List[Dict[str, Any]], heldout_obs: List[Dict[str, Any]],
                         descriptors: Dict[str, Any], model: str, grid: Dict[str, Any],
                         n_resamples: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Resample calibration windows within each configuration, refit, predict
    every held-out configuration, recompute crossovers. Returns per held-out
    cell the 95 percent interval on the encoded scale plus verdict fractions."""
    by_key: Dict[str, List[int]] = {}
    for i, o in enumerate(obs_cal):
        by_key.setdefault(o["key"], []).append(i)
    heldout_keys = sorted({o["key"] for o in heldout_obs})
    meta = {o["key"]: {k: o[k] for k in ("d", "f", "r", "B", "realization")} for o in heldout_obs}
    Xh = np.vstack([features(descriptors[k], model)[0] for k in heldout_keys])
    hgrid = dict(grid, shapes=sorted({meta[k]["d"] for k in heldout_keys}))
    X_all, y_all, _ = design(obs_cal, descriptors, model)
    encoded: Dict[str, List[float]] = {}
    coefs = []
    for _ in range(int(n_resamples)):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_key.values()])
        coef = fit_relative_nnls(X_all[idx], y_all[idx])
        coefs.append(coef)
        pred = dict(zip(heldout_keys, predict(Xh, coef)))
        for cell, c in crossovers_from_energies(pred, meta, hgrid).items():
            encoded.setdefault(cell, []).append(encode(c))
    out: Dict[str, Any] = {"n_resamples": int(n_resamples), "cells": {}}
    for cell, vals in encoded.items():
        v = np.array(vals, float)
        out["cells"][cell] = {
            "lo": float(np.quantile(v, 0.025, method="lower")),
            "hi": float(np.quantile(v, 0.975, method="higher")),
            "median": float(np.quantile(v, 0.5, method="nearest")),
            "frac_none_dense": float(np.mean(v == 0.0)),
            "frac_none_factorized": float(np.mean(np.isinf(v))),
        }
    C = np.vstack(coefs) if coefs else np.zeros((0, Xh.shape[1]))
    out["coef_quantiles"] = {name: {"lo": float(np.quantile(C[:, j], 0.025)), "hi": float(np.quantile(C[:, j], 0.975))}
                             for j, name in enumerate(FEATURE_NAMES[model])} if len(coefs) else {}
    return out


def bootstrap_measured_crossovers(obs_held: List[Dict[str, Any]], grid: Dict[str, Any],
                                  n_resamples: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Resample the held-out repetitions within each configuration, recompute
    the measured crossovers; per cell the 95 percent interval on the encoded
    scale (measurement uncertainty of the crossover)."""
    by_key: Dict[str, List[float]] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for o in obs_held:
        by_key.setdefault(o["key"], []).append(o["energy"])
        meta.setdefault(o["key"], {k: o[k] for k in ("d", "f", "r", "B", "realization")})
    hgrid = dict(grid, shapes=sorted({m["d"] for m in meta.values()}))
    encoded: Dict[str, List[float]] = {}
    for _ in range(int(n_resamples)):
        means = {k: float(np.mean(rng.choice(v, size=len(v), replace=True))) for k, v in by_key.items()}
        for cell, c in crossovers_from_energies(means, meta, hgrid).items():
            encoded.setdefault(cell, []).append(encode(c))
    out: Dict[str, Any] = {"n_resamples": int(n_resamples), "cells": {}}
    for cell, vals in encoded.items():
        v = np.array(vals, float)
        out["cells"][cell] = {
            "lo": float(np.quantile(v, 0.025, method="lower")),
            "hi": float(np.quantile(v, 0.975, method="higher")),
            "frac_none_dense": float(np.mean(v == 0.0)),
            "frac_none_factorized": float(np.mean(np.isinf(v))),
        }
    return out


def interval_hit(measured: Dict[str, Any], lo: float, hi: float) -> bool:
    m = encode(measured)
    return bool(lo <= m <= hi)


def intervals_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return bool(max(lo_a, lo_b) <= min(hi_a, hi_b))


def point_factor_hit(measured: Dict[str, Any], predicted: Dict[str, Any], factor: float) -> bool:
    """Secondary rule: same verdict kind, and numeric r* within ``factor`` of each other."""
    if measured["kind"] != predicted["kind"]:
        return False
    if measured["kind"] != "numeric":
        return True
    ratio = predicted["r_star"] / measured["r_star"]
    return bool(1.0 / factor <= ratio <= factor)


# --------------------------------------------------------------------------- #
# Regret
# --------------------------------------------------------------------------- #

def regret(pred_by_key: Dict[str, float], meas_by_key: Dict[str, float],
           grid: Dict[str, Any], shapes: Sequence[int]) -> Dict[str, Any]:
    tot_min = tot_model = tot_flops = 0.0
    n_cells = 0
    wrong_model = wrong_flops = 0
    cell_rel_model: List[float] = []
    cell_rel_flops: List[float] = []
    for d in shapes:
        for B in [int(x) for x in grid["batches"]]:
            for f in [float(x) for x in grid["rank_fractions"]]:
                kd, kf = f"d{d}_f{f:g}_B{B}_dense", f"d{d}_f{f:g}_B{B}_factorized"
                if kd not in meas_by_key or kf not in meas_by_key or kd not in pred_by_key or kf not in pred_by_key:
                    continue
                ed, ef = meas_by_key[kd], meas_by_key[kf]
                best = min(ed, ef)
                choice_model = ef if pred_by_key[kf] < pred_by_key[kd] else ed
                choice_flops = ef if f < 0.5 else ed
                tot_min += best
                tot_model += choice_model
                tot_flops += choice_flops
                wrong_model += int(choice_model > best)
                wrong_flops += int(choice_flops > best)
                cell_rel_model.append((choice_model - best) / best)
                cell_rel_flops.append((choice_flops - best) / best)
                n_cells += 1
    return {"n_cells": n_cells, "oracle_energy_j": tot_min,
            "model_regret_rel": (tot_model - tot_min) / tot_min if tot_min > 0 else None,
            "flops_regret_rel": (tot_flops - tot_min) / tot_min if tot_min > 0 else None,
            "model_regret_cell_mean_rel": float(np.mean(cell_rel_model)) if cell_rel_model else None,
            "flops_regret_cell_mean_rel": float(np.mean(cell_rel_flops)) if cell_rel_flops else None,
            "model_regret_cell_max_rel": float(np.max(cell_rel_model)) if cell_rel_model else None,
            "flops_regret_cell_max_rel": float(np.max(cell_rel_flops)) if cell_rel_flops else None,
            "model_wrong_choices": wrong_model, "flops_wrong_choices": wrong_flops}


# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #

def spearman_monotone(batches: Sequence[int], encoded: Sequence[float]) -> Dict[str, Any]:
    finite_max = max([v for v in encoded if np.isfinite(v)] + [1.0])
    y = [v if np.isfinite(v) else finite_max * 10.0 for v in encoded]
    if len(set(y)) <= 1:
        return {"rho": 1.0, "constant": True, "n_decreases": 0}
    rho = spearmanr(list(batches), y).correlation
    n_dec = int(sum(1 for a, b in zip(y, y[1:]) if b < a))
    return {"rho": float(rho) if rho == rho else None, "constant": False, "n_decreases": n_dec}


def evaluate_criteria(a: Dict[str, Any], crit: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    def add(key: str, passed: Any, value: Any, threshold: Any, note: Optional[str] = None,
            gating: bool = True) -> None:
        out[key] = {"pass": bool(passed), "value": value, "threshold": threshold, "note": note, "gating": gating}

    sel = a["selected_model"]
    e1 = a["heldout_prediction"][sel]["median_ape"]
    add("E1_heldout_median_ape", e1 is not None and e1 <= crit["heldout_median_ape_max"], e1,
        crit["heldout_median_ape_max"], note=f"selected model {sel}; P0 median APE {a['heldout_prediction']['P0']['median_ape']}")

    cov = a["coverage"]
    rule = crit.get("crossover_coverage_rule", "interval")
    cov_hits = cov["point_factor_hits"] if rule == "point" else cov["hits"]
    add("E2_crossover_coverage", cov_hits >= crit["crossover_coverage_min_cells"],
        f"{cov_hits} of {cov['n_cells']}", crit["crossover_coverage_min_cells"],
        note=(f"rule: {rule}; interval overlap {cov['hits']} of {cov['n_cells']}, point rule (factor "
              f"{cov.get('point_factor')}) {cov.get('point_factor_hits')} of {cov['n_cells']}"))

    b1 = a["b1_gap"]
    b1_shapes = [str(d) for d in crit.get("b1_gap_shapes", [])] or list(b1.keys())
    b1_sel = {d: b1[d] for d in b1_shapes if d in b1}
    add("E3_b1_gap", all(c["pass"] for c in b1_sel.values()) if b1_sel else False,
        {d: c["pass"] for d, c in b1_sel.items()},
        f"r* at B=1 at most d x {crit['b1_rank_fraction_max']} (or none_dense), model on the same side; shapes {b1_shapes}")

    mono = a["monotonicity"]
    mono_shapes = [str(d) for d in crit.get("monotonicity_shapes", [])] or list(mono.keys())
    mono_sel = {d: mono[d] for d in mono_shapes if d in mono}
    add("E4_monotonicity", all((m["rho"] is not None and m["rho"] >= crit["monotonicity_spearman_min"]) or m["constant"]
                               for m in mono_sel.values()) if mono_sel else False,
        {d: m["rho"] for d, m in mono_sel.items()}, f"{crit['monotonicity_spearman_min']} on shapes {mono_shapes}")

    ex = a["exclusions"]
    add("E5_exactness_exclusions", ex["fraction"] <= crit["exactness_max_excluded_fraction"],
        ex["fraction"], crit["exactness_max_excluded_fraction"])

    reg = a["regret"]["model_regret_rel"]
    add("E6_model_regret", reg is not None and reg <= crit["regret_max"], reg, crit["regret_max"],
        note=f"FLOPs regret {a['regret']['flops_regret_rel']}", gating=False)

    q = a["quality"]
    add("E7_isolation_and_plausibility",
        q["contaminated_windows"] <= crit["contaminated_windows_max"]
        and q["implausible_fraction"] <= crit["implausible_power_fraction_max"],
        {"contaminated_windows": q["contaminated_windows"], "implausible_fraction": q["implausible_fraction"]},
        {"contaminated_windows_max": crit["contaminated_windows_max"],
         "implausible_power_fraction_max": crit["implausible_power_fraction_max"]})

    # M3 criteria (D-012): registered for the A100 via the config; exploratory
    # where the run's frozen config lacks a criteria.m3 block
    m3crit = crit.get("m3")
    m3 = a.get("m3")
    if m3 is not None:
        gating = m3crit is not None
        c = m3crit or {"crossover_point_hits_min": 12, "point_factor": 1.5,
                       "b1_small_d_fraction_max": 0.30, "b1_large_d_fraction_min": 0.45}
        note = None if gating else "exploratory: criteria.m3 absent from this run's frozen config"
        add("E8_m3_crossover_point_hits", m3["point_hits"] >= c["crossover_point_hits_min"],
            f"{m3['point_hits']} of {m3['n_cells']}", c["crossover_point_hits_min"], note=note, gating=gating)
        add("E9_m3_heldout_median_ape", m3["heldout"]["median_ape"] is not None
            and a["heldout_prediction"][sel]["median_ape"] is not None
            and m3["heldout"]["median_ape"] <= a["heldout_prediction"][sel]["median_ape"],
            m3["heldout"]["median_ape"], f"at most the selected linear model's {a['heldout_prediction'][sel]['median_ape']}",
            note=(note or "") + " reported, not gating", gating=False)
        b1 = m3["b1_structure"]
        add("E10_b1_crossover_rises_with_d", b1["pass"], b1["values"],
            {"smallest_heldout_shape_max": c["b1_small_d_fraction_max"],
             "largest_heldout_shape_min": c["b1_large_d_fraction_min"]}, note=note, gating=gating)
    return out


def gate_passes(criteria: Dict[str, Dict[str, Any]]) -> bool:
    return all(c["pass"] for c in criteria.values() if c.get("gating", True))


# --------------------------------------------------------------------------- #
# Full analysis
# --------------------------------------------------------------------------- #

def analyze(run: Dict[str, Any], n_resamples: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    config = run["config"]
    grid = run["summary"]["grid"]
    descriptors = run["descriptors"]
    crit = config["criteria"]
    acfg = config.get("analysis", {})
    n_boot = int(n_resamples if n_resamples is not None else acfg.get("bootstrap_resamples", 1000))
    source = run["summary"].get("settings", {}).get("energy_source", "auto")
    obs, dropped = observations(run["windows"], source)
    cal_shapes = [int(x) for x in grid["calibration_shapes"]]
    held_shapes = [int(x) for x in grid["heldout_shapes"]]
    obs_cal = [o for o in obs if o["d"] in cal_shapes]
    obs_held = [o for o in obs if o["d"] in held_shapes]
    means = rep_means(obs)
    meas_by_key = {k: v["mean"] for k, v in means.items()}
    meta_by_key = {k: {kk: v[kk] for kk in ("d", "f", "r", "B", "realization")} for k, v in means.items()}
    from ..utils.seeds import derive_seed
    master = int(seed if seed is not None else config.get("seed", 42))
    rng = np.random.default_rng(derive_seed(master, str(acfg.get("bootstrap_seed_component", "bootstrap"))))

    # model selection on calibration cells only
    selection = select_model(obs_cal, descriptors, ("M1", "M2"))
    sel = selection["selected"]

    # fits on all calibration windows, predictions on held-out windows
    heldout_prediction: Dict[str, Any] = {}
    coefs: Dict[str, Any] = {}
    pred_by_key: Dict[str, Dict[str, float]] = {}
    census_fallback = False
    all_keys = sorted(means.keys())
    for model in MODELS:
        coef, pred_h, fb = fit_and_predict(obs_cal, obs_held, descriptors, model) if obs_held else (
            fit_relative_nnls(*design(obs_cal, descriptors, model)[:2]), np.array([]), False)
        census_fallback |= fb
        coefs[model] = {name: float(c) for name, c in zip(FEATURE_NAMES[model], coef)}
        X_all = np.vstack([features(descriptors[k], model)[0] for k in all_keys])
        pred_by_key[model] = dict(zip(all_keys, predict(X_all, coef).tolist()))
        if obs_held:
            y = np.array([o["energy"] for o in obs_held])
            a = ape(y, pred_h)
            cal_X, cal_y, _ = design(obs_cal, descriptors, model)
            a_cal = ape(cal_y, predict(cal_X, coef))
            heldout_prediction[model] = {
                "median_ape": float(np.median(a)), "max_ape": float(np.max(a)),
                "p90_ape": float(np.quantile(a, 0.9)), "n_windows": int(len(a)),
                "calibration_median_ape": float(np.median(a_cal)),
                "per_key_ape": {k: float(np.median(a[[i for i, o in enumerate(obs_held) if o["key"] == k]]))
                                for k in sorted({o["key"] for o in obs_held})},
            }
        else:
            heldout_prediction[model] = {"median_ape": None, "max_ape": None, "p90_ape": None, "n_windows": 0,
                                         "calibration_median_ape": None, "per_key_ape": {}}
    derived = {}
    if "M1" in coefs:
        derived = {m: {"energy_per_mac_j": coefs[m]["to_compute"] * TO["mac"],
                       "energy_per_hbm_word_j": coefs[m]["to_memory"] * TO["mem_hbm"],
                       "energy_per_command_j": coefs[m]["commands"]} for m in ("M1", "M2")}

    # crossovers
    measured = crossovers_from_energies(meas_by_key, meta_by_key, grid)
    predicted = {m: crossovers_from_energies(pred_by_key[m], meta_by_key, grid) for m in MODELS}
    predicted["FLOPs"] = flops_crossovers(grid)
    boot = bootstrap_crossovers(obs_cal, obs_held, descriptors, sel, grid, n_boot, rng) if obs_held else \
        {"n_resamples": 0, "cells": {}, "coef_quantiles": {}}
    boot_meas = bootstrap_measured_crossovers(obs_held, grid, n_boot, rng) if obs_held else \
        {"n_resamples": 0, "cells": {}}

    # coverage on held-out cells (E2, amended rule: the prediction interval and the
    # measurement interval overlap on the encoded scale; point rule reported)
    factor = float(crit.get("crossover_point_factor_max", 1.5))
    cov_cells = {}
    hits = 0
    point_hits = 0
    for d in held_shapes:
        for B in [int(x) for x in grid["batches"]]:
            cell = f"d{d}_B{B}"
            if cell not in measured or cell not in boot["cells"] or cell not in boot_meas["cells"]:
                continue
            bp, bm = boot["cells"][cell], boot_meas["cells"][cell]
            hit = intervals_overlap(bp["lo"], bp["hi"], bm["lo"], bm["hi"])
            phit = point_factor_hit(measured[cell], predicted[sel][cell], factor)
            hits += int(hit)
            point_hits += int(phit)
            cov_cells[cell] = {"measured": measured[cell]["kind"], "measured_r_star": measured[cell]["r_star"],
                               "measured_interval": [bm["lo"], bm["hi"]],
                               "predicted": predicted[sel][cell]["kind"], "predicted_r_star": predicted[sel][cell]["r_star"],
                               "predicted_interval": [bp["lo"], bp["hi"]],
                               "hit": hit, "point_factor_hit": phit,
                               "measured_in_predicted_interval": interval_hit(measured[cell], bp["lo"], bp["hi"]),
                               "flops_r_star": d / 2.0}
    coverage = {"n_cells": len(cov_cells), "hits": hits, "point_factor_hits": point_hits,
                "point_factor": factor, "cells": cov_cells}

    # B = 1 gap (E3)
    fmax = float(crit["b1_rank_fraction_max"])
    b1_gap = {}
    for d in held_shapes:
        cell = f"d{d}_B1"
        if cell in measured and cell in predicted[sel]:
            m_enc, p_enc = encode(measured[cell]), encode(predicted[sel][cell])
            b1_gap[str(d)] = {"measured": measured[cell]["kind"], "measured_r_star": measured[cell]["r_star"],
                              "predicted": predicted[sel][cell]["kind"], "predicted_r_star": predicted[sel][cell]["r_star"],
                              "threshold_rank": d * fmax,
                              "pass": bool(m_enc <= d * fmax and p_enc <= d * fmax)}

    # monotonicity (E4)
    monotonicity = {}
    for d in held_shapes:
        bs = [int(x) for x in grid["batches"]]
        enc = [encode(measured[f"d{d}_B{B}"]) for B in bs if f"d{d}_B{B}" in measured]
        if enc:
            monotonicity[str(d)] = dict(spearman_monotone(bs[:len(enc)], enc), encoded=[e if np.isfinite(e) else None for e in enc])

    # exclusions (E5), quality (E7), regret (E6)
    n_configs_total = len(grid["shapes"]) * len(grid["rank_fractions"]) * len(grid["batches"])
    excluded = run["summary"].get("excluded", [])
    exclusions = {"n_excluded": len(excluded), "n_configs": n_configs_total,
                  "fraction": len(excluded) / n_configs_total if n_configs_total else 0.0}
    n_samples = run["summary"].get("samples_total") or 0
    quality = {"contaminated_windows": int(run["summary"].get("contaminated_windows", dropped["contaminated"])),
               "implausible_fraction": (run["summary"].get("power_implausible_samples_total", 0) / n_samples) if n_samples else 0.0,
               "dropped": dropped, "n_windows_used": len(obs)}
    reg = regret(pred_by_key[sel], meas_by_key, grid, held_shapes)

    # M3 (D-012): fitted on calibration windows, scored on held-out shapes
    m3: Optional[Dict[str, Any]] = None
    if obs_cal:
        p_cap = run["summary"].get("settings", {}).get("power_limit_enforced_w")
        fit3 = fit_m3(obs_cal, descriptors, p_cap_w=p_cap)
        pred3 = m3_predict(fit3["theta"], descriptors, all_keys)
        pred_by_key["M3"] = pred3
        cross3 = crossovers_from_energies(pred3, meta_by_key, grid)
        predicted["M3"] = cross3
        c3 = crit.get("m3") or {}
        factor3 = float(c3.get("point_factor", crit.get("crossover_point_factor_max", 1.5)))
        hits3 = 0
        cells3 = {}
        for d in held_shapes:
            for B in [int(x) for x in grid["batches"]]:
                cell = f"d{d}_B{B}"
                if cell in measured and cell in cross3:
                    hit = point_factor_hit(measured[cell], cross3[cell], factor3)
                    hits3 += int(hit)
                    cells3[cell] = {"measured": measured[cell]["kind"], "measured_r_star": measured[cell]["r_star"],
                                    "predicted": cross3[cell]["kind"], "predicted_r_star": cross3[cell]["r_star"],
                                    "hit": hit}
        held3 = {"median_ape": None, "p90_ape": None, "max_ape": None, "n_windows": 0}
        if obs_held:
            y = np.array([o["energy"] for o in obs_held])
            e3 = np.array([pred3[o["key"]] for o in obs_held])
            a3 = ape(y, e3)
            held3 = {"median_ape": float(np.median(a3)), "p90_ape": float(np.quantile(a3, 0.9)),
                     "max_ape": float(np.max(a3)), "n_windows": int(len(a3)),
                     "by_batch": {str(B): float(np.median(a3[[i for i, o in enumerate(obs_held) if o["B"] == B]]))
                                  for B in sorted({o["B"] for o in obs_held})}}
        small_max = float(c3.get("b1_small_d_fraction_max", 0.30))
        large_min = float(c3.get("b1_large_d_fraction_min", 0.45))
        b1_vals: Dict[str, Any] = {}
        b1_pass = bool(held_shapes)
        if held_shapes:
            d_small, d_large = min(held_shapes), max(held_shapes)
            for d in (d_small, d_large):
                cell = f"d{d}_B1"
                mv = encode(measured[cell]) / d if cell in measured else None
                pv = encode(cross3[cell]) / d if cell in cross3 else None
                b1_vals[str(d)] = {"measured_fraction": None if mv is None or not np.isfinite(mv) else mv,
                                   "predicted_fraction": None if pv is None or not np.isfinite(pv) else pv}
            ms, ml = b1_vals[str(d_small)]["measured_fraction"], b1_vals[str(d_large)]["measured_fraction"]
            b1_pass = (ms is not None and ms <= small_max) and (ml is not None and ml >= large_min)
        m3 = {"fit": fit3, "transition": m3_transition_size(fit3["theta"]),
              "heldout": held3, "point_factor": factor3, "point_hits": hits3, "n_cells": len(cells3),
              "cells": cells3, "b1_structure": {"pass": b1_pass, "values": b1_vals,
                                                "small_d_fraction_max": small_max, "large_d_fraction_min": large_min},
              "registered": crit.get("m3") is not None}

    analysis: Dict[str, Any] = {
        "run_id": run["summary"].get("run_id"), "platform_tag": run["summary"].get("platform_tag"),
        "energy_source": source, "quick": run["summary"].get("quick"),
        "calibration_shapes": cal_shapes, "heldout_shapes": held_shapes,
        "n_windows": {"total": len(run["windows"]), "used": len(obs), "calibration": len(obs_cal), "heldout": len(obs_held)},
        "census_fallback_used": census_fallback,
        "selected_model": sel, "selection": selection,
        "coefficients": coefs, "coefficient_quantiles_selected": boot.get("coef_quantiles", {}),
        "derived_costs": derived,
        "heldout_prediction": heldout_prediction,
        "predicted_energy_by_key": pred_by_key,
        "rep_stats": {k: {kk: v[kk] for kk in ("n", "mean", "std", "cv")} for k, v in means.items()},
        "measured_crossovers": measured,
        "predicted_crossovers": predicted,
        "bootstrap": {"n_resamples": boot["n_resamples"], "cells": boot["cells"],
                      "measured_cells": boot_meas["cells"]},
        "coverage": coverage,
        "b1_gap": b1_gap,
        "monotonicity": monotonicity,
        "exclusions": exclusions,
        "quality": quality,
        "regret": reg,
        "m3": m3,
    }
    analysis["criteria"] = evaluate_criteria(analysis, crit)
    analysis["h2a_supported"] = gate_passes(analysis["criteria"])
    analysis["valid"] = not run["summary"].get("quick", False)
    return analysis


def criteria_table(analysis: Dict[str, Any]) -> str:
    lines = [f"selected model: {analysis['selected_model']} (energy source {analysis['energy_source']})"]
    for k, c in analysis["criteria"].items():
        flag = "PASS" if c["pass"] else "FAIL"
        gate = "" if c.get("gating", True) else " [reported, not gating]"
        lines.append(f"  {flag} {k}: value={c['value']} threshold={c['threshold']}{gate}")
    lines.append(f"H2a on this platform: {'SUPPORTED' if analysis['h2a_supported'] else 'NOT SUPPORTED'}"
                 f"{'' if analysis['valid'] else ' (quick run, not valid)'}")
    return "\n".join(lines)
