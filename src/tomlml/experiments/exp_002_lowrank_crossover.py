"""EXP-002: exact low-rank crossover, dense versus factorized GEMM (inference).

Measurement half of the pre-registered protocol (LOGBOOK.md EXP-002,
STUDY_DESIGN.md H2a). Per shape d: build inputs and K cycled weight sets,
verify exactness for every (f, B), census the commands per configuration
with the profiler, calibrate the call count per configuration, settle the
GPU thermally under a mid-grid load, then run the randomized sequence of
measured windows. The analysis (fits, held-out predictions, crossovers,
criteria) is a separate script so it can run after the measurement finishes
and again if the analysis code changes, always against the frozen windows.

``run`` accepts injectable factories so the whole pipeline runs in the test
suite without a GPU.
"""

from __future__ import annotations

import csv
import gzip
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..measure import (
    ProtocolSettings, command_census, measure_window, plausibility_ceiling, probe_capabilities,
    regime_label, settle_under_load,
)
from ..measure.meter import SAMPLE_FIELDS, MeasurementWindow
from ..to_model.gemm import (
    dense_descriptor, factorized_descriptor, factorized_weights_fit_l2, weight_sets_for_l2,
)
from ..utils.manifest import write_json
from ..workloads.lowrank import LowRankShape, l2_cache_bytes, torch_device_string
from ..workloads.reference import calibrate_calls
from . import RunContext


def config_key(d: int, f: float, B: int, realization: str) -> str:
    return f"d{d}_f{f:g}_B{B}_{realization}"


def build_plan(grid: Dict[str, Any], rng: Any) -> Dict[int, List[Dict[str, Any]]]:
    """Per shape, the randomized list of (f, B, realization, rep) cells."""
    plan: Dict[int, List[Dict[str, Any]]] = {}
    for d in grid["shapes"]:
        cells = [{"f": float(f), "B": int(B), "realization": str(rz), "rep": int(rep)}
                 for f in grid["rank_fractions"] for B in grid["batches"]
                 for rz in grid["realizations"] for rep in range(int(grid["reps"]))]
        order = rng.permutation(len(cells)).tolist() if rng is not None else list(range(len(cells)))
        plan[int(d)] = [cells[i] for i in order]
    return plan


def _apply_tf32_setting(allow_tf32: bool) -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": False}
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if not allow_tf32:
        torch.set_float32_matmul_precision("highest")
    return {"torch": True, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def _write_windows_gz(path: Path, windows: List[MeasurementWindow]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window"] + SAMPLE_FIELDS)
        for win in windows:
            for s in win.samples:
                w.writerow([win.label] + [getattr(s, k) for k in SAMPLE_FIELDS])
    return path


def _default_census(run_once: Callable[[], None], sync_fn: Callable[[], None],
                    settings: ProtocolSettings) -> Dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"method": "unavailable", "commands_per_call": None, "reason": "no CUDA"}
    except ImportError:
        return {"method": "unavailable", "commands_per_call": None, "reason": "torch not installed"}
    return command_census(run_once, sync_fn, warmup_calls=settings.census_warmup_calls,
                          n_calls=settings.census_calls)


def run(ctx: RunContext,
        shape_factory: Optional[Callable[..., Any]] = None,
        census_fn: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    cfg = ctx.config
    log = ctx.logger
    dev = ctx.dev
    sync = ctx.sync_fn
    grid = cfg["quick_grid"] if ctx.quick else cfg["grid"]
    wcfg = cfg["workload"]
    tol = float(wcfg["exactness_tolerance_rel_frobenius"])
    t_run0 = time.perf_counter()

    tf32 = _apply_tf32_setting(bool(wcfg.get("allow_tf32", False)))
    cap = probe_capabilities(dev)
    write_json(ctx.results_dir / "capabilities.json", cap)
    pmax = plausibility_ceiling(cap)
    enforced = cap["power_limit_enforced_w"]["value"]
    settings = ProtocolSettings.from_config(cfg, ctx.quick_factor, pmax, enforced)
    l2 = l2_cache_bytes(ctx.torch_device)
    factory = shape_factory or (lambda d, fracs, batches, K: LowRankShape(
        d, fracs, batches, K, device=torch_device_string(ctx.torch_device),
        data_seed=int(wcfg.get("data_seed", 1234)), dtype=str(wcfg.get("dtype", "float32"))))
    census = census_fn or (lambda run_once: _default_census(run_once, sync, settings))
    plan = build_plan(grid, ctx.rng)

    log.info("EXP-002 grid: shapes %s, fractions %s, batches %s, reps %s; window %.1f s, warmup %.1f s, "
             "energy source %s, plausibility ceiling %s W, L2 %s (%s)",
             grid["shapes"], grid["rank_fractions"], grid["batches"], grid["reps"],
             settings.window_s, settings.warmup_s, settings.energy_source, pmax, l2["bytes"], l2["source"])

    windows: List[Dict[str, Any]] = []
    census_out: Dict[str, Any] = {}
    exactness: Dict[str, Any] = {}
    descriptors: Dict[str, Any] = {}
    calibration: Dict[str, Any] = {}
    excluded: List[Dict[str, Any]] = []
    shapes_meta: Dict[str, Any] = {}
    settles: Dict[str, Any] = {}
    total_cells = sum(len(v) for v in plan.values())
    done_cells = 0

    for d in [int(x) for x in grid["shapes"]]:
        t_shape0 = time.perf_counter()
        K = weight_sets_for_l2(d, d, l2["bytes"], float(wcfg.get("weight_cycling_l2_multiple", 2.0)))
        shape = factory(d, grid["rank_fractions"], grid["batches"], K)
        shape.setup()
        shapes_meta[str(d)] = dict(shape.describe(), n_sets=K)
        log.info("shape d=%d: %d weight sets (%.1f MB dense footprint), ranks %s",
                 d, K, K * d * d * 4 / 2**20, shapes_meta[str(d)].get("ranks"))

        # exactness (D-010 item 6)
        for f in [float(x) for x in grid["rank_fractions"]]:
            for B in [int(x) for x in grid["batches"]]:
                ex = shape.exactness(f, B)
                key = f"d{d}_f{f:g}_B{B}"
                exactness[key] = ex
                if ex["max_rel_frobenius"] is None or ex["max_rel_frobenius"] > tol:
                    excluded.append({"key": key, "reason": "exactness", "max_rel_frobenius": ex["max_rel_frobenius"],
                                     "tolerance": tol})
                    log.warning("  %s excluded: relative Frobenius %.3g > %.1e", key, ex["max_rel_frobenius"] or -1, tol)
        excluded_keys = {e["key"] for e in excluded}

        # census, calibration, descriptors per configuration
        for f in [float(x) for x in grid["rank_fractions"]]:
            r = shape.ranks[f]
            for B in [int(x) for x in grid["batches"]]:
                if f"d{d}_f{f:g}_B{B}" in excluded_keys:
                    continue
                fit_l2 = factorized_weights_fit_l2(d, d, r, K, l2["bytes"])
                for rz in [str(x) for x in grid["realizations"]]:
                    key = config_key(d, f, B, rz)
                    runner = shape.runner(f, B, rz)
                    census_out[key] = census(runner)
                    calibration[key] = calibrate_calls(_Wrap(runner), settings.window_s, sync)
                    if rz == "dense":
                        m1 = dense_descriptor(B, d, d, weights_in_l2=False)
                        m2 = dense_descriptor(B, d, d, weights_in_l2=False)
                    else:
                        m1 = factorized_descriptor(B, d, d, r, weights_in_l2=False)
                        m2 = factorized_descriptor(B, d, d, r, weights_in_l2=fit_l2)
                    descriptors[key] = {"m1": m1, "m2": m2, "factorized_weights_fit_l2": fit_l2,
                                        "commands_per_call": census_out[key].get("commands_per_call"),
                                        "per_call_s_calibration": calibration[key]["per_call_s"]}
        log.info("  census and calibration done for d=%d (%d configurations)", d,
                 sum(1 for k in descriptors if k.startswith(f"d{d}_")))

        # thermal settle under the mid-grid load (D-010 item 2)
        sl = cfg["measurement"]["settle_load"]
        f_mid = float(sl["rank_fraction"]) if float(sl["rank_fraction"]) in shape.ranks else float(grid["rank_fractions"][len(grid["rank_fractions"]) // 2])
        B_mid = int(sl["batch"]) if int(sl["batch"]) in shape.batches else int(grid["batches"][len(grid["batches"]) // 2])
        settles[str(d)] = settle_under_load(dev, shape.runner(f_mid, B_mid, str(sl["realization"])), settings, sync,
                                            log=lambda m: log.info("  %s", m))

        # measured windows in randomized order
        shape_windows: List[MeasurementWindow] = []
        for cell in plan[d]:
            f, B, rz, rep = cell["f"], cell["B"], cell["realization"], cell["rep"]
            key = config_key(d, f, B, rz)
            if f"d{d}_f{f:g}_B{B}" in excluded_keys:
                continue
            w = measure_window(dev, shape.runner(f, B, rz), settings, sync,
                               label=f"{key}_rep{rep}", per_call_s=calibration[key]["per_call_s"])
            shape_windows.append(w)
            rec = w.to_dict()
            rec.update({"d": d, "f": f, "r": shape.ranks[f], "B": B, "realization": rz, "rep": rep,
                        "key": key, "per_call_s": w.duration_s / w.n_calls if w.n_calls else None,
                        "energy_per_call_j": w.energy_per_call_j(),
                        "energy_per_call_counter_j": w.energy_per_call_j("counter"),
                        "energy_per_call_integral_j": w.energy_per_call_j("power_integral"),
                        "regime": regime_label(w, settings.power_limit_enforced_w)})
            windows.append(rec)
            done_cells += 1
            if done_cells % 25 == 0 or ctx.quick:
                elapsed = time.perf_counter() - t_run0
                log.info("  [%d/%d] %s rep%d: %.4g J/call, %.1f W, %s, %.1f s; elapsed %.0f min, projected total %.0f min",
                         done_cells, total_cells, key, rep, rec["energy_per_call_j"] or 0.0,
                         w.mean_power_w or 0.0, rec["regime"], w.duration_s,
                         elapsed / 60.0, elapsed / 60.0 * total_cells / max(done_cells, 1))
        _write_windows_gz(ctx.samples_dir / f"d{d}.csv.gz", shape_windows)
        shape.teardown()
        log.info("shape d=%d finished in %.0f min", d, (time.perf_counter() - t_shape0) / 60.0)

        # persist incrementally so a crash keeps what was measured
        write_json(ctx.results_dir / "windows.json", windows)

    summary = {
        "experiment_id": ctx.exp_id, "run_id": ctx.run_id, "platform_tag": ctx.platform_tag,
        "quick": ctx.quick, "quick_factor": ctx.quick_factor,
        "device": dev.resolved.to_dict(),
        "grid": grid,
        "settings": settings.to_dict(),
        "tf32": tf32,
        "l2_cache": l2,
        "shapes": shapes_meta,
        "settles": settles,
        "plan_order": {str(d): [[c["f"], c["B"], c["realization"], c["rep"]] for c in cells]
                       for d, cells in plan.items()},
        "n_windows": len(windows),
        "n_configurations": len(descriptors),
        "excluded": excluded,
        "contaminated_windows": sum(1 for w in windows if w["contaminated"]),
        "samples_total": sum(w["n_samples"] for w in windows),
        "power_implausible_samples_total": sum(w["power_implausible_samples"] for w in windows),
        "counter_monotonic_all_windows": all(w["counter_monotonic"] for w in windows if w["counter_monotonic"] is not None)
        if windows else None,
        "wall_time_s": time.perf_counter() - t_run0,
        "analysis_pending": True,
        "gate_valid": not ctx.quick,
    }
    write_json(ctx.results_dir / "windows.json", windows)
    write_json(ctx.results_dir / "census.json", census_out)
    write_json(ctx.results_dir / "exactness.json", exactness)
    write_json(ctx.results_dir / "descriptors.json", descriptors)
    write_json(ctx.results_dir / "calibration.json", calibration)
    write_json(ctx.results_dir / "summary.json", summary)
    log.info("EXP-002 measurement complete: %d windows, %d excluded configurations, %d contaminated windows, %.0f min. "
             "Run scripts/analyze_exp_002.py on this run directory.",
             len(windows), len(excluded), summary["contaminated_windows"], summary["wall_time_s"] / 60.0)
    return summary


class _Wrap:
    def __init__(self, fn: Callable[[], None]):
        self.run_once = fn
        self.name = "runner"
