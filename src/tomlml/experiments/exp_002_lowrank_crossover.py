"""EXP-002: exact low-rank crossover, dense versus factorized GEMM (inference).

Measurement half of the pre-registered protocol (LOGBOOK.md EXP-002,
STUDY_DESIGN.md H2a). Per shape d: build inputs and K cycled weight sets,
verify exactness for every (f, B), census the commands per configuration
with the profiler, calibrate the call count per configuration, settle the
GPU thermally under a mid-grid load, then run the randomized sequence of
measured windows. The analysis (fits, held-out predictions, crossovers,
criteria) is a separate script so it can run after the measurement finishes
and again if the analysis code changes, always against the frozen windows.

Crash safety and resume: every measured window is appended to
``results/windows.jsonl`` (fsync) and to the shape's gzip sample log before
the next window starts; every shape's pre-measurement artifacts (exactness,
census, calibration, descriptors, settle) are written to
``results/shape_d<d>.json`` before its windows start. A resumed run
(``scripts/run_experiment.py --resume <run-id>``) reloads both, skips every
window already in the ledger, and keeps the same seeded order.

``run`` accepts injectable factories so the whole pipeline runs in the test
suite without a GPU.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..measure import (
    ProtocolSettings, capped_fraction, command_census, measure_window, plausibility_ceiling,
    probe_capabilities, regime_label, settle_under_load,
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


def window_label(d: int, cell: Dict[str, Any]) -> str:
    return f"{config_key(d, cell['f'], cell['B'], cell['realization'])}_rep{cell['rep']}"


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


# --------------------------------------------------------------------------- #
# Crash-safe persistence
# --------------------------------------------------------------------------- #

def load_ledger(path: Path) -> List[Dict[str, Any]]:
    """Window records from the JSON-lines ledger. A partial last line (crash
    mid-write) is dropped."""
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def append_ledger(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_window_gz(path: Path, window: MeasurementWindow) -> None:
    """Append one window's samples as a new gzip member (readers see one
    stream); the header is written only when the file is created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with gzip.open(path, "at", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["window"] + SAMPLE_FIELDS)
        for s in window.samples:
            w.writerow([window.label] + [getattr(s, k) for k in SAMPLE_FIELDS])


def load_shape_records(results_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for p in sorted(results_dir.glob("shape_d*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            out[int(rec["d"])] = rec
        except (ValueError, KeyError):
            continue
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

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


class _Wrap:
    def __init__(self, fn: Callable[[], None]):
        self.run_once = fn
        self.name = "runner"


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #

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
    ledger_path = ctx.results_dir / "windows.jsonl"

    # ---- resume bookkeeping ----------------------------------------------- #
    if ledger_path.exists() and not ctx.resume:
        raise RuntimeError(f"{ledger_path} exists but this is not a resume; refusing to overwrite a run")
    windows: List[Dict[str, Any]] = load_ledger(ledger_path) if ctx.resume else []
    completed = {r["label"] for r in windows}
    shape_records: Dict[int, Dict[str, Any]] = load_shape_records(ctx.results_dir) if ctx.resume else {}
    if ctx.resume:
        log.info("resuming: %d windows already in the ledger, %d shapes with artifacts",
                 len(windows), len(shape_records))

    tf32 = _apply_tf32_setting(bool(wcfg.get("allow_tf32", False)))
    cap = probe_capabilities(dev)
    write_json(ctx.results_dir / ("capabilities.resume.json" if ctx.resume else "capabilities.json"), cap)
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

    total_cells = sum(len(v) for v in plan.values())
    done_cells = len(windows)
    skipped = 0
    measured_now = 0

    for d in [int(x) for x in grid["shapes"]]:
        cells = plan[d]
        remaining = [c for c in cells if window_label(d, c) not in completed]
        if not remaining:
            log.info("shape d=%d: all %d windows already measured, skipping", d, len(cells))
            skipped += len(cells)
            continue
        t_shape0 = time.perf_counter()
        K = weight_sets_for_l2(d, d, l2["bytes"], float(wcfg.get("weight_cycling_l2_multiple", 2.0)))
        shape = factory(d, grid["rank_fractions"], grid["batches"], K)
        shape.setup()
        rec: Dict[str, Any] = {"d": d, "n_sets": K, "describe": shape.describe(), "exactness": {},
                               "excluded": [], "census": {}, "calibration": {}, "descriptors": {},
                               "settle": None, "resumed": ctx.resume}
        log.info("shape d=%d: %d weight sets (%.1f MB dense footprint), ranks %s%s",
                 d, K, K * d * d * 4 / 2**20, rec["describe"].get("ranks"),
                 f"; {len(cells) - len(remaining)} of {len(cells)} windows already measured" if ctx.resume else "")

        # exactness (D-010 item 6)
        for f in [float(x) for x in grid["rank_fractions"]]:
            for B in [int(x) for x in grid["batches"]]:
                ex = shape.exactness(f, B)
                key = f"d{d}_f{f:g}_B{B}"
                rec["exactness"][key] = ex
                if ex["max_rel_frobenius"] is None or ex["max_rel_frobenius"] > tol:
                    rec["excluded"].append({"key": key, "reason": "exactness",
                                            "max_rel_frobenius": ex["max_rel_frobenius"], "tolerance": tol})
                    log.warning("  %s excluded: relative Frobenius %.3g > %.1e", key,
                                ex["max_rel_frobenius"] if ex["max_rel_frobenius"] is not None else -1, tol)
        excluded_keys = {e["key"] for e in rec["excluded"]}

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
                    rec["census"][key] = census(runner)
                    rec["calibration"][key] = calibrate_calls(_Wrap(runner), settings.window_s, sync)
                    if rz == "dense":
                        m1 = dense_descriptor(B, d, d, weights_in_l2=False)
                        m2 = dense_descriptor(B, d, d, weights_in_l2=False)
                    else:
                        m1 = factorized_descriptor(B, d, d, r, weights_in_l2=False)
                        m2 = factorized_descriptor(B, d, d, r, weights_in_l2=fit_l2)
                    rec["descriptors"][key] = {"m1": m1, "m2": m2, "factorized_weights_fit_l2": fit_l2,
                                               "commands_per_call": rec["census"][key].get("commands_per_call"),
                                               "per_call_s_calibration": rec["calibration"][key]["per_call_s"]}
        log.info("  census and calibration done for d=%d (%d configurations)", d, len(rec["descriptors"]))

        # thermal settle under the mid-grid load (D-010 item 2)
        sl = cfg["measurement"]["settle_load"]
        f_mid = float(sl["rank_fraction"]) if float(sl["rank_fraction"]) in shape.ranks \
            else float(grid["rank_fractions"][len(grid["rank_fractions"]) // 2])
        B_mid = int(sl["batch"]) if int(sl["batch"]) in shape.batches \
            else int(grid["batches"][len(grid["batches"]) // 2])
        rec["settle"] = settle_under_load(dev, shape.runner(f_mid, B_mid, str(sl["realization"])), settings, sync,
                                          log=lambda m: log.info("  %s", m))
        shape_records[d] = rec
        write_json(ctx.results_dir / f"shape_d{d}.json", rec)

        # measured windows in randomized order; each appended to the ledger before the next
        for cell in cells:
            label = window_label(d, cell)
            if label in completed:
                skipped += 1
                continue
            f, B, rz, rep = cell["f"], cell["B"], cell["realization"], cell["rep"]
            key = config_key(d, f, B, rz)
            if f"d{d}_f{f:g}_B{B}" in excluded_keys:
                continue
            w = measure_window(dev, shape.runner(f, B, rz), settings, sync, label=label,
                               per_call_s=rec["calibration"][key]["per_call_s"])
            wrec = w.to_dict()
            wrec.update({"d": d, "f": f, "r": shape.ranks[f], "B": B, "realization": rz, "rep": rep,
                         "key": key, "per_call_s": w.duration_s / w.n_calls if w.n_calls else None,
                         "energy_per_call_j": w.energy_per_call_j(),
                         "energy_per_call_counter_j": w.energy_per_call_j("counter"),
                         "energy_per_call_integral_j": w.energy_per_call_j("power_integral"),
                         "regime": regime_label(w, settings.power_limit_enforced_w),
                         "swpowercap_fraction": capped_fraction(w),
                         "measured_in_resume": ctx.resume})
            append_ledger(ledger_path, wrec)
            append_window_gz(ctx.samples_dir / f"d{d}.csv.gz", w)
            windows.append(wrec)
            completed.add(label)
            done_cells += 1
            measured_now += 1
            if measured_now % 25 == 0 or ctx.quick:
                elapsed = time.perf_counter() - t_run0
                remaining_cells = total_cells - done_cells
                log.info("  [%d/%d] %s: %.4g J/call, %.1f W, %s, %.1f s; elapsed %.0f min, "
                         "remaining about %.0f min",
                         done_cells, total_cells, label, wrec["energy_per_call_j"] or 0.0,
                         w.mean_power_w or 0.0, wrec["regime"], w.duration_s, elapsed / 60.0,
                         elapsed / 60.0 / max(measured_now, 1) * remaining_cells)
        shape.teardown()
        log.info("shape d=%d finished in %.0f min", d, (time.perf_counter() - t_shape0) / 60.0)

    # ---- consolidation ---------------------------------------------------- #
    census_out = {k: v for rec in shape_records.values() for k, v in rec["census"].items()}
    exactness = {k: v for rec in shape_records.values() for k, v in rec["exactness"].items()}
    descriptors = {k: v for rec in shape_records.values() for k, v in rec["descriptors"].items()}
    calibration = {k: v for rec in shape_records.values() for k, v in rec["calibration"].items()}
    excluded = [e for rec in shape_records.values() for e in rec["excluded"]]
    summary = {
        "experiment_id": ctx.exp_id, "run_id": ctx.run_id, "platform_tag": ctx.platform_tag,
        "quick": ctx.quick, "quick_factor": ctx.quick_factor,
        "device": dev.resolved.to_dict(),
        "grid": grid,
        "settings": settings.to_dict(),
        "tf32": tf32,
        "l2_cache": l2,
        "shapes": {str(d): dict(rec["describe"], n_sets=rec["n_sets"]) for d, rec in shape_records.items()},
        "settles": {str(d): rec["settle"] for d, rec in shape_records.items()},
        "plan_order": {str(d): [[c["f"], c["B"], c["realization"], c["rep"]] for c in cells]
                       for d, cells in plan.items()},
        "n_windows": len(windows),
        "n_windows_measured_this_session": measured_now,
        "n_windows_skipped_on_resume": skipped,
        "resumed": ctx.resume,
        "n_configurations": len(descriptors),
        "excluded": excluded,
        "contaminated_windows": sum(1 for w in windows if w["contaminated"]),
        "samples_total": sum(w["n_samples"] for w in windows),
        "power_implausible_samples_total": sum(w["power_implausible_samples"] for w in windows),
        "counter_monotonic_all_windows": all(w["counter_monotonic"] for w in windows
                                             if w["counter_monotonic"] is not None) if windows else None,
        "wall_time_s_this_session": time.perf_counter() - t_run0,
        "analysis_pending": True,
        "gate_valid": not ctx.quick,
    }
    write_json(ctx.results_dir / "windows.json", windows)
    write_json(ctx.results_dir / "census.json", census_out)
    write_json(ctx.results_dir / "exactness.json", exactness)
    write_json(ctx.results_dir / "descriptors.json", descriptors)
    write_json(ctx.results_dir / "calibration.json", calibration)
    write_json(ctx.results_dir / "summary.json", summary)
    log.info("EXP-002 measurement complete: %d windows (%d this session, %d skipped on resume), "
             "%d excluded configurations, %d contaminated windows, %.0f min this session. "
             "Run scripts/analyze_exp_002.py on this run directory.",
             len(windows), measured_now, skipped, len(excluded), summary["contaminated_windows"],
             summary["wall_time_s_this_session"] / 60.0)
    return summary
