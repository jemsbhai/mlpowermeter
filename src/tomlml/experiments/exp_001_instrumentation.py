"""EXP-001: instrumentation validation (gate G0).

The pre-registered protocol and pass criteria are in LOGBOOK.md (EXP-001).
This module implements the protocol as ordered sections and evaluates the
criteria. The analysis functions are pure so they can be tested without a
GPU; ``run`` accepts a workload factory so the whole pipeline can be
exercised with host-side stand-ins.

Sections, in execution order:

  S0 capabilities        NVML capability probe and initial isolation check
  S1 resolution          counter and power update period, idle and under load
  S2 idle_pre            thermal settle, then idle baseline with drift fit
  S3 thermal             heat load, time to settle, post-heat idle baseline
  S4 repeatability       per workload: warmup, then back-to-back blocks
  S5 window_sufficiency  per workload: randomized windows of several lengths
  S6 isolation           final process check
  S7 criteria            evaluate the pre-registered criteria
"""

from __future__ import annotations

import csv
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..measure import (
    EnergyMeter, MeasurementWindow, measure_idle, probe_capabilities,
    probe_power_limit_write_permission, wait_for_thermal_settle,
)
from ..measure.meter import SAMPLE_FIELDS
from ..utils.manifest import write_json
from ..workloads import BackgroundLoad, calibrate_calls, make_workload, run_for
from . import RunContext


# --------------------------------------------------------------------------- #
# Pure analysis helpers
# --------------------------------------------------------------------------- #

def update_period_stats(times: Sequence[float], values: Sequence[Any]) -> Dict[str, Any]:
    """Statistics of the intervals between value changes in a polled series.

    ``times`` are perf-counter seconds. A change is any sample whose value
    differs from the previous one; for a monotone counter this is exactly the
    sensor's update event.
    """
    n = len(values)
    out: Dict[str, Any] = {"n_polls": n, "n_changes": 0, "distinct_values": 0,
                           "poll_rate_hz": None, "span_s": 0.0,
                           "period_ms_median": None, "period_ms_min": None,
                           "period_ms_max": None, "period_ms_p90": None}
    if n < 2:
        return out
    span = float(times[-1] - times[0])
    out["span_s"] = span
    out["poll_rate_hz"] = (n - 1) / span if span > 0 else None
    valid = [v for v in values if v is not None]
    out["distinct_values"] = len(set(valid))
    change_times = [times[i] for i in range(1, n)
                    if values[i] is not None and values[i - 1] is not None and values[i] != values[i - 1]]
    out["n_changes"] = len(change_times)
    intervals = [b - a for a, b in zip(change_times, change_times[1:]) if b > a]
    if intervals:
        ms = sorted(x * 1000.0 for x in intervals)
        out["period_ms_median"] = statistics.median(ms)
        out["period_ms_min"] = ms[0]
        out["period_ms_max"] = ms[-1]
        out["period_ms_p90"] = ms[min(len(ms) - 1, int(math.floor(0.9 * (len(ms) - 1))))]
    return out


def linear_drift(times: Sequence[float], values: Sequence[float]) -> Dict[str, Any]:
    """Least-squares slope of ``values`` against ``times`` plus a first-half
    versus second-half comparison. Returns None fields when underdetermined."""
    pts = [(float(t), float(v)) for t, v in zip(times, values) if v is not None]
    out: Dict[str, Any] = {"n": len(pts), "slope_per_s": None, "slope_per_min": None,
                           "intercept": None, "r2": None, "first_half_mean": None,
                           "second_half_mean": None, "half_diff": None}
    if len(pts) < 3:
        return out
    t0 = pts[0][0]
    xs = [t - t0 for t, _ in pts]
    ys = [v for _, v in pts]
    xm = statistics.fmean(xs)
    ym = statistics.fmean(ys)
    sxx = sum((x - xm) ** 2 for x in xs)
    if sxx == 0:
        return out
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = ym - slope * xm
    ss_tot = sum((y - ym) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    half = len(ys) // 2
    first = statistics.fmean(ys[:half])
    second = statistics.fmean(ys[half:])
    out.update(slope_per_s=slope, slope_per_min=slope * 60.0, intercept=intercept,
               r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
               first_half_mean=first, second_half_mean=second, half_diff=second - first)
    return out


def cv(values: Sequence[float]) -> Optional[float]:
    """Coefficient of variation with sample standard deviation (ddof=1)."""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    mean = statistics.fmean(vals)
    if mean == 0:
        return None
    return statistics.stdev(vals) / abs(mean)


def window_stats(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "cv": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "cv": cv(vals),
        "min": min(vals),
        "max": max(vals),
    }


def min_sufficient_window(per_length: Dict[str, Dict[str, Any]], target_cv: float) -> Optional[float]:
    """Smallest window length such that it and every longer length have a
    CV at or below ``target_cv``. Keys are string-formatted lengths in
    seconds. Returns None if no such length exists."""
    lengths = sorted((float(k), v) for k, v in per_length.items())
    for i, (length, _) in enumerate(lengths):
        tail = lengths[i:]
        if all(st.get("cv") is not None and st["cv"] <= target_cv for _, st in tail):
            return length
    return None


def select_energy_source(summary: Dict[str, Any], crit: Dict[str, Any]) -> Dict[str, Any]:
    """Per-platform energy source (DECISIONS.md D-009).

    The counter qualifies only if it is supported and monotonic (C1), updates
    at least every ``counter_update_period_ms_max`` (C2), and agrees with the
    integrated power reading within ``counter_vs_integral_rel_max`` on every
    repeatability block (C3). Otherwise the platform uses the power integral.
    """
    counter = summary.get("counter", {})
    c1 = counter.get("supported") is True and counter.get("monotonic_all_windows") is True
    period = (summary.get("resolution", {}).get("load", {}).get("energy", {})
              .get("period_ms_median"))
    c2 = period is not None and period <= crit["counter_update_period_ms_max"]
    rels = [r.get("counter_vs_integral_max_abs_rel") for r in summary.get("repeatability", {}).values()]
    rels = [r for r in rels if r is not None]
    worst = max(rels) if rels else None
    c3 = worst is not None and worst <= crit["counter_vs_integral_rel_max"]
    reasons = [k for k, ok in (("C1", c1), ("C2", c2), ("C3", c3)) if not ok]
    return {
        "selected": "counter" if not reasons else "power_integral",
        "counter_rejected_reasons": reasons,
        "c1": c1, "c2": c2, "c3": c3,
        "counter_update_period_ms_median": period,
        "counter_vs_integral_worst_abs_rel": worst,
    }


def evaluate_criteria(summary: Dict[str, Any], crit: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Evaluate the pre-registered EXP-001 criteria (protocol v2, D-009).

    C1 to C3 qualify the counter and select the energy source; they are
    reported but not gating. C4 and C7 are evaluated on the selected source.
    A criterion's ``gating`` flag says whether it counts toward the gate.
    """
    out: Dict[str, Dict[str, Any]] = {}

    def add(key: str, passed: Any, value: Any, threshold: Any, note: Optional[str] = None,
            gating: bool = True) -> None:
        out[key] = {"pass": bool(passed), "value": value, "threshold": threshold,
                    "note": note, "gating": gating}

    sel = select_energy_source(summary, crit)
    source = sel["selected"]
    qual_note = "counter qualification; selects the energy source (D-009), not gating"
    counter = summary.get("counter", {})
    add("C1_counter_supported_and_monotonic", sel["c1"],
        {"supported": counter.get("supported"), "monotonic": counter.get("monotonic_all_windows")},
        {"supported": True, "monotonic": True}, note=qual_note, gating=False)
    add("C2_counter_update_period_ms", sel["c2"], sel["counter_update_period_ms_median"],
        crit["counter_update_period_ms_max"], note=qual_note, gating=False)
    add("C3_counter_vs_power_integral_rel", sel["c3"], sel["counter_vs_integral_worst_abs_rel"],
        crit["counter_vs_integral_rel_max"], note=qual_note + "; same-sensor consistency", gating=False)

    stats_key = "stats_counter" if source == "counter" else "stats_integral"
    for name, cvmax in crit["repeatability_cv_max"].items():
        value = summary.get("repeatability", {}).get(name, {}).get(stats_key, {}).get("cv")
        add(f"C4_repeatability_cv_{name}", value is not None and value <= cvmax, value, cvmax,
            note=f"evaluated on energy source {source}")

    settle = summary.get("thermal", {}).get("settle_post_heat", {})
    add("C5_thermal_settle_before_timeout", settle.get("timed_out") is False,
        settle.get("wait_time_s"), summary.get("thermal", {}).get("settle_timeout_s"))

    n_cont = summary.get("isolation", {}).get("contaminated_windows")
    add("C6_no_foreign_compute_processes", n_cont == 0, n_cont, 0)

    mw_key = "min_sufficient_window_counter_s" if source == "counter" else "min_sufficient_window_integral_s"
    for name in crit.get("min_window_workloads", ["matmul_large"]):
        mw = summary.get("window_sufficiency", {}).get(name, {}).get(mw_key)
        add(f"C7_min_sufficient_window_s_{name}", mw is not None and mw <= crit["min_window_s_max"],
            mw, crit["min_window_s_max"], note=f"evaluated on energy source {source}")

    forbidden = set(crit.get("forbidden_throttle_reasons", []))
    seen: set = set()
    for r in summary.get("repeatability", {}).values():
        seen.update(r.get("throttle_reasons_union", []))
    seen.update(summary.get("thermal", {}).get("heat", {}).get("throttle_reasons", []))
    hits = sorted(seen & forbidden)
    add("C8_no_forbidden_throttle_reasons", not hits, hits, sorted(forbidden),
        note=f"reasons observed under load: {sorted(seen)}")

    verified = summary.get("device", {}).get("uuid_verified")
    add("C9_device_uuid_verified", verified is True, verified, True,
        note="None means torch reported no uuid; verify the mapping manually and record it in the logbook")

    iso = summary.get("isolation", {})
    n_bad = iso.get("power_implausible_samples_total")
    n_all = iso.get("samples_total")
    frac = (n_bad / n_all) if (n_bad is not None and n_all) else None
    add("C10_implausible_power_sample_fraction",
        frac is not None and frac <= crit.get("implausible_power_fraction_max", 0.01),
        frac, crit.get("implausible_power_fraction_max", 0.01),
        note=f"{n_bad} of {n_all} samples above the plausibility ceiling")
    return out


def gate_passes(criteria: Dict[str, Dict[str, Any]]) -> bool:
    return all(c["pass"] for c in criteria.values() if c.get("gating", True))


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def _write_windows_csv(path: Path, windows: List[MeasurementWindow]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window"] + SAMPLE_FIELDS)
        for win in windows:
            for s in win.samples:
                w.writerow([win.label] + [getattr(s, k) for k in SAMPLE_FIELDS])
    return path


def _write_change_events_csv(path: Path, poll: Dict[str, List[Any]], max_rows: int = 50_000) -> Path:
    """Store only the polls at which the energy counter or the power reading
    changed (plus the first and last), which is what the period analysis
    needs; a tight polling loop can produce hundreds of thousands of rows.
    Rows are capped at ``max_rows`` to keep the repository small."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t, e, p = poll["t_perf"], poll["energy_mj"], poll["power_w"]
    written = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_perf", "energy_mj", "power_w", "energy_changed", "power_changed"])
        for i in range(len(t)):
            ec = i > 0 and e[i] != e[i - 1]
            pc = i > 0 and p[i] != p[i - 1]
            if i == 0 or i == len(t) - 1 or ec or pc:
                w.writerow([t[i], e[i], p[i], int(ec), int(pc)])
                written += 1
                if written >= max_rows:
                    break
    return path


def poll_sensor(dev: Any, duration_s: float, read_energy: bool = True,
                max_polls: int = 2_000_000) -> Dict[str, List[Any]]:
    """Poll the counter and the power reading as fast as NVML allows."""
    times: List[float] = []
    energies: List[Optional[int]] = []
    powers: List[float] = []
    t_end = time.perf_counter() + duration_s
    while time.perf_counter() < t_end and len(times) < max_polls:
        times.append(time.perf_counter())
        energies.append(dev.energy_mj() if read_energy else None)
        powers.append(dev.power_w())
    return {"t_perf": times, "energy_mj": energies, "power_w": powers}


def _window_brief(w: MeasurementWindow) -> Dict[str, Any]:
    return {
        "label": w.label, "duration_s": w.duration_s, "n_calls": w.n_calls,
        "energy_counter_j": w.energy_counter_j,
        "energy_power_integral_j": w.energy_power_integral_j,
        "energy_per_call_counter_j": w.energy_per_call_j("counter"),
        "energy_per_call_integral_j": w.energy_per_call_j("power_integral"),
        "mean_power_w": w.mean_power_w, "power_std_w": w.power_std_w,
        "mean_temp_c": w.mean_temp_c, "max_temp_c": w.max_temp_c,
        "mean_sm_clock_mhz": w.mean_sm_clock_mhz, "min_sm_clock_mhz": w.min_sm_clock_mhz,
        "throttle_reasons": w.throttle_reasons, "counter_monotonic": w.counter_monotonic,
        "contaminated": w.contaminated, "n_samples": w.n_samples,
        "power_implausible_samples": w.power_implausible_samples,
    }


def _rel(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #

def run(ctx: RunContext,
        workload_factory: Optional[Callable[[str, Dict[str, Any]], Any]] = None) -> Dict[str, Any]:
    cfg = ctx.config
    log = ctx.logger
    dev = ctx.dev
    sync = ctx.sync_fn
    interval = float(cfg.get("sample_interval_s", 0.05))
    factory = workload_factory or (
        lambda name, params: make_workload(name, params, device_index=ctx.torch_device))

    summary: Dict[str, Any] = {
        "experiment_id": ctx.exp_id, "run_id": ctx.run_id, "platform_tag": ctx.platform_tag,
        "quick": ctx.quick, "quick_factor": ctx.quick_factor,
        "device": dev.resolved.to_dict(),
    }
    all_windows: List[Dict[str, Any]] = []

    def record(section: str, workload: Optional[str], w: MeasurementWindow) -> Dict[str, Any]:
        d = w.to_dict()
        d["section"] = section
        d["workload"] = workload
        all_windows.append(d)
        return d

    t_run0 = time.perf_counter()

    # ---- S0 capabilities -------------------------------------------------- #
    log.info("S0 capabilities: probing NVML")
    cap = probe_capabilities(dev)
    if cfg.get("probe_control_permissions", False):
        cap["power_limit_write_probe"] = probe_power_limit_write_permission(dev)
    write_json(ctx.results_dir / "capabilities.json", cap)
    counter_supported = cap["energy_counter_mj"]["error"] is None
    constraints = cap["power_limit_constraints_w"]["value"]
    pmax = (2.0 * float(constraints[1])) if isinstance(constraints, list) and len(constraints) == 2 else None
    summary["capabilities_brief"] = {
        "driver_version": cap["system"]["driver_version"],
        "cuda_driver_version_str": cap["system"]["cuda_driver_version_str"],
        "energy_counter_supported": counter_supported,
        "energy_counter_error": cap["energy_counter_mj"]["error"],
        "power_limit_enforced_w": cap["power_limit_enforced_w"]["value"],
        "power_limit_enforced_error": cap["power_limit_enforced_w"]["error"],
        "power_limit_constraints_w": cap["power_limit_constraints_w"]["value"],
        "sm_clock_max_mhz": cap["sm_clock_max_mhz"]["value"],
        "persistence_mode": cap["persistence_mode"]["value"],
        "throttle_api": cap["throttle"]["api"],
        "native_power_sample_period_ms": cap.get("power_samples_api", {}).get("native_period_ms_median"),
        "power_max_plausible_w": pmax,
    }
    summary["isolation"] = {"start": cap["running_processes"]}
    log.info("  counter supported=%s driver=%s enforced_limit=%s W",
             counter_supported, cap["system"]["driver_version"],
             cap["power_limit_enforced_w"]["value"])

    # ---- S1 resolution ---------------------------------------------------- #
    res_cfg = cfg["resolution"]
    dur = ctx.scaled(res_cfg["duration_s"], 0.2)
    log.info("S1 resolution: polling sensor at idle for %.1f s", dur)
    poll_idle = poll_sensor(dev, dur, read_energy=counter_supported)
    _write_change_events_csv(ctx.samples_dir / "resolution_idle.csv", poll_idle)
    wl_params = cfg["workloads"]
    wl_large = factory("matmul_large", wl_params.get("matmul_large", {}))
    wl_large.setup()
    log.info("S1 resolution: polling sensor under load for %.1f s", dur)
    with BackgroundLoad(wl_large, sync, device_index=ctx.torch_device) as bg:
        time.sleep(min(1.0, dur / 2))
        poll_load = poll_sensor(dev, dur, read_energy=counter_supported)
    if bg.error:
        raise RuntimeError(f"background load failed: {bg.error}")
    _write_change_events_csv(ctx.samples_dir / "resolution_load.csv", poll_load)
    summary["resolution"] = {
        "duration_s": dur,
        "background_load_calls": bg.calls,
        "idle": {"energy": update_period_stats(poll_idle["t_perf"], poll_idle["energy_mj"]),
                 "power": update_period_stats(poll_idle["t_perf"], poll_idle["power_w"])},
        "load": {"energy": update_period_stats(poll_load["t_perf"], poll_load["energy_mj"]),
                 "power": update_period_stats(poll_load["t_perf"], poll_load["power_w"])},
    }
    log.info("  counter update period under load: median %s ms (idle %s ms); poll rate %.0f Hz",
             summary["resolution"]["load"]["energy"]["period_ms_median"],
             summary["resolution"]["idle"]["energy"]["period_ms_median"],
             summary["resolution"]["load"]["energy"]["poll_rate_hz"] or 0.0)

    # ---- S2 idle_pre ------------------------------------------------------ #
    th = cfg["thermal"]
    settle_kw = dict(threshold_c=th["threshold_c"], window_s=th["window_s"],
                     poll_s=th.get("poll_s", 1.0), timeout_s=ctx.scaled(th["timeout_s"], th["window_s"]))
    idle_cfg = cfg["idle"]
    idle_dur = ctx.scaled(idle_cfg["duration_s"], 0.5)
    idle_discard = ctx.scaled(idle_cfg["discard_s"], 0.05)
    log.info("S2 idle_pre: waiting for thermal settle (timeout %.0f s)", settle_kw["timeout_s"])
    settle_pre = wait_for_thermal_settle(dev, **settle_kw)
    log.info("  settled at %.0f C after %.1f s (timed_out=%s); idle baseline %.1f s",
             settle_pre["settled_temp_c"], settle_pre["wait_time_s"], settle_pre["timed_out"], idle_dur)
    w_idle_pre = measure_idle(dev, duration_s=idle_dur, discard_s=idle_discard,
                              sample_interval_s=interval, label="idle_pre",
                              power_max_plausible_w=pmax)
    record("idle_pre", None, w_idle_pre)
    _write_windows_csv(ctx.samples_dir / "idle_pre.csv", [w_idle_pre])
    drift_pre = linear_drift([s.t_perf for s in w_idle_pre.samples],
                             [s.power_w for s in w_idle_pre.samples])
    summary["idle"] = {"pre_heat": dict(_window_brief(w_idle_pre), drift=drift_pre, settle=settle_pre)}
    log.info("  idle power %.2f W (std %.2f), drift %.3f W/min",
             w_idle_pre.mean_power_w or 0.0, w_idle_pre.power_std_w or 0.0,
             drift_pre["slope_per_min"] or 0.0)

    # ---- S3 thermal ------------------------------------------------------- #
    heat_s = ctx.scaled(th["heat_duration_s"], 0.2)
    log.info("S3 thermal: heat load for %.1f s", heat_s)
    meter = EnergyMeter(dev, sample_interval_s=interval, sync_fn=sync, label="heat",
                        power_max_plausible_w=pmax)
    meter.start()
    heat_calls = run_for(wl_large, heat_s, sync)
    w_heat = meter.stop(n_calls=heat_calls)
    record("heat", "matmul_large", w_heat)
    _write_windows_csv(ctx.samples_dir / "heat.csv", [w_heat])
    log.info("  heat: %.1f W mean, max %s C, sm clock %.0f MHz, throttle %s; settling",
             w_heat.mean_power_w or 0.0, w_heat.max_temp_c, w_heat.mean_sm_clock_mhz or 0.0,
             w_heat.throttle_reasons)
    settle_post = wait_for_thermal_settle(dev, **settle_kw)
    w_idle_post = measure_idle(dev, duration_s=idle_dur, discard_s=idle_discard,
                               sample_interval_s=interval, label="idle_post",
                               power_max_plausible_w=pmax)
    record("idle_post", None, w_idle_post)
    _write_windows_csv(ctx.samples_dir / "idle_post.csv", [w_idle_post])
    drift_post = linear_drift([s.t_perf for s in w_idle_post.samples],
                              [s.power_w for s in w_idle_post.samples])
    summary["thermal"] = {
        "heat": _window_brief(w_heat),
        "settle_post_heat": settle_post,
        "settle_timeout_s": settle_kw["timeout_s"],
        "idle_post_heat": dict(_window_brief(w_idle_post), drift=drift_post),
        "hot_idle_bias_w": ((w_idle_post.mean_power_w or 0.0) - (w_idle_pre.mean_power_w or 0.0)),
    }
    summary["idle"]["post_heat"] = summary["thermal"]["idle_post_heat"]
    log.info("  settled after %.1f s (timed_out=%s); post-heat idle %.2f W (bias %+.2f W)",
             settle_post["wait_time_s"], settle_post["timed_out"],
             w_idle_post.mean_power_w or 0.0, summary["thermal"]["hot_idle_bias_w"])

    # ---- S4 repeatability and S5 window sufficiency ----------------------- #
    rep_cfg = cfg["repeatability"]
    ws_cfg = cfg["window_sufficiency"]
    summary["repeatability"] = {}
    summary["window_sufficiency"] = {}
    summary["workload_descriptions"] = {}
    workloads: Dict[str, Any] = {"matmul_large": wl_large}
    for name, params in wl_params.items():
        if name not in workloads:
            wl = factory(name, params or {})
            wl.setup()
            workloads[name] = wl
        wl = workloads[name]
        summary["workload_descriptions"][name] = wl.describe()

        block_s = ctx.scaled(rep_cfg["block_duration_s"], 0.1)
        n_blocks = ctx.scaled_int(rep_cfg["n_blocks"], 3)
        warm_s = ctx.scaled(rep_cfg["warmup_s"], 0.1)
        log.info("S4 repeatability [%s]: warmup %.1f s, then %d blocks of %.1f s",
                 name, warm_s, n_blocks, block_s)
        run_for(wl, warm_s, sync)
        calib = calibrate_calls(wl, block_s, sync)
        log.info("  calibrated %.3f ms per call -> %d calls per block",
                 calib["per_call_s"] * 1000.0, calib["calls"])
        blocks: List[MeasurementWindow] = []
        for b in range(n_blocks):
            w = EnergyMeter(dev, sample_interval_s=interval, sync_fn=sync,
                            power_max_plausible_w=pmax).measure(
                wl.run_once, label=f"{name}_block{b:02d}", n_calls=calib["calls"])
            record("repeatability", name, w)
            blocks.append(w)
            log.info("  block %02d: %.3f s, %.1f W, %.4g J/call (counter) %.4g J/call (integral), monotonic=%s, throttle=%s",
                     b, w.duration_s, w.mean_power_w or 0.0, w.energy_per_call_j("counter") or 0.0,
                     w.energy_per_call_j("power_integral") or 0.0, w.counter_monotonic, w.throttle_reasons)
        _write_windows_csv(ctx.samples_dir / f"repeatability_{name}.csv", blocks)
        epc_counter = [w.energy_per_call_j("counter") for w in blocks]
        epc_integral = [w.energy_per_call_j("power_integral") for w in blocks]
        rels = [_rel(w.energy_counter_j, w.energy_power_integral_j) for w in blocks]
        rels = [r for r in rels if r is not None]
        reasons: set = set()
        for w in blocks:
            reasons.update(w.throttle_reasons)
        summary["repeatability"][name] = {
            "calibration": calib,
            "n_blocks": n_blocks,
            "block_duration_s_target": block_s,
            "blocks": [_window_brief(w) for w in blocks],
            "energy_per_call_counter_j": epc_counter,
            "energy_per_call_integral_j": epc_integral,
            "stats_counter": window_stats(epc_counter),
            "stats_integral": window_stats(epc_integral),
            "mean_power_w": statistics.fmean([w.mean_power_w for w in blocks if w.mean_power_w is not None])
            if any(w.mean_power_w is not None for w in blocks) else None,
            "max_temp_c": max([w.max_temp_c for w in blocks if w.max_temp_c is not None], default=None),
            "mean_sm_clock_mhz": statistics.fmean([w.mean_sm_clock_mhz for w in blocks
                                                   if w.mean_sm_clock_mhz is not None])
            if any(w.mean_sm_clock_mhz is not None for w in blocks) else None,
            "counter_vs_integral_rel": rels,
            "counter_vs_integral_max_abs_rel": max(abs(r) for r in rels) if rels else None,
            "throttle_reasons_union": sorted(reasons),
            "counter_monotonic_all": all(w.counter_monotonic for w in blocks)
            if all(w.counter_monotonic is not None for w in blocks) else None,
            "contaminated_blocks": sum(1 for w in blocks if w.contaminated),
            "power_implausible_samples": sum(w.power_implausible_samples for w in blocks),
        }
        st_c = summary["repeatability"][name]["stats_counter"]
        st_i = summary["repeatability"][name]["stats_integral"]
        log.info("  repeatability [%s]: counter %s J/call CV %s; integral %s J/call CV %s; counter vs integral max |rel| %s",
                 name, f"{st_c['mean']:.4g}" if st_c["mean"] is not None else None,
                 f"{st_c['cv']:.4f}" if st_c["cv"] is not None else None,
                 f"{st_i['mean']:.4g}" if st_i["mean"] is not None else None,
                 f"{st_i['cv']:.4f}" if st_i["cv"] is not None else None,
                 summary["repeatability"][name]["counter_vs_integral_max_abs_rel"])

        lengths = [float(x) for x in ws_cfg["lengths_s"]]
        reps = ctx.scaled_int(ws_cfg["reps"], 2)
        ws_warm = ctx.scaled(ws_cfg["warmup_s"], 0.05)
        plan = [(L, r) for L in lengths for r in range(reps)]
        order = ctx.rng.permutation(len(plan)).tolist() if ctx.rng is not None else list(range(len(plan)))
        plan = [plan[i] for i in order]
        log.info("S5 window_sufficiency [%s]: lengths %s x %d reps, randomized order", name, lengths, reps)
        per_length_counter: Dict[str, List[Optional[float]]] = {f"{L:g}": [] for L in lengths}
        per_length_integral: Dict[str, List[Optional[float]]] = {f"{L:g}": [] for L in lengths}
        targets: Dict[str, float] = {f"{L:g}": ctx.scaled(L, 0.02) for L in lengths}
        ws_windows: List[MeasurementWindow] = []
        for L, r in plan:
            target = targets[f"{L:g}"]
            calls = max(1, int(math.ceil(target / calib["per_call_s"])))
            run_for(wl, ws_warm, sync)
            w = EnergyMeter(dev, sample_interval_s=interval, sync_fn=sync,
                            power_max_plausible_w=pmax).measure(
                wl.run_once, label=f"{name}_L{L:g}_r{r}", n_calls=calls)
            record("window_sufficiency", name, w)
            ws_windows.append(w)
            per_length_counter[f"{L:g}"].append(w.energy_per_call_j("counter"))
            per_length_integral[f"{L:g}"].append(w.energy_per_call_j("power_integral"))
        _write_windows_csv(ctx.samples_dir / f"window_sufficiency_{name}.csv", ws_windows)
        target_cv = cfg["criteria"]["min_window_cv_target"]
        stats_counter = {k: dict(window_stats(v), target_s=targets[k]) for k, v in per_length_counter.items()}
        stats_integral = {k: dict(window_stats(v), target_s=targets[k]) for k, v in per_length_integral.items()}
        summary["window_sufficiency"][name] = {
            "order": [[L, r] for L, r in plan],
            "target_s_by_length": targets,
            "per_length_counter": stats_counter,
            "per_length_integral": stats_integral,
            "target_cv": target_cv,
            "min_sufficient_window_counter_s": min_sufficient_window(stats_counter, target_cv),
            "min_sufficient_window_integral_s": min_sufficient_window(stats_integral, target_cv),
        }
        log.info("  window sufficiency [%s]: CV by length, counter %s / integral %s; smallest sufficient window counter %s s, integral %s s",
                 name,
                 {k: (round(v["cv"], 4) if v["cv"] is not None else None) for k, v in stats_counter.items()},
                 {k: (round(v["cv"], 4) if v["cv"] is not None else None) for k, v in stats_integral.items()},
                 summary["window_sufficiency"][name]["min_sufficient_window_counter_s"],
                 summary["window_sufficiency"][name]["min_sufficient_window_integral_s"])

    for wl in workloads.values():
        wl.teardown()

    # ---- S6 isolation ----------------------------------------------------- #
    summary["isolation"]["end"] = dev.running_processes()
    summary["isolation"]["contaminated_windows"] = sum(1 for d in all_windows if d["contaminated"])
    summary["isolation"]["windows_total"] = len(all_windows)
    summary["isolation"]["samples_total"] = sum(d["n_samples"] for d in all_windows)
    summary["isolation"]["power_implausible_samples_total"] = sum(
        d["power_implausible_samples"] for d in all_windows)
    monotonic_flags = [d["counter_monotonic"] for d in all_windows]
    summary["counter"] = {
        "supported": counter_supported,
        "monotonic_all_windows": (all(f is True for f in monotonic_flags)
                                  if monotonic_flags and all(f is not None for f in monotonic_flags)
                                  else None),
        "windows_with_counter": sum(1 for f in monotonic_flags if f is not None),
    }
    write_json(ctx.results_dir / "windows.json", all_windows)

    # ---- S7 criteria ------------------------------------------------------ #
    summary["energy_source"] = select_energy_source(summary, cfg["criteria"])
    summary["criteria"] = evaluate_criteria(summary, cfg["criteria"])
    summary["gate_pass"] = gate_passes(summary["criteria"])
    summary["gate_valid"] = not ctx.quick
    summary["wall_time_s"] = time.perf_counter() - t_run0
    write_json(ctx.results_dir / "summary.json", summary)
    log.info("energy source for this platform: %s%s", summary["energy_source"]["selected"],
             (" (counter rejected: " + ", ".join(summary["energy_source"]["counter_rejected_reasons"]) + ")")
             if summary["energy_source"]["counter_rejected_reasons"] else "")
    for key, c in summary["criteria"].items():
        log.info("  %s %s (value=%s threshold=%s)%s", "PASS" if c["pass"] else "FAIL", key,
                 c["value"], c["threshold"], "" if c.get("gating", True) else " [not gating]")
    log.info("GATE %s (%s) in %.0f s", "PASS" if summary["gate_pass"] else "FAIL",
             "valid" if summary["gate_valid"] else "QUICK RUN, NOT VALID FOR THE GATE",
             summary["wall_time_s"])
    return summary
