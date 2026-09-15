"""Shared measurement protocol for every experiment after the gate (D-010).

Settings come from the merged config (platform overlay included) and are
scaled by the quick factor. The helpers implement: thermal settle under the
workload itself, a same-configuration warmup, and one measured window with
a calibrated call count, returning the window plus its regime label.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..workloads.reference import calibrate_calls, run_for
from .meter import EnergyMeter, MeasurementWindow
from .nvml import decode_throttle_reasons


@dataclass
class ProtocolSettings:
    window_s: float = 20.0
    warmup_s: float = 2.0
    sample_interval_s: float = 0.05
    energy_source: str = "auto"
    power_max_plausible_w: Optional[float] = None
    thermal_settle_threshold_c: float = 1.0
    thermal_settle_window_s: float = 5.0
    thermal_settle_timeout_s: float = 300.0
    thermal_settle_poll_s: float = 1.0
    census_warmup_calls: int = 5
    census_calls: int = 10
    power_limit_enforced_w: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], quick_factor: float = 1.0,
                    power_max_plausible_w: Optional[float] = None,
                    power_limit_enforced_w: Optional[float] = None) -> "ProtocolSettings":
        m = cfg.get("measurement", {}) or {}
        s = cls(
            window_s=max(0.5, float(m.get("window_s", 20.0)) * quick_factor),
            warmup_s=max(0.05, float(m.get("warmup_s", 2.0)) * quick_factor),
            sample_interval_s=float(cfg.get("sample_interval_s", 0.05)),
            energy_source=str(cfg.get("energy_source", "auto")),
            power_max_plausible_w=power_max_plausible_w,
            thermal_settle_threshold_c=float(m.get("thermal_settle_threshold_c", 1.0)),
            thermal_settle_window_s=float(m.get("thermal_settle_window_s", 5.0)),
            thermal_settle_timeout_s=max(float(m.get("thermal_settle_window_s", 5.0)),
                                         float(m.get("thermal_settle_timeout_s", 300.0)) * quick_factor),
            thermal_settle_poll_s=float(m.get("thermal_settle_poll_s", 1.0)),
            census_warmup_calls=int(m.get("census_warmup_calls", 5)),
            census_calls=int(m.get("census_calls", 10)),
            power_limit_enforced_w=power_limit_enforced_w,
        )
        if quick_factor != 1.0:
            s.notes.append(f"quick factor {quick_factor}: window, warmup and settle timeout scaled")
        return s

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def plausibility_ceiling(capabilities: Dict[str, Any], multiple: float = 2.0) -> Optional[float]:
    """Power plausibility ceiling from a capability probe: ``multiple`` times
    the device's maximum power-limit constraint, or None if unknown."""
    constraints = (capabilities.get("power_limit_constraints_w") or {}).get("value")
    if isinstance(constraints, list) and len(constraints) == 2 and constraints[1]:
        return multiple * float(constraints[1])
    return None


class _Callable:
    """Adapter so ``run_for`` and ``calibrate_calls`` (which expect an object
    with ``run_once``) accept a bare callable."""

    def __init__(self, fn: Callable[[], None]):
        self.run_once = fn
        self.name = getattr(fn, "__name__", "callable")


def _as_workload(run_once: Any) -> Any:
    return run_once if hasattr(run_once, "run_once") else _Callable(run_once)


def settle_under_load(dev: Any, run_once: Callable[[], None], settings: ProtocolSettings,
                      sync_fn: Callable[[], None], clock: Callable[[], float] = time.perf_counter,
                      log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Run the workload until the GPU temperature is stable under it.

    Stability: max minus min temperature over the last ``thermal_settle_window_s``
    seconds is at most ``thermal_settle_threshold_c``. The workload runs in
    slices of about ``thermal_settle_poll_s`` between temperature reads, so
    the device never idles during the settle (D-010 item 2).
    """
    wl = _as_workload(run_once)
    temps: List[float] = []
    times: List[float] = []
    t0 = clock()
    calls = 0
    while True:
        calls += run_for(wl, settings.thermal_settle_poll_s, sync_fn)
        now = clock()
        temps.append(float(dev.temperature_c()))
        times.append(now)
        cutoff = now - settings.thermal_settle_window_s
        while times and times[0] < cutoff:
            times.pop(0)
            temps.pop(0)
        elapsed = now - t0
        spread = (max(temps) - min(temps)) if temps else None
        if elapsed >= settings.thermal_settle_window_s and len(temps) >= 3 \
                and spread is not None and spread <= settings.thermal_settle_threshold_c:
            result = {"settled_temp_c": temps[-1], "wait_time_s": elapsed, "timed_out": False,
                      "final_spread_c": spread, "calls": calls}
            break
        if elapsed >= settings.thermal_settle_timeout_s:
            result = {"settled_temp_c": temps[-1], "wait_time_s": elapsed, "timed_out": True,
                      "final_spread_c": spread, "calls": calls}
            break
    if log is not None:
        log(f"settled under load at {result['settled_temp_c']:.0f} C after {result['wait_time_s']:.1f} s "
            f"(timed_out={result['timed_out']}, {calls} calls)")
    return result


def capped_fraction(window: MeasurementWindow) -> Optional[float]:
    """Fraction of the window's samples whose throttle mask carries
    ``SwPowerCap``; None when no sample has a mask."""
    masks = [s.throttle_mask for s in window.samples if s.throttle_mask is not None]
    if not masks:
        return None
    n_capped = sum(1 for m in masks if "SwPowerCap" in decode_throttle_reasons(m))
    return n_capped / len(masks)


def regime_label(window: MeasurementWindow, power_limit_enforced_w: Optional[float],
                 capped_sample_fraction_min: float = 0.5) -> str:
    """``capped`` when the window ran at the power cap: at least
    ``capped_sample_fraction_min`` of its samples report ``SwPowerCap``, or
    mean power is within 3 percent of the enforced limit; else ``uncapped``.
    A single carried-over sample at the cap does not label the window."""
    frac = capped_fraction(window)
    if frac is not None and frac >= capped_sample_fraction_min:
        return "capped"
    if power_limit_enforced_w and window.mean_power_w is not None \
            and window.mean_power_w >= 0.97 * power_limit_enforced_w:
        return "capped"
    return "uncapped"


def measure_window(dev: Any, run_once: Callable[[], None], settings: ProtocolSettings,
                   sync_fn: Callable[[], None], label: str,
                   per_call_s: Optional[float] = None,
                   check_processes: bool = True) -> MeasurementWindow:
    """Same-configuration warmup, then one measured window of ``settings.window_s``
    seconds with the call count calibrated to fill it."""
    wl = _as_workload(run_once)
    run_for(wl, settings.warmup_s, sync_fn)
    if per_call_s is None:
        per_call_s = calibrate_calls(wl, settings.window_s, sync_fn)["per_call_s"]
    n_calls = max(1, int(math.ceil(settings.window_s / per_call_s)))
    meter = EnergyMeter(dev, sample_interval_s=settings.sample_interval_s, sync_fn=sync_fn,
                        check_processes=check_processes, energy_source=settings.energy_source,
                        power_max_plausible_w=settings.power_max_plausible_w)
    return meter.measure(wl.run_once, label=label, n_calls=n_calls)
