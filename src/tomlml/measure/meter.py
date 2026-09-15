"""Energy measurement windows.

A window is bounded by two CUDA synchronizations. At each boundary the NVML
cumulative energy counter is read; between them a background thread samples
power, temperature, clocks, throttle reasons, and the counter itself at a
fixed interval. The window summary carries both the counter delta (primary)
and the integrated power (same-sensor consistency check), plus the operating
context needed to interpret them (DECISIONS.md D-003).
"""

from __future__ import annotations

import csv
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .nvml import NvmlDevice, decode_throttle_reasons, nvml_error_name


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #

@dataclass
class StateSample:
    t_wall: float
    t_perf: float
    power_w: Optional[float]
    temp_c: Optional[int]
    sm_clock_mhz: Optional[int]
    mem_clock_mhz: Optional[int]
    energy_mj: Optional[int]
    throttle_mask: Optional[int]


SAMPLE_FIELDS = ["t_wall", "t_perf", "power_w", "temp_c", "sm_clock_mhz",
                 "mem_clock_mhz", "energy_mj", "throttle_mask"]


def write_samples_csv(samples: List[StateSample], path: Path, window_label: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window"] + SAMPLE_FIELDS)
        for s in samples:
            w.writerow([window_label] + [getattr(s, k) for k in SAMPLE_FIELDS])
    return path


class StateSampler:
    """Background thread sampling device state at a fixed interval.

    Fields that report NotSupported are disabled after the first failure and
    recorded as None. Other errors are counted per field and the sample keeps
    None for that field.
    """

    def __init__(self, dev: NvmlDevice, interval_s: float = 0.05,
                 read_energy: bool = True, read_throttle: bool = True,
                 read_clocks: bool = True, read_temp: bool = True):
        self.dev = dev
        self.interval_s = float(interval_s)
        self.samples: List[StateSample] = []
        self.errors: Dict[str, int] = {}
        self.disabled: Dict[str, str] = {}
        self._enabled = {
            "power": True, "temp": read_temp, "sm_clock": read_clocks,
            "mem_clock": read_clocks, "energy": read_energy, "throttle": read_throttle,
        }
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _safe(self, key: str, fn: Callable[[], Any]) -> Any:
        if not self._enabled[key]:
            return None
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 - NVML errors are data here
            name = nvml_error_name(err)
            self.errors[key] = self.errors.get(key, 0) + 1
            if name in ("NotSupported", "NvmlUnavailable") or name.startswith("BindingError"):
                self._enabled[key] = False
                self.disabled[key] = name
            return None

    def read_one(self) -> StateSample:
        t_wall = time.time()
        t_perf = time.perf_counter()
        return StateSample(
            t_wall=t_wall,
            t_perf=t_perf,
            power_w=self._safe("power", self.dev.power_w),
            temp_c=self._safe("temp", self.dev.temperature_c),
            sm_clock_mhz=self._safe("sm_clock", self.dev.sm_clock_mhz),
            mem_clock_mhz=self._safe("mem_clock", self.dev.mem_clock_mhz),
            energy_mj=self._safe("energy", self.dev.energy_mj),
            throttle_mask=self._safe("throttle", self.dev.throttle_mask),
        )

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            self.samples.append(self.read_one())
            next_t += self.interval_s
            delay = next_t - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.perf_counter()  # fell behind; resynchronize

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tomlml-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> List[StateSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        return self.samples


# --------------------------------------------------------------------------- #
# Window summary
# --------------------------------------------------------------------------- #

@dataclass
class MeasurementWindow:
    label: str
    t_start_wall: float
    t_end_wall: float
    duration_s: float
    counter_start_mj: Optional[int]
    counter_end_mj: Optional[int]
    energy_counter_j: Optional[float]
    energy_power_integral_j: Optional[float]
    energy_mean_power_j: Optional[float]
    mean_power_w: Optional[float]
    power_std_w: Optional[float]
    power_min_w: Optional[float]
    power_max_w: Optional[float]
    n_samples: int
    sample_interval_s: float
    mean_temp_c: Optional[float]
    max_temp_c: Optional[int]
    mean_sm_clock_mhz: Optional[float]
    min_sm_clock_mhz: Optional[int]
    mean_mem_clock_mhz: Optional[float]
    throttle_union_mask: Optional[int]
    throttle_reasons: List[str]
    counter_monotonic: Optional[bool]
    processes_before: List[Dict[str, Any]]
    processes_after: List[Dict[str, Any]]
    contaminated: bool
    sampler_errors: Dict[str, int]
    sampler_disabled: Dict[str, str]
    n_calls: Optional[int] = None
    samples: List[StateSample] = field(default_factory=list, repr=False, compare=False)

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not include_samples:
            d.pop("samples", None)
        return d

    @property
    def energy_j(self) -> Optional[float]:
        """Primary energy: counter delta, falling back to the power integral."""
        if self.energy_counter_j is not None:
            return self.energy_counter_j
        return self.energy_power_integral_j

    def energy_per_call_j(self) -> Optional[float]:
        if not self.n_calls or self.energy_j is None:
            return None
        return self.energy_j / self.n_calls


def _mean(xs: List[float]) -> Optional[float]:
    return float(statistics.fmean(xs)) if xs else None


def _pstd(xs: List[float]) -> Optional[float]:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else (0.0 if xs else None)


def integrate_power(samples: List[StateSample], t0: float, t1: float) -> Optional[float]:
    """Trapezoidal integral of sampled power over [t0, t1] in perf-counter
    seconds. The first and last sample values are held to the window edges."""
    pts = [(s.t_perf, s.power_w) for s in samples if s.power_w is not None]
    if not pts:
        return None
    pts.sort()
    if pts[0][0] > t0:
        pts.insert(0, (t0, pts[0][1]))
    if pts[-1][0] < t1:
        pts.append((t1, pts[-1][1]))
    energy = 0.0
    for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
        if tb <= ta:
            continue
        energy += 0.5 * (pa + pb) * (tb - ta)
    return energy


def foreign_compute_processes(procs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [p for p in procs if p.get("kind") == "compute"]


def summarize_window(
    label: str,
    t0_wall: float, t1_wall: float,
    t0_perf: float, t1_perf: float,
    counter_start_mj: Optional[int], counter_end_mj: Optional[int],
    samples: List[StateSample],
    processes_before: List[Dict[str, Any]],
    processes_after: List[Dict[str, Any]],
    sample_interval_s: float,
    sampler_errors: Optional[Dict[str, int]] = None,
    sampler_disabled: Optional[Dict[str, str]] = None,
    n_calls: Optional[int] = None,
) -> MeasurementWindow:
    duration = max(t1_perf - t0_perf, 0.0)
    powers = [s.power_w for s in samples if s.power_w is not None]
    temps = [s.temp_c for s in samples if s.temp_c is not None]
    sms = [s.sm_clock_mhz for s in samples if s.sm_clock_mhz is not None]
    mems = [s.mem_clock_mhz for s in samples if s.mem_clock_mhz is not None]
    energies = [s.energy_mj for s in samples if s.energy_mj is not None]
    masks = [s.throttle_mask for s in samples if s.throttle_mask is not None]

    counter_j: Optional[float] = None
    if counter_start_mj is not None and counter_end_mj is not None:
        counter_j = (counter_end_mj - counter_start_mj) / 1000.0

    monotonic: Optional[bool] = None
    series: List[int] = []
    if counter_start_mj is not None:
        series.append(counter_start_mj)
    series.extend(energies)
    if counter_end_mj is not None:
        series.append(counter_end_mj)
    if len(series) >= 2:
        monotonic = all(b >= a for a, b in zip(series, series[1:]))

    union = 0
    for m in masks:
        union |= m
    mean_p = _mean(powers)

    contaminated = bool(foreign_compute_processes(processes_before) or
                        foreign_compute_processes(processes_after))

    return MeasurementWindow(
        label=label,
        t_start_wall=t0_wall,
        t_end_wall=t1_wall,
        duration_s=duration,
        counter_start_mj=counter_start_mj,
        counter_end_mj=counter_end_mj,
        energy_counter_j=counter_j,
        energy_power_integral_j=integrate_power(samples, t0_perf, t1_perf),
        energy_mean_power_j=(mean_p * duration) if mean_p is not None else None,
        mean_power_w=mean_p,
        power_std_w=_pstd(powers),
        power_min_w=min(powers) if powers else None,
        power_max_w=max(powers) if powers else None,
        n_samples=len(samples),
        sample_interval_s=sample_interval_s,
        mean_temp_c=_mean(temps),
        max_temp_c=max(temps) if temps else None,
        mean_sm_clock_mhz=_mean(sms),
        min_sm_clock_mhz=min(sms) if sms else None,
        mean_mem_clock_mhz=_mean(mems),
        throttle_union_mask=union if masks else None,
        throttle_reasons=decode_throttle_reasons(union) if masks else [],
        counter_monotonic=monotonic,
        processes_before=processes_before,
        processes_after=processes_after,
        contaminated=contaminated,
        sampler_errors=dict(sampler_errors or {}),
        sampler_disabled=dict(sampler_disabled or {}),
        n_calls=n_calls,
        samples=samples,
    )


# --------------------------------------------------------------------------- #
# Meter
# --------------------------------------------------------------------------- #

def make_cuda_sync(device_index: Optional[int] = None) -> Callable[[], None]:
    """Return a callable that synchronizes the given torch CUDA device (or the
    current one), and does nothing if torch or CUDA is unavailable."""
    def _sync() -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_index)
    return _sync


def no_sync() -> None:
    """Sync stand-in for windows with no CUDA work (idle baselines)."""
    return None


class EnergyMeter:
    """Measure the energy of work executed between ``start()`` and ``stop()``.

    Usage::

        meter = EnergyMeter(dev)
        window = meter.measure(lambda: model(x), label="inference", n_calls=100)

    or::

        with EnergyMeter(dev, label="train_epoch") as meter:
            train_one_epoch()
        window = meter.window
    """

    def __init__(self, dev: NvmlDevice, sample_interval_s: float = 0.05,
                 sync_fn: Optional[Callable[[], None]] = None,
                 check_processes: bool = True, read_energy: bool = True,
                 read_throttle: bool = True, label: str = ""):
        self.dev = dev
        self.sample_interval_s = float(sample_interval_s)
        self.sync_fn = sync_fn if sync_fn is not None else make_cuda_sync()
        self.check_processes = check_processes
        self.read_energy = read_energy
        self.read_throttle = read_throttle
        self.label = label
        self.window: Optional[MeasurementWindow] = None
        self._sampler: Optional[StateSampler] = None
        self._counter_error: Optional[str] = None
        self._t0_wall = self._t0_perf = 0.0
        self._c0: Optional[int] = None
        self._procs_before: List[Dict[str, Any]] = []

    def _read_counter(self) -> Optional[int]:
        if not self.read_energy:
            return None
        try:
            return self.dev.energy_mj()
        except Exception as err:  # noqa: BLE001
            self._counter_error = nvml_error_name(err)
            return None

    def start(self, label: Optional[str] = None) -> "EnergyMeter":
        if label is not None:
            self.label = label
        self.sync_fn()
        self._procs_before = self.dev.running_processes() if self.check_processes else []
        self._sampler = StateSampler(self.dev, self.sample_interval_s,
                                     read_energy=self.read_energy,
                                     read_throttle=self.read_throttle)
        self._c0 = self._read_counter()
        self._t0_wall = time.time()
        self._t0_perf = time.perf_counter()
        self._sampler.start()
        return self

    def stop(self, n_calls: Optional[int] = None) -> MeasurementWindow:
        if self._sampler is None:
            raise RuntimeError("EnergyMeter.stop() called before start()")
        self.sync_fn()
        t1_perf = time.perf_counter()
        t1_wall = time.time()
        c1 = self._read_counter()
        samples = self._sampler.stop()
        procs_after = self.dev.running_processes() if self.check_processes else []
        errors = dict(self._sampler.errors)
        disabled = dict(self._sampler.disabled)
        if self._counter_error:
            disabled["counter_boundary"] = self._counter_error
        self.window = summarize_window(
            self.label, self._t0_wall, t1_wall, self._t0_perf, t1_perf,
            self._c0, c1, samples, self._procs_before, procs_after,
            self.sample_interval_s, errors, disabled, n_calls=n_calls,
        )
        self._sampler = None
        return self.window

    def measure(self, fn: Callable[[], Any], label: Optional[str] = None,
                n_calls: int = 1) -> MeasurementWindow:
        self.start(label)
        for _ in range(int(n_calls)):
            fn()
        return self.stop(n_calls=n_calls)

    def __enter__(self) -> "EnergyMeter":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
# Protocol helpers ported from the TOMLSignals stack, counter added
# --------------------------------------------------------------------------- #

def wait_for_thermal_settle(
    dev: NvmlDevice,
    threshold_c: float = 1.0,
    window_s: float = 5.0,
    timeout_s: float = 120.0,
    poll_s: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> Dict[str, Any]:
    """Poll temperature until the max-min spread over the last ``window_s``
    seconds is at most ``threshold_c``. Returns the settled temperature, the
    wait time, and whether the timeout was hit (the caller decides whether a
    timed-out settle is acceptable and logs it)."""
    temps: List[float] = []
    times: List[float] = []
    t0 = clock()
    while True:
        now = clock()
        temps.append(float(dev.temperature_c()))
        times.append(now)
        cutoff = now - window_s
        while times and times[0] < cutoff:
            times.pop(0)
            temps.pop(0)
        elapsed = now - t0
        if elapsed >= window_s and len(temps) >= 3:
            spread = max(temps) - min(temps)
            if spread <= threshold_c:
                return {"settled_temp_c": temps[-1], "wait_time_s": elapsed,
                        "timed_out": False, "final_spread_c": spread}
        if elapsed >= timeout_s:
            spread = (max(temps) - min(temps)) if temps else None
            return {"settled_temp_c": temps[-1], "wait_time_s": elapsed,
                    "timed_out": True, "final_spread_c": spread}
        sleep_fn(poll_s)


def measure_idle(
    dev: NvmlDevice,
    duration_s: float = 3.0,
    discard_s: float = 1.0,
    sample_interval_s: float = 0.05,
    check_processes: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
    label: str = "idle",
) -> MeasurementWindow:
    """Idle baseline: sleep ``discard_s`` (device settling), then measure the
    remaining ``duration_s - discard_s`` with no work submitted."""
    if discard_s > 0:
        sleep_fn(discard_s)
    meter = EnergyMeter(dev, sample_interval_s=sample_interval_s, sync_fn=no_sync,
                        check_processes=check_processes, label=label)
    meter.start()
    sleep_fn(max(duration_s - discard_s, 0.0))
    return meter.stop()
