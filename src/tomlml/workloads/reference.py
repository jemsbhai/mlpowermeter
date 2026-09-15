"""Reference workloads for instrumentation validation.

These are not algorithms under study. They exist to exercise the sensor in
two regimes that bracket the study's workloads:

* ``matmul_large``: one large FP32 GEMM per call; compute bound, high power,
  few commands per joule.
* ``matmul_tiny_loop``: one tiny FP32 GEMM per call; dispatch bound, low
  power, the regime where reinforcement learning agents spend their time.

A workload exposes ``setup``, ``run_once`` (one call, asynchronous with
respect to the host), ``teardown``, and a ``describe`` dict for the manifest.
``calibrate_calls`` turns a target duration into a call count.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Dict, Optional, Protocol


class Workload(Protocol):
    name: str

    def setup(self) -> None: ...
    def run_once(self) -> None: ...
    def teardown(self) -> None: ...
    def describe(self) -> Dict[str, Any]: ...


class _TorchMatmul:
    """FP32 square GEMM ``c = a @ b`` with preallocated output."""

    def __init__(self, name: str, n: int, device_index: int = 0, dtype: str = "float32"):
        self.name = name
        self.n = int(n)
        self.device_index = int(device_index)
        self.dtype = dtype
        self._a = self._b = self._c = None

    def setup(self) -> None:
        import torch
        dt = getattr(torch, self.dtype)
        dev = torch.device("cuda", self.device_index)
        gen = torch.Generator(device="cpu").manual_seed(1234)
        self._a = torch.randn(self.n, self.n, generator=gen, dtype=torch.float32).to(dev, dt)
        self._b = torch.randn(self.n, self.n, generator=gen, dtype=torch.float32).to(dev, dt)
        self._c = torch.empty(self.n, self.n, device=dev, dtype=dt)
        torch.cuda.synchronize(dev)

    def run_once(self) -> None:
        import torch
        torch.mm(self._a, self._b, out=self._c)

    def teardown(self) -> None:
        import torch
        self._a = self._b = self._c = None
        torch.cuda.empty_cache()

    def describe(self) -> Dict[str, Any]:
        import torch
        return {
            "name": self.name,
            "kind": "torch.mm square",
            "n": self.n,
            "dtype": self.dtype,
            "flop_per_call": 2 * self.n ** 3,
            "n_mac_per_call": self.n ** 3,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }


class SleepWorkload:
    """Host-side stand-in used by the test suite (no GPU): each call sleeps."""

    def __init__(self, name: str = "sleep", per_call_s: float = 0.001):
        self.name = name
        self.per_call_s = float(per_call_s)

    def setup(self) -> None:
        return None

    def run_once(self) -> None:
        time.sleep(self.per_call_s)

    def teardown(self) -> None:
        return None

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": "sleep", "per_call_s": self.per_call_s}


REFERENCE_WORKLOADS = ("matmul_large", "matmul_tiny_loop")


def make_workload(name: str, params: Optional[Dict[str, Any]] = None,
                  device_index: int = 0) -> Workload:
    params = dict(params or {})
    if name == "matmul_large":
        return _TorchMatmul(name, n=params.get("n", 4096), device_index=device_index,
                            dtype=params.get("dtype", "float32"))
    if name == "matmul_tiny_loop":
        return _TorchMatmul(name, n=params.get("n", 64), device_index=device_index,
                            dtype=params.get("dtype", "float32"))
    raise KeyError(f"unknown reference workload {name!r}; known: {REFERENCE_WORKLOADS}")


def calibrate_calls(workload: Workload, target_s: float, sync_fn: Callable[[], None],
                    warmup_calls: int = 5, timing_calls: Optional[int] = None,
                    min_timing_s: float = 0.5) -> Dict[str, Any]:
    """Estimate the number of calls that fills ``target_s`` seconds.

    Times a batch of calls (growing the batch until it lasts at least
    ``min_timing_s``) so that per-call time is measured with the same
    dispatch pattern the measurement will use.
    """
    for _ in range(warmup_calls):
        workload.run_once()
    sync_fn()
    n = int(timing_calls) if timing_calls else 20
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            workload.run_once()
        sync_fn()
        elapsed = time.perf_counter() - t0
        if elapsed >= min_timing_s or n >= 10_000_000:
            break
        n = int(n * max(2.0, min(50.0, min_timing_s / max(elapsed, 1e-6) * 1.5)))
    per_call = elapsed / n
    calls = max(1, int(math.ceil(target_s / per_call)))
    return {"per_call_s": per_call, "timing_calls": n, "timing_elapsed_s": elapsed,
            "target_s": target_s, "calls": calls}


def run_for(workload: Workload, duration_s: float, sync_fn: Callable[[], None]) -> int:
    """Run the workload until at least ``duration_s`` has elapsed (unmeasured
    warmup). Returns the number of calls made."""
    calls = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        workload.run_once()
        calls += 1
    sync_fn()
    return calls


class BackgroundLoad:
    """Run a workload continuously in a thread while the caller does
    something else (used to poll the sensor under load)."""

    def __init__(self, workload: Workload, sync_fn: Callable[[], None],
                 device_index: Optional[int] = None):
        self.workload = workload
        self.sync_fn = sync_fn
        self.device_index = device_index
        self.calls = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def _loop(self) -> None:
        try:
            if self.device_index is not None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.set_device(self.device_index)
                except ImportError:
                    pass  # host-side workloads need no device pin
            while not self._stop.is_set():
                self.workload.run_once()
                self.calls += 1
            self.sync_fn()
        except Exception as err:  # noqa: BLE001 - surfaced to the caller
            self.error = f"{type(err).__name__}: {err}"

    def __enter__(self) -> "BackgroundLoad":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tomlml-load", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=120.0)
