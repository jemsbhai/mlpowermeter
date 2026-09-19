#!/usr/bin/env python
"""Diagnose the torch.profiler device-event census on this machine.

    python scripts/diag_profiler.py

Profiles ten small GEMMs and reports what the profiler saw: total events,
device events by device_type, kernels attached to CPU op events, and the
CUPTI-related warnings, so the census failure of EXP-002 run 20260915T180708
(zero commands per call on every configuration) can be attributed.
"""

from __future__ import annotations

import collections
import warnings

import torch
from torch.profiler import ProfilerActivity, profile


def main() -> int:
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("no CUDA device; nothing to diagnose")
        return 1
    dev = torch.device("cuda", 0)
    a = torch.randn(512, 512, device=dev)
    b = torch.randn(512, 512, device=dev)
    c = torch.empty(512, 512, device=dev)
    for _ in range(5):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize(dev)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                torch.mm(a, b, out=c)
            torch.cuda.synchronize(dev)
    for w in caught:
        print("warning during profiling:", w.message)

    events = list(prof.events())
    by_type = collections.Counter(str(getattr(e, "device_type", None)) for e in events)
    print("events total:", len(events), "| by device_type:", dict(by_type))
    device_names = collections.Counter(e.name for e in events if "CUDA" in str(getattr(e, "device_type", "")))
    print("device event names:", dict(device_names) or "(none)")
    attached = sum(len(getattr(e, "kernels", None) or []) for e in events)
    print("kernels attached to CPU op events (FunctionEvent.kernels):", attached)
    ka = prof.key_averages()
    rows = [(r.key, getattr(r, "device_time_total", getattr(r, "cuda_time_total", 0)), r.count) for r in ka]
    rows = [r for r in rows if r[1] and r[1] > 0]
    print("key_averages rows with device time:", len(rows))
    for name, t, n in sorted(rows, key=lambda r: -r[1])[:8]:
        print(f"   {name[:60]:60s} device_time_total_us={t:10.1f} count={n}")
    try:
        prof.export_chrome_trace("profiler_diag_trace.json")
        print("trace written to profiler_diag_trace.json (inspect for 'kernel' entries)")
    except Exception as err:  # noqa: BLE001
        print("trace export failed:", err)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
