"""GPU command census with torch.profiler (D-010 item 4).

The dispatch term of the model counts commands issued to the device per
call: kernels plus memcpy plus memset. TOMLSignals measured this with Nsight
Systems (F-022); the profiler gives the same census portably, and the
laptop cross-checks a subset against nsys once.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable


def classify_command(name: str) -> str:
    """``memcpy``, ``memset``, or ``kernel`` from a CUDA event name."""
    n = name.lower()
    if "memcpy" in n:
        return "memcpy"
    if "memset" in n:
        return "memset"
    return "kernel"


def count_commands_from_events(events: Iterable[Any], n_calls: int,
                               is_device_event: Callable[[Any], bool]) -> Dict[str, Any]:
    """Aggregate device events into per-call command counts.

    ``events`` are profiler events; ``is_device_event`` selects the ones that
    executed on the device. Names are kept so kernel selection (for example
    gemv versus gemm, split-K) is visible in the results.
    """
    kernels: Dict[str, int] = {}
    memcpy = 0
    memset = 0
    for evt in events:
        if not is_device_event(evt):
            continue
        name = str(getattr(evt, "name", ""))
        kind = classify_command(name)
        if kind == "memcpy":
            memcpy += 1
        elif kind == "memset":
            memset += 1
        else:
            kernels[name] = kernels.get(name, 0) + 1
    n_kernels = sum(kernels.values())
    total = n_kernels + memcpy + memset
    n = max(1, int(n_calls))
    return {
        "n_calls": n,
        "kernels_per_call": n_kernels / n,
        "memcpy_per_call": memcpy / n,
        "memset_per_call": memset / n,
        "commands_per_call": total / n,
        "kernel_names_per_call": {k: v / n for k, v in sorted(kernels.items())},
        "distinct_kernels": len(kernels),
    }


def command_census(run_once: Callable[[], None], sync_fn: Callable[[], None],
                   warmup_calls: int = 5, n_calls: int = 10) -> Dict[str, Any]:
    """Profile ``n_calls`` calls after ``warmup_calls`` unprofiled ones and
    return per-call command counts. Requires torch with CUDA."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup_calls):
        run_once()
    sync_fn()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(n_calls):
            run_once()
        sync_fn()
    cuda_type = torch.autograd.DeviceType.CUDA

    def is_device(evt: Any) -> bool:
        return getattr(evt, "device_type", None) == cuda_type

    out = count_commands_from_events(prof.events(), n_calls, is_device)
    out["method"] = "torch.profiler"
    out["warmup_calls"] = warmup_calls
    return out
