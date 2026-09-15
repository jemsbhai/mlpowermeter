"""Transistor-operation (TO) cost table and counting helpers.

The table is inherited unchanged from FLAIRS-39 and IEEE MLSP 2026
(TOMLSignals, shared/to_model.py) as the FP32 baseline (DECISIONS.md D-005).
Costs are in transistor operations at the 45 nm reference node, consistent
with Horowitz (ISSCC 2014) energy-per-operation data.

Methodology recap:

* Each floating-point operation on a GPU dispatches a full FMA unit, so one
  FP add, multiply, or FMA counts as one MAC (5,000 TOs).
* Transcendentals use iterative or polynomial algorithms taking several FMA
  cycles, hence the higher costs.
* Memory distinguishes on-chip SRAM words from off-chip HBM words.
* Counts are derived from an algorithm's mathematical definition, not from a
  particular kernel; implementation effects (dispatch, fusion, library
  overhead) are captured by the fitted device coefficients, not by the table.

Precision-specific costs (FP16, BF16, TF32, INT8) are not yet defined; asking
for them raises ``NotImplementedError`` until a dated decision adds them.
"""

from __future__ import annotations

from typing import Dict, Mapping

TO: Dict[str, int] = {
    "mac":       5_000,   # FP32 multiply-accumulate (one FMA dispatch)
    "div":      15_000,   # division (iterative, about 3 FMA cycles)
    "sqrt":     15_000,   # square root (iterative, about 3 FMA cycles)
    "exp":      18_000,   # exponential (polynomial plus range reduction)
    "log":      18_000,   # logarithm (same cost class as exp)
    "sin":      18_000,   # sine (polynomial approximation)
    "cos":      18_000,   # cosine (polynomial approximation)
    "tanh":     15_000,   # hyperbolic tangent
    "sigmoid":  18_000,   # sigmoid = exp plus division
    "relu":        100,   # comparison plus mux
    "gelu":     20_000,   # erf approximation
    "softmax":  25_000,   # per element: exp plus sum plus division
    "layernorm": 12_000,  # per element (TOMLTransformers)
    "cmp":          50,   # single comparison
    "abs":       5_000,   # mask plus select, dispatches a full FMA
    "neg":       5_000,   # sign flip, dispatches a full FMA
    "mem_sram":    192,   # on-chip read or write, 32-bit word
    "mem_hbm":  10_000,   # off-chip read or write, 32-bit word
}

SUPPORTED_PRECISIONS = ("fp32",)

COMPUTE_OPS = tuple(k for k in TO if not k.startswith("mem_"))
MEMORY_OPS = ("mem_sram", "mem_hbm")


def _check_precision(precision: str) -> None:
    if precision not in SUPPORTED_PRECISIONS:
        raise NotImplementedError(
            f"precision {precision!r} has no TO costs yet; supported: {SUPPORTED_PRECISIONS} "
            "(precision-specific costs require a dated DECISIONS.md entry, D-005)")


def op_tos(op: str, count: float = 1, precision: str = "fp32") -> float:
    """TOs for ``count`` operations of type ``op``."""
    _check_precision(precision)
    if op not in TO:
        raise KeyError(f"unknown operation {op!r}; known: {sorted(TO)}")
    return float(count) * TO[op]


def memory_tos(n_words: float, level: str = "hbm") -> float:
    """TOs for moving ``n_words`` 32-bit words at the given memory level."""
    key = f"mem_{level}"
    if key not in TO:
        raise KeyError(f"unknown memory level {level!r}; known: sram, hbm")
    return float(n_words) * TO[key]


def tos_from_counts(counts: Mapping[str, float], precision: str = "fp32") -> Dict[str, float]:
    """Split a dict of operation counts into compute, memory, and total TOs.

    ``counts`` maps operation names (keys of ``TO``) to counts.
    """
    _check_precision(precision)
    compute = 0.0
    memory = 0.0
    for op, n in counts.items():
        if op not in TO:
            raise KeyError(f"unknown operation {op!r}; known: {sorted(TO)}")
        if op in MEMORY_OPS:
            memory += float(n) * TO[op]
        else:
            compute += float(n) * TO[op]
    return {"to_compute": compute, "to_memory": memory, "to_total": compute + memory}
