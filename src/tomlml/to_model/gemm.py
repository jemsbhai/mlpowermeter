"""TO descriptors for dense and factorized linear maps (EXP-002, EXP-003).

For y = x W with W of shape (d_in, d_out) and exact rank r, W = V U with
V (d_in, r) and U (r, d_out):

    dense       y = x W                 B d_in d_out MACs, one GEMM
    factorized  y = (x V) U             B r (d_in + d_out) MACs, two GEMMs,
                                        intermediate z of shape (B, r)

Memory words count each operand read or written once per call (the
TOMLSignals convention): inputs, weights, intermediate written then read,
outputs. Weights are charged at HBM cost unless the caller declares them
L2-resident (model M2, D-010 item 5), in which case they are charged at SRAM
cost. FP32 words throughout (4 bytes).

The dispatch term (command count) is measured, not derived (D-010 item 4);
it is attached to a descriptor by the experiment from the census.
"""

from __future__ import annotations

from typing import Any, Dict

from .costs import TO

BYTES_PER_WORD = 4


def dense_descriptor(B: int, d_in: int, d_out: int, weights_in_l2: bool = False) -> Dict[str, Any]:
    n_mac = B * d_in * d_out
    words_x = B * d_in
    words_w = d_in * d_out
    words_y = B * d_out
    weight_cost = TO["mem_sram"] if weights_in_l2 else TO["mem_hbm"]
    to_memory = (words_x + words_y) * TO["mem_hbm"] + words_w * weight_cost
    return {
        "realization": "dense",
        "B": B, "d_in": d_in, "d_out": d_out, "r": None,
        "n_mac": n_mac,
        "n_gemm": 1,
        "words": {"x": words_x, "weights": words_w, "intermediate": 0, "y": words_y},
        "weight_bytes": words_w * BYTES_PER_WORD,
        "weights_in_l2": weights_in_l2,
        "to_compute": n_mac * TO["mac"],
        "to_memory": to_memory,
    }


def factorized_descriptor(B: int, d_in: int, d_out: int, r: int,
                          weights_in_l2: bool = False) -> Dict[str, Any]:
    n_mac = B * r * (d_in + d_out)
    words_x = B * d_in
    words_w = d_in * r + r * d_out
    words_z = 2 * B * r  # written by the first GEMM, read by the second
    words_y = B * d_out
    weight_cost = TO["mem_sram"] if weights_in_l2 else TO["mem_hbm"]
    to_memory = (words_x + words_z + words_y) * TO["mem_hbm"] + words_w * weight_cost
    return {
        "realization": "factorized",
        "B": B, "d_in": d_in, "d_out": d_out, "r": r,
        "n_mac": n_mac,
        "n_gemm": 2,
        "words": {"x": words_x, "weights": words_w, "intermediate": words_z, "y": words_y},
        "weight_bytes": words_w * BYTES_PER_WORD,
        "weights_in_l2": weights_in_l2,
        "to_compute": n_mac * TO["mac"],
        "to_memory": to_memory,
    }


def flops_crossover_rank(d_in: int, d_out: int) -> float:
    """Rank at which factorized and dense MAC counts are equal: d_in d_out /
    (d_in + d_out), which is d/2 for square maps. Independent of batch."""
    return d_in * d_out / (d_in + d_out)


def weight_sets_for_l2(d_in: int, d_out: int, l2_bytes: int, multiple: float = 2.0) -> int:
    """Number of independent dense weight sets whose combined footprint is at
    least ``multiple`` times the L2 cache (D-010 item 5). At least 1."""
    per_set = d_in * d_out * BYTES_PER_WORD
    if per_set <= 0:
        return 1
    return max(1, int(-(-multiple * l2_bytes // per_set)))


def factorized_weights_fit_l2(d_in: int, d_out: int, r: int, n_sets: int, l2_bytes: int) -> bool:
    """Whether the cycled factorized weights (all sets) fit in L2, the M2 rule."""
    return n_sets * (d_in * r + r * d_out) * BYTES_PER_WORD <= l2_bytes
