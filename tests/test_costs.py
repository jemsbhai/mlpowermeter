"""Tests for the TO cost table and counting helpers."""

from __future__ import annotations

import pytest

from tomlml.to_model import COMPUTE_OPS, MEMORY_OPS, TO, memory_tos, op_tos, tos_from_counts


PUBLISHED = {
    "mac": 5_000, "div": 15_000, "sqrt": 15_000, "exp": 18_000, "sin": 18_000,
    "cos": 18_000, "tanh": 15_000, "sigmoid": 18_000, "relu": 100, "gelu": 20_000,
    "softmax": 25_000, "cmp": 50, "abs": 5_000, "neg": 5_000,
    "mem_sram": 192, "mem_hbm": 10_000,
}


def test_table_matches_published_constants():
    for op, cost in PUBLISHED.items():
        assert TO[op] == cost, op
    assert TO["log"] == TO["exp"]
    assert TO["layernorm"] == 12_000


def test_compute_and_memory_op_partition():
    assert set(MEMORY_OPS) == {"mem_sram", "mem_hbm"}
    assert set(COMPUTE_OPS) | set(MEMORY_OPS) == set(TO)
    assert not set(COMPUTE_OPS) & set(MEMORY_OPS)


def test_op_tos_and_memory_tos():
    assert op_tos("mac", 3) == 15_000
    assert op_tos("div") == 15_000
    assert memory_tos(2, "hbm") == 20_000
    assert memory_tos(10, "sram") == 1_920
    with pytest.raises(KeyError):
        op_tos("fma")
    with pytest.raises(KeyError):
        memory_tos(1, "l2")


def test_tos_from_counts_splits_compute_and_memory():
    r = tos_from_counts({"mac": 100, "exp": 2, "mem_hbm": 4, "mem_sram": 10})
    assert r["to_compute"] == 100 * 5_000 + 2 * 18_000
    assert r["to_memory"] == 4 * 10_000 + 10 * 192
    assert r["to_total"] == r["to_compute"] + r["to_memory"]
    with pytest.raises(KeyError):
        tos_from_counts({"nope": 1})


def test_precision_other_than_fp32_is_refused():
    with pytest.raises(NotImplementedError):
        op_tos("mac", 1, precision="fp16")
    with pytest.raises(NotImplementedError):
        tos_from_counts({"mac": 1}, precision="bf16")
