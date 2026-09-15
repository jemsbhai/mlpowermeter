"""Tests for the dense and factorized GEMM descriptors and the exact-rank workload."""

from __future__ import annotations

import pytest

from tomlml.to_model import TO
from tomlml.to_model.gemm import (
    dense_descriptor, factorized_descriptor, factorized_weights_fit_l2, flops_crossover_rank,
    weight_sets_for_l2,
)
from tomlml.workloads.lowrank import rank_for


def test_dense_descriptor_counts():
    d = dense_descriptor(B=16, d_in=1024, d_out=2048)
    assert d["n_mac"] == 16 * 1024 * 2048 and d["n_gemm"] == 1 and d["r"] is None
    assert d["words"] == {"x": 16 * 1024, "weights": 1024 * 2048, "intermediate": 0, "y": 16 * 2048}
    assert d["to_compute"] == d["n_mac"] * TO["mac"]
    assert d["to_memory"] == (16 * 1024 + 16 * 2048 + 1024 * 2048) * TO["mem_hbm"]
    assert d["weight_bytes"] == 1024 * 2048 * 4
    d2 = dense_descriptor(B=16, d_in=1024, d_out=2048, weights_in_l2=True)
    assert d2["to_memory"] == (16 * 1024 + 16 * 2048) * TO["mem_hbm"] + 1024 * 2048 * TO["mem_sram"]


def test_factorized_descriptor_counts():
    f = factorized_descriptor(B=4, d_in=1024, d_out=1024, r=64)
    assert f["n_mac"] == 4 * 64 * 2048 and f["n_gemm"] == 2 and f["r"] == 64
    assert f["words"] == {"x": 4096, "weights": 2 * 1024 * 64, "intermediate": 2 * 4 * 64, "y": 4096}
    assert f["to_memory"] == (4096 + 512 + 4096) * TO["mem_hbm"] + 2 * 1024 * 64 * TO["mem_hbm"]
    f2 = factorized_descriptor(B=4, d_in=1024, d_out=1024, r=64, weights_in_l2=True)
    assert f2["to_memory"] == (4096 + 512 + 4096) * TO["mem_hbm"] + 2 * 1024 * 64 * TO["mem_sram"]
    # at the FLOPs crossover the MAC counts match
    r_star = flops_crossover_rank(1024, 1024)
    assert r_star == 512
    assert factorized_descriptor(4, 1024, 1024, 512)["n_mac"] == dense_descriptor(4, 1024, 1024)["n_mac"]
    assert flops_crossover_rank(1024, 3072) == pytest.approx(768.0)


def test_weight_set_cycling_rule():
    l2 = 64 * 2**20
    assert weight_sets_for_l2(4096, 4096, l2) == 2          # 64 MB per set, need 128 MB
    assert weight_sets_for_l2(2048, 2048, l2) == 8
    assert weight_sets_for_l2(512, 512, l2) == 128
    assert weight_sets_for_l2(8192, 8192, l2) == 1          # one set already exceeds 2x L2
    assert weight_sets_for_l2(1024, 1024, 40 * 2**20) == 20
    assert factorized_weights_fit_l2(4096, 4096, 64, 2, l2) is True     # 2 x 2 MB
    assert factorized_weights_fit_l2(4096, 4096, 3072, 2, l2) is False  # 2 x 96 MB


def test_rank_for():
    assert rank_for(512, 1 / 64) == 8 and rank_for(4096, 0.75) == 3072
    assert rank_for(4, 0.01) == 1


@pytest.mark.parametrize("d,B", [(64, 1), (64, 8)])
def test_lowrank_shape_exact_pair_on_cpu(d, B):
    torch = pytest.importorskip("torch")
    from tomlml.workloads.lowrank import LowRankShape

    shape = LowRankShape(d, [0.125, 0.5], [B], n_sets=3, device="cpu")
    shape.setup()
    try:
        assert shape.ranks == {0.125: 8, 0.5: 32}
        ex = shape.exactness(0.5, B)
        assert ex["sets_checked"] == 3 and ex["max_rel_frobenius"] < 1e-4
        run_d = shape.runner(0.5, B, "dense")
        run_f = shape.runner(0.5, B, "factorized")
        run_d()
        yd = shape._Y[B].clone()
        run_f()
        yf = shape._Y[B].clone()
        # runners cycle sets: call 0 used set 0 in both, so outputs agree
        assert float(torch.linalg.norm(yd - yf) / torch.linalg.norm(yd)) < 1e-4
        desc = shape.describe()
        assert desc["n_sets"] == 3 and desc["ranks"] == {"0.125": 8, "0.5": 32}
        with pytest.raises(ValueError):
            shape.runner(0.5, B, "sparse")
    finally:
        shape.teardown()
    assert shape._W == {}
