"""Exact low-rank linear maps: dense versus factorized realizations (EXP-002).

For one shape d, a ``LowRankShape`` holds, for every rank fraction f and
batch size B in the grid: inputs X (B, d), K independent weight sets
V_k (d, r), U_k (r, d) and W_k = fp32(fp64(V_k) fp64(U_k)) (d, d), the
output Y (B, d) and the intermediate Z (B, r). Both realizations compute the
same map from the same inputs with the same weight set, cycling through the
K sets call by call (D-010 item 5). Nothing is allocated in the measured
loop.

The class runs on CUDA for measurement and on CPU for the test suite.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

from ..utils.seeds import derive_seed

REALIZATIONS = ("dense", "factorized")


def rank_for(d: int, fraction: float) -> int:
    return max(1, int(round(d * float(fraction))))


class LowRankShape:
    def __init__(self, d: int, rank_fractions: List[float], batches: List[int], n_sets: int,
                 device: str = "cuda:0", data_seed: int = 1234, dtype: str = "float32"):
        self.d = int(d)
        self.rank_fractions = [float(f) for f in rank_fractions]
        self.batches = [int(b) for b in batches]
        self.n_sets = max(1, int(n_sets))
        self.device = device
        self.data_seed = int(data_seed)
        self.dtype = dtype
        self.ranks: Dict[float, int] = {f: rank_for(self.d, f) for f in self.rank_fractions}
        self._X: Dict[int, Any] = {}
        self._Y: Dict[int, Any] = {}
        self._Z: Dict[tuple, Any] = {}
        self._W: Dict[float, List[Any]] = {}
        self._V: Dict[float, List[Any]] = {}
        self._U: Dict[float, List[Any]] = {}
        self._ready = False

    # -- construction ------------------------------------------------------ #

    def _generator(self, name: str):
        import torch
        dev = torch.device(self.device)
        g = torch.Generator(device="cuda" if dev.type == "cuda" else "cpu")
        g.manual_seed(derive_seed(self.data_seed, f"{name}:d{self.d}"))
        return g

    def setup(self) -> None:
        import torch
        dt = getattr(torch, self.dtype)
        dev = torch.device(self.device)
        for B in self.batches:
            g = self._generator(f"x:B{B}")
            self._X[B] = torch.randn(B, self.d, generator=g, device=dev, dtype=torch.float32).to(dt)
            self._Y[B] = torch.empty(B, self.d, device=dev, dtype=dt)
        for f in self.rank_fractions:
            r = self.ranks[f]
            self._W[f], self._V[f], self._U[f] = [], [], []
            for k in range(self.n_sets):
                g = self._generator(f"w:f{f}:k{k}")
                V = torch.randn(self.d, r, generator=g, device=dev, dtype=torch.float32)
                U = torch.randn(r, self.d, generator=g, device=dev, dtype=torch.float32) / math.sqrt(r)
                W = (V.double() @ U.double()).to(dt)
                self._V[f].append(V.to(dt))
                self._U[f].append(U.to(dt))
                self._W[f].append(W)
            for B in self.batches:
                self._Z[(B, f)] = torch.empty(B, r, device=dev, dtype=dt)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        self._ready = True

    def teardown(self) -> None:
        self._X.clear(); self._Y.clear(); self._Z.clear()
        self._W.clear(); self._V.clear(); self._U.clear()
        self._ready = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- runners ------------------------------------------------------------ #

    def runner(self, f: float, B: int, realization: str) -> Callable[[], None]:
        """Zero-argument callable executing one call, cycling weight sets."""
        import torch
        if realization not in REALIZATIONS:
            raise ValueError(f"realization must be one of {REALIZATIONS}, got {realization!r}")
        X, Y = self._X[B], self._Y[B]
        K = self.n_sets
        state = {"i": 0}
        if realization == "dense":
            Ws = self._W[f]

            def run_dense() -> None:
                k = state["i"] % K
                state["i"] += 1
                torch.mm(X, Ws[k], out=Y)
            return run_dense
        Vs, Us, Z = self._V[f], self._U[f], self._Z[(B, f)]

        def run_factorized() -> None:
            k = state["i"] % K
            state["i"] += 1
            torch.mm(X, Vs[k], out=Z)
            torch.mm(Z, Us[k], out=Y)
        return run_factorized

    # -- checks and description --------------------------------------------- #

    def exactness(self, f: float, B: int, n_sets_check: int = 4) -> Dict[str, Any]:
        """Relative Frobenius difference between the two realizations' outputs
        over the first ``n_sets_check`` weight sets (D-010 item 6)."""
        import torch
        X = self._X[B]
        rels = []
        for k in range(min(self.n_sets, n_sets_check)):
            Yd = torch.mm(X, self._W[f][k])
            Yf = torch.mm(torch.mm(X, self._V[f][k]), self._U[f][k])
            denom = torch.linalg.norm(Yd.double())
            rels.append(float(torch.linalg.norm((Yf - Yd).double()) / denom) if denom > 0 else 0.0)
        return {"d": self.d, "f": f, "r": self.ranks[f], "B": B, "sets_checked": len(rels),
                "rel_frobenius": rels, "max_rel_frobenius": max(rels) if rels else None}

    def describe(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "d": self.d,
            "rank_fractions": self.rank_fractions,
            "ranks": {f"{f:g}": r for f, r in self.ranks.items()},
            "batches": self.batches,
            "n_sets": self.n_sets,
            "dense_weight_bytes_per_set": self.d * self.d * 4,
            "factorized_weight_bytes_per_set": {f"{f:g}": 2 * self.d * r * 4 for f, r in self.ranks.items()},
            "device": self.device,
            "dtype": self.dtype,
            "data_seed": self.data_seed,
            "weight_scale": "V ~ N(0,1), U ~ N(0,1)/sqrt(r), W = fp32(fp64(V) fp64(U))",
        }
        try:
            import torch
            info["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
            info["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        except ImportError:
            pass
        return info


def l2_cache_bytes(device_index: int = 0, fallback_bytes: int = 64 * 1024 * 1024) -> Dict[str, Any]:
    """L2 cache size from torch device properties, with a recorded fallback."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_index)
            size = getattr(props, "L2_cache_size", None)
            if size:
                return {"bytes": int(size), "source": "torch.cuda.get_device_properties"}
    except ImportError:
        pass
    return {"bytes": int(fallback_bytes), "source": "fallback (assumed 64 MiB)"}


def torch_device_string(device_index: Optional[int]) -> str:
    """``cuda:<i>`` when CUDA is available, else ``cpu`` (test suite)."""
    try:
        import torch
        if device_index is not None and torch.cuda.is_available():
            return f"cuda:{device_index}"
    except ImportError:
        pass
    return "cpu"
