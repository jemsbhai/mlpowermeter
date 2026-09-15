"""Seed management.

Every stochastic library is seeded from one master seed and the resulting
record is written to ``seed.json`` in the experiment directory.

Deterministic algorithm selection is deliberately NOT forced by default.
Forcing ``torch.use_deterministic_algorithms(True)`` can change which kernels
run (and therefore the energy being measured), so measured workloads execute
exactly as they would in practice, and the flag's state is recorded. Runs that
need bitwise reproducibility of outputs (not energy) opt in with
``deterministic=True`` (DECISIONS.md D-008).
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any, Dict, Optional


def derive_seed(master_seed: int, name: str) -> int:
    """Stable sub-seed for a named component (e.g. 'data_split', 'init')."""
    h = hashlib.sha256(f"{int(master_seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % (2**31 - 1)


def set_all_seeds(master_seed: int, deterministic: bool = False) -> Dict[str, Any]:
    """Seed random, numpy, and torch (CPU and all CUDA devices) from one value.

    Returns the record to store as seed.json.
    """
    master_seed = int(master_seed)
    record: Dict[str, Any] = {
        "master_seed": master_seed,
        "python_random": master_seed,
        "numpy": None,
        "torch": None,
        "torch_cuda": None,
        "deterministic_algorithms": False,
        "cudnn_benchmark": None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "notes": [],
    }
    random.seed(master_seed)

    try:
        import numpy as np
        np.random.seed(master_seed)
        record["numpy"] = master_seed
    except ImportError:
        record["notes"].append("numpy not installed")

    try:
        import torch
        torch.manual_seed(master_seed)
        record["torch"] = master_seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(master_seed)
            record["torch_cuda"] = master_seed
        if deterministic:
            if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                record["cublas_workspace_config"] = ":4096:8"
                record["notes"].append(
                    "CUBLAS_WORKSPACE_CONFIG set at seed time; it only takes effect "
                    "if no CUDA context existed yet")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            record["deterministic_algorithms"] = True
            record["notes"].append(
                "deterministic algorithms forced (warn_only); kernel selection "
                "may differ from the default execution path")
        else:
            record["notes"].append(
                "deterministic algorithms not forced: measured workloads run "
                "their default kernels (D-008)")
        record["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except ImportError:
        record["notes"].append("torch not installed")

    return record


def seed_record_for_config(config: Dict[str, Any], default_seed: int = 42) -> int:
    """Master seed from a config dict (``seed`` key), else the default."""
    value: Optional[Any] = config.get("seed", default_seed)
    return int(default_seed if value is None else value)
