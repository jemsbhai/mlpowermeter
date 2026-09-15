"""Experiment modules.

Each experiment is a module exposing ``run(ctx: RunContext) -> dict``. The
generic runner (``scripts/run_experiment.py``) handles the lab-runner
boilerplate: run directory, frozen config, environment snapshot, seeds,
manifest, logging, device resolution. The module handles the science and
returns a JSON-serializable summary.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict


@dataclass
class RunContext:
    exp_id: str
    name: str
    platform_tag: str
    run_id: str
    out_dir: Path
    config: Dict[str, Any]
    dev: Any                       # tomlml.measure.NvmlDevice
    sync_fn: Callable[[], None]
    logger: logging.Logger
    quick: bool = False
    quick_factor: float = 1.0
    torch_device: int = 0
    seed: int = 42
    rng: Any = None                # numpy.random.Generator
    resume: bool = False           # True when continuing an interrupted run in its own directory
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def results_dir(self) -> Path:
        return self.out_dir / "results"

    @property
    def samples_dir(self) -> Path:
        return self.out_dir / "samples"

    @property
    def energy_source(self) -> str:
        """Energy source setting for this run (D-009): from the platform
        overlay or the experiment config, ``auto`` when neither sets it."""
        return str(self.config.get("energy_source", "auto"))

    def scaled(self, value: float, minimum: float = 0.0) -> float:
        """Scale a duration by the quick factor (identity for full runs)."""
        return max(minimum, float(value) * self.quick_factor)

    def scaled_int(self, value: int, minimum: int = 1) -> int:
        return max(int(minimum), int(round(int(value) * self.quick_factor)))


def load_experiment(name: str) -> ModuleType:
    """Import ``tomlml.experiments.<name>`` and check it exposes ``run``."""
    module = importlib.import_module(f"tomlml.experiments.{name}")
    if not hasattr(module, "run"):
        raise AttributeError(f"experiment module {name!r} has no run(ctx) function")
    return module


__all__ = ["RunContext", "load_experiment"]
