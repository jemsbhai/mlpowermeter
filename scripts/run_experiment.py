#!/usr/bin/env python
"""Generic experiment runner (lab-runner boilerplate around a module's run()).

    python scripts/run_experiment.py --config configs/exp_001_instrumentation.yaml
    python scripts/run_experiment.py --config configs/exp_001_instrumentation.yaml --quick
    python scripts/run_experiment.py --config configs/exp_002_lowrank_crossover.yaml --resume 20260915T200000

New run: load and merge config (configs/base.yaml, then the platform overlay
configs/platform/<tag>.yaml, then the experiment file), check the git tree
is clean (full runs only), seed everything, resolve the GPU, create the run
directory, freeze the effective config, write seed.json, environment.json
and manifest.json, run the experiment module, finalize the manifest.

Resume: reuse the interrupted run's directory. The frozen config.yaml of
that run is the config (not the current files), the seeds are re-applied so
the randomized order is identical, the git commit must equal the one in the
manifest (a code change means a new experiment, per protocol), and the
experiment module skips whatever its ledger says is done. Every resume is
recorded in the manifest with its own environment snapshot.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from tomlml.experiments import RunContext, load_experiment
from tomlml.measure import make_cuda_sync, nvml_shutdown, open_device
from tomlml.utils import (
    build_manifest, derive_seed, experiment_folder, finalize_manifest, find_repo_root, git_info,
    make_experiment_dir, platform_tag_from_gpu_name, read_json, seed_record_for_config,
    set_all_seeds, sha256_file, snapshot_environment, utc_now_iso, write_json,
)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Path, base_path: Optional[Path],
                platform_path: Optional[Path] = None) -> Dict[str, Any]:
    """Merge base -> platform -> experiment (later files override earlier)."""
    config: Dict[str, Any] = {}
    sources = []
    for path in (base_path, platform_path):
        if path is not None and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                config = deep_merge(config, yaml.safe_load(f) or {})
            sources.append({"path": path.as_posix(), "sha256": sha256_file(path)})
    with open(config_path, "r", encoding="utf-8") as f:
        config = deep_merge(config, yaml.safe_load(f) or {})
    sources.append({"path": config_path.as_posix(), "sha256": sha256_file(config_path)})
    config["_sources"] = sources
    return config


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("tomlml")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _execute(module_name: str, ctx: RunContext, manifest: Dict[str, Any], out_dir: Path,
             logger: logging.Logger) -> Optional[Dict[str, Any]]:
    module = load_experiment(module_name)
    status = "failed"
    summary: Optional[Dict[str, Any]] = None
    try:
        summary = module.run(ctx)
        status = "completed"
    except BaseException as err:  # noqa: BLE001 - recorded, then re-raised
        logger.exception("experiment failed: %s", err)
        manifest["error"] = f"{type(err).__name__}: {err}"
        status = "interrupted" if isinstance(err, KeyboardInterrupt) else "failed"
        raise
    finally:
        brief = None
        if summary is not None:
            brief = {"gate_pass": summary.get("gate_pass"), "gate_valid": summary.get("gate_valid"),
                     "analysis_pending": summary.get("analysis_pending"),
                     "summary_path": (out_dir / "results" / "summary.json").as_posix()}
        finalize_manifest(manifest, status, summary=brief)
        write_json(out_dir / "manifest.json", manifest)
        logger.info("status %s; manifest %s", status, out_dir / "manifest.json")
    return summary


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment config YAML")
    p.add_argument("--base", default=None, help="base config (default: base.yaml next to --config)")
    p.add_argument("--platform-tag", default=None, help="override the auto-detected platform tag")
    p.add_argument("--torch-device", type=int, default=None, help="torch CUDA device index")
    p.add_argument("--run-id", default=None, help="override the timestamp run id")
    p.add_argument("--out-root", default=None, help="repository root for experiments/ (default: auto)")
    p.add_argument("--quick", action="store_true", help="scale durations down for a pipeline smoke test")
    p.add_argument("--quick-factor", type=float, default=0.05)
    p.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes (recorded)")
    p.add_argument("--resume", default=None, metavar="RUN_ID",
                   help="continue an interrupted run of this experiment in its existing directory")
    p.add_argument("--resume-any-commit", action="store_true",
                   help="allow resuming even if the code commit differs from the original run (recorded)")
    args = p.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    base_path = Path(args.base).resolve() if args.base else cfg_path.parent / "base.yaml"
    config = load_config(cfg_path, base_path)
    exp_id = str(config["experiment_id"])
    name = str(config["name"])
    module_name = str(config["experiment"])

    repo_root = find_repo_root(cfg_path) or find_repo_root(Path(__file__)) or Path.cwd()
    out_root = Path(args.out_root).resolve() if args.out_root else repo_root
    git = git_info(repo_root)
    quick = bool(args.quick)
    if git.get("dirty") and not quick and not (args.allow_dirty or config.get("allow_dirty")):
        print("Refusing to start a full run on a dirty working tree. Commit first, or pass "
              "--allow-dirty (the dirty flag is recorded in the manifest either way).",
              file=sys.stderr)
        return 2

    torch_device = args.torch_device if args.torch_device is not None else int(config.get("torch_device", 0))
    dev = open_device(torch_index=torch_device, override_uuid=os.environ.get("TOMLML_NVML_UUID") or None)
    try:
        platform_tag = (args.platform_tag or os.environ.get("TOMLML_PLATFORM_TAG")
                        or config.get("platform_tag") or platform_tag_from_gpu_name(dev.name))

        if args.resume:
            return _resume(args, config, exp_id, name, module_name, repo_root, out_root, git,
                           torch_device, dev, platform_tag)

        # Platform overlay (configs/platform/<tag>.yaml) sits between base and
        # experiment config; it carries what the platform's gate run established
        # (energy source, window defaults). Re-merge now that the tag is known.
        platform_path = cfg_path.parent / "platform" / f"{platform_tag}.yaml"
        if platform_path.exists():
            config = load_config(cfg_path, base_path, platform_path)
        seed = seed_record_for_config(config)
        seed_record = set_all_seeds(seed, deterministic=bool(config.get("deterministic", False)))
        rng = np.random.default_rng(derive_seed(seed, "experiment_order"))

        out_dir = make_experiment_dir(out_root, exp_id, name, platform_tag, run_id=args.run_id)
        run_id = out_dir.name
        logger = setup_logger(out_dir / "logs" / "run.log")
        logger.info("%s %s on %s (run %s) -> %s", exp_id, name, platform_tag, run_id, out_dir)
        logger.info("config sources: %s", ", ".join(Path(s["path"]).name for s in config["_sources"]))
        logger.info("energy source setting: %s", config.get("energy_source", "auto"))
        logger.info("device: nvml index %s, %s, uuid %s, resolved by %s, uuid_verified=%s%s",
                    dev.index, dev.name, dev.uuid, dev.resolved.method, dev.resolved.uuid_verified,
                    (" notes: " + "; ".join(dev.resolved.notes)) if dev.resolved.notes else "")
        if git.get("dirty"):
            logger.warning("working tree is dirty (commit %s)", git.get("commit"))

        quick_factor = float(args.quick_factor) if quick else 1.0
        config["quick"] = quick
        config["quick_factor"] = quick_factor
        with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        write_json(out_dir / "seed.json", seed_record)
        env = snapshot_environment(
            device_state={"resolved": dev.resolved.to_dict(), "state": dev.state()},
            repo_root=str(repo_root))
        write_json(out_dir / "environment.json", env)
        manifest = build_manifest(exp_id, name, platform_tag, run_id, out_dir, config, git,
                                  device=dev.resolved.to_dict(),
                                  extra={"quick": quick, "quick_factor": quick_factor,
                                         "hostname": env.get("hostname"),
                                         "torch_device": torch_device, "seed": seed, "resumes": []})
        write_json(out_dir / "manifest.json", manifest)

        ctx = RunContext(exp_id=exp_id, name=name, platform_tag=platform_tag, run_id=run_id,
                         out_dir=out_dir, config=config, dev=dev,
                         sync_fn=make_cuda_sync(torch_device), logger=logger, quick=quick,
                         quick_factor=quick_factor, torch_device=torch_device, seed=seed, rng=rng)
        summary = _execute(module_name, ctx, manifest, out_dir, logger)
        return 0 if quick or summary is None or summary.get("gate_pass", True) else 1
    finally:
        nvml_shutdown()


def _resume(args: argparse.Namespace, fresh_config: Dict[str, Any], exp_id: str, name: str,
            module_name: str, repo_root: Path, out_root: Path, git: Dict[str, Any],
            torch_device: int, dev: Any, platform_tag: str) -> int:
    run_dir = Path(args.resume)
    if not run_dir.is_dir():
        run_dir = out_root / "experiments" / experiment_folder(exp_id, name) / platform_tag / args.resume
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Cannot resume: no manifest at {manifest_path}", file=sys.stderr)
        return 3
    manifest = read_json(manifest_path)
    if manifest.get("experiment_id") != exp_id:
        print(f"Cannot resume: run is {manifest.get('experiment_id')}, config is {exp_id}", file=sys.stderr)
        return 3
    if manifest.get("status") == "completed":
        print(f"Cannot resume: run {run_dir.name} already completed", file=sys.stderr)
        return 3
    original_commit = (manifest.get("git") or {}).get("commit")
    if original_commit != git.get("commit") and not args.resume_any_commit:
        print(f"Cannot resume: run was started at commit {original_commit}, HEAD is {git.get('commit')}. "
              "A code change means a new experiment (protocol); pass --resume-any-commit to override "
              "(recorded in the manifest).", file=sys.stderr)
        return 3

    with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    quick = bool(config.get("quick", False))
    quick_factor = float(config.get("quick_factor", 1.0))
    seed = seed_record_for_config(config)
    set_all_seeds(seed, deterministic=bool(config.get("deterministic", False)))
    rng = np.random.default_rng(derive_seed(seed, "experiment_order"))

    n_resume = len(manifest.get("resumes") or []) + 1
    logger = setup_logger(run_dir / "logs" / "run.log")
    logger.info("RESUME %d of %s %s on %s (run %s) at commit %s%s", n_resume, exp_id, name, platform_tag,
                run_dir.name, git.get("commit"),
                " (commit differs from the original run, override recorded)"
                if original_commit != git.get("commit") else "")
    logger.info("device: nvml index %s, %s, uuid %s, resolved by %s, uuid_verified=%s",
                dev.index, dev.name, dev.uuid, dev.resolved.method, dev.resolved.uuid_verified)
    env = snapshot_environment(device_state={"resolved": dev.resolved.to_dict(), "state": dev.state()},
                               repo_root=str(repo_root))
    write_json(run_dir / f"environment.resume{n_resume}.json", env)
    manifest.setdefault("resumes", []).append({
        "n": n_resume, "started_utc": utc_now_iso(), "commit": git.get("commit"),
        "dirty": git.get("dirty"), "commit_differs": original_commit != git.get("commit"),
        "previous_status": manifest.get("status"), "previous_error": manifest.get("error"),
    })
    manifest["status"] = "running"
    manifest.pop("error", None)
    manifest["monotonic_start"] = time.perf_counter()   # wall_time_s of this session, not a stale value
    write_json(manifest_path, manifest)

    ctx = RunContext(exp_id=exp_id, name=name, platform_tag=platform_tag, run_id=run_dir.name,
                     out_dir=run_dir, config=config, dev=dev, sync_fn=make_cuda_sync(torch_device),
                     logger=logger, quick=quick, quick_factor=quick_factor, torch_device=torch_device,
                     seed=seed, rng=rng, resume=True)
    summary = _execute(module_name, ctx, manifest, run_dir, logger)
    return 0 if quick or summary is None or summary.get("gate_pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
