"""Run manifests, experiment directories, git provenance, and file hashing.

Every experiment run produces, under ``experiments/exp_NNN_name/<platform>/<run_id>/``:

    config.yaml        frozen copy of the config that ran (never configs/ at runtime)
    manifest.json      experiment id, run id, platform, git commit and dirty flag,
                       config hash, device resolution, timestamps
    environment.json   software and hardware snapshot (utils.envsnap)
    seed.json          all seeds (utils.seeds)
    results/           metrics and summaries (committed)
    samples/           raw timestamped sensor samples (committed if small)
    logs/              execution logs
    figures/           per-experiment plots

Run directories are never overwritten; a new run gets a new run id.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_run_id(now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    return now.strftime("%Y%m%dT%H%M%S")


def find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upwards from ``start`` (default: this file) to the first directory
    containing ``.git``. Returns None if not inside a repository."""
    p = Path(start or __file__).resolve()
    for candidate in [p] + list(p.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def git_info(repo_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Commit SHA, branch, and dirty flag. Never raises; errors are recorded."""
    root = Path(repo_root) if repo_root else find_repo_root()
    info: Dict[str, Any] = {"commit": None, "branch": None, "dirty": None,
                            "root": root.as_posix() if root else None, "error": None}
    if root is None:
        info["error"] = "not inside a git repository"
        return info

    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=str(root),
                                       stderr=subprocess.STDOUT, text=True).strip()

    try:
        info["commit"] = run("rev-parse", "HEAD")
        info["branch"] = run("rev-parse", "--abbrev-ref", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=no")
        info["dirty"] = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as err:
        info["error"] = f"{type(err).__name__}: {err}"
    return info


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def platform_tag_from_gpu_name(gpu_name: Optional[str]) -> str:
    """Canonical platform tag for a GPU name.

    Known devices map to the tags used in LOGBOOK.md; unknown devices get a
    slug of the name with vendor and brand words removed.
    """
    if not gpu_name:
        return "unknown-gpu"
    n = gpu_name.lower()
    if "4090" in n and "laptop" in n:
        return "rtx4090-laptop"
    if "a100" in n:
        mem = "80gb" if "80gb" in n else ("40gb" if "40gb" in n else "")
        form = "sxm4" if "sxm" in n else ("pcie" if "pcie" in n else "")
        return "-".join(x for x in ("a100", form, mem) if x)
    for word in ("nvidia", "geforce", "tesla", "gpu"):
        n = n.replace(word, " ")
    return slugify(n) or "unknown-gpu"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True, default=_json_default)
        f.write("\n")
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()  # platform-independent rendering in manifests
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "tolist"):  # numpy scalars and arrays
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def experiment_folder(exp_id: str, name: str) -> str:
    """``exp_NNN_<slug>`` folder name under experiments/."""
    exp_num = exp_id.upper().replace("EXP-", "").replace("EXP", "")
    return f"exp_{int(exp_num):03d}_{slugify(name)}"


def make_experiment_dir(root: Path, exp_id: str, name: str, platform_tag: str,
                        run_id: Optional[str] = None) -> Path:
    """Create ``experiments/exp_NNN_name/<platform>/<run_id>/`` with its
    subdirectories. Raises if the run directory already exists."""
    folder = experiment_folder(exp_id, name)
    run_id = run_id or new_run_id()
    out = Path(root) / "experiments" / folder / platform_tag / run_id
    if out.exists():
        raise FileExistsError(f"run directory already exists: {out}")
    for sub in ("results", "samples", "logs", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=False)
    return out


def freeze_config(config_path: Path, out_dir: Path) -> Dict[str, Any]:
    """Copy the config into the run directory and hash it."""
    src = Path(config_path)
    dst = Path(out_dir) / "config.yaml"
    shutil.copyfile(src, dst)
    return {"source": str(src), "frozen": str(dst), "sha256": sha256_file(dst)}


def build_manifest(exp_id: str, name: str, platform_tag: str, run_id: str,
                   out_dir: Path, config: Dict[str, Any], git: Dict[str, Any],
                   device: Optional[Dict[str, Any]] = None,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "experiment_id": exp_id,
        "name": name,
        "platform_tag": platform_tag,
        "run_id": run_id,
        "output_dir": Path(out_dir).as_posix(),
        "created_utc": utc_now_iso(),
        "created_local": local_now_iso(),
        "config": config,
        "git": git,
        "device": device,
        "status": "running",
        "monotonic_start": time.perf_counter(),
    }
    if extra:
        m.update(extra)
    return m


def finalize_manifest(manifest: Dict[str, Any], status: str,
                      summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    manifest["status"] = status
    manifest["finished_utc"] = utc_now_iso()
    manifest["finished_local"] = local_now_iso()
    start = manifest.pop("monotonic_start", None)
    if start is not None:
        manifest["wall_time_s"] = time.perf_counter() - start
    if summary:
        manifest["summary"] = summary
    return manifest
