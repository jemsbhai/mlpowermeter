"""Environment snapshot for reproducibility (environment.json).

Captures software versions, hardware identity, and the scheduler context
without requiring anything beyond the standard library, torch, and NVML.
Every field is best effort: a missing source records None, never an exception.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
from importlib import metadata
from typing import Any, Dict, List, Optional

from .manifest import git_info, utc_now_iso, local_now_iso

PACKAGES_OF_INTEREST: List[str] = [
    "torch", "torchvision", "torchaudio", "numpy", "scipy", "scikit-learn",
    "pandas", "pyyaml", "nvidia-ml-py", "pynvml", "gymnasium", "stable-baselines3",
    "xgboost", "lightgbm", "cuml", "cupy", "triton", "tomlml",
]

ENV_KEYS: List[str] = [
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT",
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
    "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE", "SLURM_NTASKS", "TOMLML_PLATFORM_TAG", "TOMLML_NVML_UUID",
]


def _package_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in PACKAGES_OF_INTEREST:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _cpu_model() -> Optional[str]:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Windows":
            import winreg  # type: ignore[import-not-found]
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        elif system == "Darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or None


def _ram_gb() -> Optional[float]:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = float(line.split()[1])
                        return round(kb / (1024 ** 2), 2)
        elif system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return round(stat.ullTotalPhys / (1024 ** 3), 2)
    except Exception:  # noqa: BLE001
        pass
    return None


def _torch_info() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info: Dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": p.name,
                "uuid": str(getattr(p, "uuid", None)) if getattr(p, "uuid", None) is not None else None,
                "total_memory_bytes": int(p.total_memory),
                "multi_processor_count": int(p.multi_processor_count),
                "compute_capability": f"{p.major}.{p.minor}",
            })
    return info


def _nvml_info(device_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from ..measure.nvml import HAS_NVML, nvml_init, nvml_shutdown, system_info, list_devices
    except Exception as err:  # noqa: BLE001
        return {"available": False, "error": f"{type(err).__name__}: {err}"}
    if not HAS_NVML:
        return {"available": False, "error": "nvidia-ml-py not installed"}
    try:
        nvml_init()
    except Exception as err:  # noqa: BLE001
        return {"available": False, "error": f"{type(err).__name__}: {err}"}
    try:
        out: Dict[str, Any] = {"available": True}
        out.update(system_info())
        out["devices"] = [d.to_dict() for d in list_devices()]
        if device_state is not None:
            out["target_device_state"] = device_state
        return out
    finally:
        nvml_shutdown()


def snapshot_environment(device_state: Optional[Dict[str, Any]] = None,
                         repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Assemble the environment record. ``device_state`` may carry the target
    device's resolved identity and a one-shot state read, if the caller has it."""
    uname = platform.uname()
    return {
        "timestamp_utc": utc_now_iso(),
        "timestamp_local": local_now_iso(),
        "hostname": socket.gethostname(),
        "os": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "platform": platform.platform(),
        },
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": {
            "model": _cpu_model(),
            "logical_cores": os.cpu_count(),
        },
        "ram_gb": _ram_gb(),
        "torch": _torch_info(),
        "nvml": _nvml_info(device_state),
        "packages": _package_versions(),
        "environment_variables": {k: os.environ.get(k) for k in ENV_KEYS},
        "git": git_info(repo_root),
    }
