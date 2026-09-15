"""NVML access for the tomlml measurement stack.

Everything in this module is read-only with respect to device state, with one
documented exception (``probe_power_limit_write_permission``) that is off by
default and writes the current management limit back to itself.

Design decisions are recorded in DECISIONS.md (D-003):

* The torch CUDA device is resolved to its physical GPU by UUID, so that
  measurements taken under ``CUDA_VISIBLE_DEVICES`` remapping (for example
  inside a SLURM allocation) attach to the right sensor. The resolution method
  and a UUID verification flag are recorded for every run.
* The cumulative energy counter (``nvmlDeviceGetTotalEnergyConsumption``) is
  the primary energy quantity. Power sampling is a consistency check on the
  same sensor, not an independent measurement.

The module tolerates two generations of bindings: the NVIDIA maintained
``nvidia-ml-py`` package and the legacy ``pynvml`` shim, whose older versions
returned bytes where the current one returns str.
"""

from __future__ import annotations

import os
import statistics
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import pynvml  # module provided by nvidia-ml-py; legacy pynvml is a shim
        HAS_NVML = True
    except ImportError:  # pragma: no cover - only without the binding
        pynvml = None  # type: ignore[assignment]
        HAS_NVML = False


class NvmlUnavailable(RuntimeError):
    """Raised when the NVML binding or library cannot be used."""


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #

def _s(value: Any) -> Any:
    """Decode bytes returned by older bindings; pass everything else through."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _norm_uuid(value: Optional[str]) -> Optional[str]:
    """Normalize a GPU UUID to lowercase with the ``gpu-`` prefix."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None
    if not v.startswith("gpu-"):
        v = "gpu-" + v
    return v


def nvml_error_name(err: BaseException) -> str:
    """Short name of an NVML error class, e.g. ``NotSupported``."""
    name = type(err).__name__
    if name.startswith("NVMLError_"):
        return name[len("NVMLError_"):]
    return name


def _call(name: str, *args: Any, **kwargs: Any) -> Tuple[Any, Optional[str]]:
    """Call ``pynvml.<name>`` and return ``(value, None)`` or ``(None, error)``.

    Missing symbols in older bindings and signature mismatches are reported as
    ``BindingError:...`` rather than raised, so capability probes degrade
    gracefully and record what was unavailable.
    """
    fn = getattr(pynvml, name, None)
    if fn is None:
        return None, "BindingError:missing_symbol"
    try:
        return fn(*args, **kwargs), None
    except pynvml.NVMLError as err:  # type: ignore[union-attr]
        return None, nvml_error_name(err)
    except (AttributeError, TypeError, ValueError) as err:
        return None, f"BindingError:{type(err).__name__}"


def _is_not_supported(err: BaseException) -> bool:
    return nvml_error_name(err) == "NotSupported"


# --------------------------------------------------------------------------- #
# Library lifecycle (reference counted so nested users can share one init)
# --------------------------------------------------------------------------- #

_init_depth = 0


def nvml_init() -> None:
    global _init_depth
    if not HAS_NVML:
        raise NvmlUnavailable("nvidia-ml-py is not installed (pip install nvidia-ml-py)")
    if _init_depth == 0:
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as err:
            raise NvmlUnavailable(f"nvmlInit failed: {nvml_error_name(err)}") from err
    _init_depth += 1


def nvml_shutdown() -> None:
    global _init_depth
    if _init_depth <= 0:
        return
    _init_depth -= 1
    if _init_depth == 0:
        _call("nvmlShutdown")


def system_info() -> Dict[str, Any]:
    """Driver, NVML, and CUDA driver versions as reported by NVML."""
    driver, e_driver = _call("nvmlSystemGetDriverVersion")
    nvml_version, e_nvml = _call("nvmlSystemGetNVMLVersion")
    cuda, e_cuda = _call("nvmlSystemGetCudaDriverVersion_v2")
    if e_cuda:
        cuda, e_cuda = _call("nvmlSystemGetCudaDriverVersion")
    cuda_str = None
    if isinstance(cuda, int):
        cuda_str = f"{cuda // 1000}.{(cuda % 1000) // 10}"
    return {
        "driver_version": _s(driver) if driver is not None else None,
        "driver_version_error": e_driver,
        "nvml_version": _s(nvml_version) if nvml_version is not None else None,
        "nvml_version_error": e_nvml,
        "cuda_driver_version": cuda if isinstance(cuda, int) else None,
        "cuda_driver_version_str": cuda_str,
        "cuda_driver_version_error": e_cuda,
    }


# --------------------------------------------------------------------------- #
# Device enumeration and resolution
# --------------------------------------------------------------------------- #

def _public_fields(obj: Any) -> Dict[str, Any]:
    """Dataclass fields as a plain dict, skipping the NVML handle.

    ``dataclasses.asdict`` is not used because it deep-copies every field and
    the real handle is a ctypes pointer, which cannot be copied.
    """
    out: Dict[str, Any] = {}
    for f in fields(obj):
        if f.name == "handle":
            continue
        value = getattr(obj, f.name)
        out[f.name] = list(value) if isinstance(value, list) else value
    return out


@dataclass
class DeviceInfo:
    nvml_index: int
    uuid: Optional[str]
    name: Optional[str]
    pci_bus_id: Optional[str]
    error: Optional[str] = None
    handle: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return _public_fields(self)


def list_devices() -> List[DeviceInfo]:
    """Enumerate NVML devices. Devices whose handle cannot be acquired (for
    example, GPUs outside the current cgroup allocation) are listed with an
    error and no handle rather than aborting the enumeration."""
    count, err = _call("nvmlDeviceGetCount")
    if err or count is None:
        return []
    out: List[DeviceInfo] = []
    for i in range(int(count)):
        handle, err = _call("nvmlDeviceGetHandleByIndex", i)
        if err:
            out.append(DeviceInfo(i, None, None, None, error=err))
            continue
        uuid, _ = _call("nvmlDeviceGetUUID", handle)
        name, _ = _call("nvmlDeviceGetName", handle)
        pci, _ = _call("nvmlDeviceGetPciInfo", handle)
        bus = _s(getattr(pci, "busId", None)) if pci is not None else None
        out.append(DeviceInfo(
            nvml_index=i,
            uuid=_norm_uuid(_s(uuid)) if uuid is not None else None,
            name=_s(name) if name is not None else None,
            pci_bus_id=bus.lower() if isinstance(bus, str) else None,
            handle=handle,
        ))
    return out


def torch_device_identity(index: int = 0) -> Dict[str, Any]:
    """UUID, name, and PCI bus id of a torch CUDA device, if torch is present."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda.is_available() is False"}
    props = torch.cuda.get_device_properties(index)
    uuid = getattr(props, "uuid", None)
    bus = None
    if all(hasattr(props, a) for a in ("pci_domain_id", "pci_bus_id", "pci_device_id")):
        bus = f"{props.pci_domain_id:08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    return {
        "available": True,
        "index": index,
        "name": props.name,
        "uuid": _norm_uuid(str(uuid)) if uuid is not None else None,
        "pci_bus_id": bus,
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


@dataclass
class ResolvedDevice:
    nvml_index: int
    uuid: Optional[str]
    name: Optional[str]
    pci_bus_id: Optional[str]
    method: str
    torch_index: Optional[int]
    torch_uuid: Optional[str]
    torch_name: Optional[str]
    uuid_verified: Optional[bool]
    notes: List[str] = field(default_factory=list)
    handle: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return _public_fields(self)


def resolve_device(
    torch_index: int = 0,
    override_uuid: Optional[str] = None,
    torch_identity: Optional[Dict[str, Any]] = None,
) -> ResolvedDevice:
    """Map a torch CUDA device index to the physical NVML device.

    Resolution order, first hit wins, and the method used is recorded:

    1. ``override_uuid`` argument or ``TOMLML_NVML_UUID`` environment variable
    2. torch device UUID (torch >= 2.1 exposes it)
    3. torch PCI bus id
    4. ``CUDA_VISIBLE_DEVICES`` entry at ``torch_index`` (integer or UUID form)
    5. NVML index equal to ``torch_index``
    6. first accessible NVML device (noted as a fallback)

    ``uuid_verified`` is True when the chosen NVML device's UUID equals the
    torch device's UUID, False on a mismatch, and None when torch did not
    report a UUID.
    """
    devices = [d for d in list_devices() if d.error is None]
    if not devices:
        raise NvmlUnavailable("NVML enumerated no accessible devices")
    ident = torch_identity if torch_identity is not None else torch_device_identity(torch_index)
    notes: List[str] = []
    chosen: Optional[DeviceInfo] = None
    method = ""

    override = override_uuid or os.environ.get("TOMLML_NVML_UUID") or None
    if override:
        ov = _norm_uuid(override)
        chosen = next((d for d in devices if d.uuid == ov), None)
        method = "override_uuid"
        if chosen is None:
            notes.append(f"override uuid {override} not found among NVML devices")

    if chosen is None and ident.get("available") and ident.get("uuid"):
        chosen = next((d for d in devices if d.uuid == ident["uuid"]), None)
        method = "torch_uuid"
        if chosen is None:
            notes.append("torch device uuid not found among NVML devices")

    if chosen is None and ident.get("available") and ident.get("pci_bus_id"):
        chosen = next((d for d in devices if d.pci_bus_id == ident["pci_bus_id"]), None)
        method = "torch_pci_bus_id"

    if chosen is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            parts = [p.strip() for p in cvd.split(",") if p.strip()]
            if torch_index < len(parts):
                part = parts[torch_index]
                if part.isdigit():
                    chosen = next((d for d in devices if d.nvml_index == int(part)), None)
                    method = "cuda_visible_devices_index"
                else:
                    pn = _norm_uuid(part)
                    chosen = next(
                        (d for d in devices if d.uuid and (d.uuid == pn or d.uuid.startswith(pn or "\0"))),
                        None,
                    )
                    method = "cuda_visible_devices_uuid"
            if chosen is None:
                notes.append(f"CUDA_VISIBLE_DEVICES={cvd!r} did not resolve torch index {torch_index}")

    if chosen is None:
        chosen = next((d for d in devices if d.nvml_index == torch_index), None)
        method = "index"

    if chosen is None:
        chosen = devices[0]
        method = "first_accessible"
        notes.append("fell back to the first accessible NVML device")

    verified: Optional[bool] = None
    if ident.get("available") and ident.get("uuid") and chosen.uuid:
        verified = ident["uuid"] == chosen.uuid
        if not verified:
            notes.append("UUID MISMATCH between the torch device and the NVML device")

    return ResolvedDevice(
        nvml_index=chosen.nvml_index,
        uuid=chosen.uuid,
        name=chosen.name,
        pci_bus_id=chosen.pci_bus_id,
        method=method,
        torch_index=ident.get("index") if ident.get("available") else None,
        torch_uuid=ident.get("uuid") if ident.get("available") else None,
        torch_name=ident.get("name") if ident.get("available") else None,
        uuid_verified=verified,
        notes=notes,
        handle=chosen.handle,
    )


# --------------------------------------------------------------------------- #
# Throttle reason decoding
# --------------------------------------------------------------------------- #

_THROTTLE_PREFIXES = ("nvmlClocksEventReason", "nvmlClocksThrottleReason")


def throttle_reason_table() -> Dict[int, str]:
    """Bit -> name table built from whichever constants the binding exposes."""
    table: Dict[int, str] = {}
    for prefix in _THROTTLE_PREFIXES:
        for attr in dir(pynvml):
            if not attr.startswith(prefix):
                continue
            name = attr[len(prefix):]
            if name in ("None", "All", ""):
                continue
            val = getattr(pynvml, attr)
            if isinstance(val, int) and val > 0 and val not in table:
                table[val] = name
    return table


def decode_throttle_reasons(mask: Optional[int]) -> List[str]:
    if not mask:
        return []
    table = throttle_reason_table()
    names = [name for bit, name in sorted(table.items()) if mask & bit]
    known = 0
    for bit in table:
        known |= bit
    unknown = mask & ~known
    if unknown:
        names.append(f"unknown_bits:0x{unknown:x}")
    return names


# --------------------------------------------------------------------------- #
# Device wrapper
# --------------------------------------------------------------------------- #

class NvmlDevice:
    """Thin wrapper around an NVML handle with typed, unit-converted reads.

    Hot-path reads (``energy_mj``, ``power_w``, ``temperature_c``, clocks,
    ``throttle_mask``) raise ``pynvml.NVMLError`` subclasses on failure so the
    sampler can decide per field whether to disable or retry. Probe-style
    methods never raise.
    """

    def __init__(self, resolved: ResolvedDevice):
        self.resolved = resolved
        self.handle = resolved.handle
        self.index = resolved.nvml_index
        self.uuid = resolved.uuid
        self.name = resolved.name

    # -- hot path ----------------------------------------------------------- #

    def energy_mj(self) -> int:
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def power_w(self) -> float:
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def temperature_c(self) -> int:
        return int(pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU))

    def sm_clock_mhz(self) -> int:
        return int(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM))

    def mem_clock_mhz(self) -> int:
        return int(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM))

    def throttle_mask(self) -> int:
        last_err: Optional[BaseException] = None
        for api in ("nvmlDeviceGetCurrentClocksEventReasons",
                    "nvmlDeviceGetCurrentClocksThrottleReasons"):
            fn = getattr(pynvml, api, None)
            if fn is None:
                continue
            try:
                return int(fn(self.handle))
            except pynvml.NVMLError as err:
                last_err = err
                if _is_not_supported(err):
                    continue
                raise
        if last_err is not None:
            raise last_err
        raise NvmlUnavailable("no throttle-reason API in this binding")

    # -- probes (never raise) ---------------------------------------------- #

    def running_processes(self, exclude_self: bool = True) -> List[Dict[str, Any]]:
        """Compute and graphics processes currently on this GPU."""
        procs: List[Dict[str, Any]] = []
        me = os.getpid()
        for api, kind in (("nvmlDeviceGetComputeRunningProcesses", "compute"),
                          ("nvmlDeviceGetGraphicsRunningProcesses", "graphics")):
            items, err = _call(api, self.handle)
            if err or not items:
                continue
            for p in items:
                pid = int(getattr(p, "pid", -1))
                if exclude_self and pid == me:
                    continue
                mem = getattr(p, "usedGpuMemory", None)
                procs.append({
                    "pid": pid,
                    "kind": kind,
                    "used_gpu_memory_bytes": int(mem) if isinstance(mem, int) else None,
                })
        return procs

    def state(self) -> Dict[str, Any]:
        """One tolerant snapshot of the operating state (for logs, not loops)."""
        out: Dict[str, Any] = {}
        for key, fn in (("power_w", self.power_w), ("temperature_c", self.temperature_c),
                        ("sm_clock_mhz", self.sm_clock_mhz), ("mem_clock_mhz", self.mem_clock_mhz),
                        ("energy_mj", self.energy_mj), ("throttle_mask", self.throttle_mask)):
            try:
                out[key] = fn()
            except Exception as err:  # noqa: BLE001 - probe must not raise
                out[key] = None
                out[key + "_error"] = nvml_error_name(err)
        out["throttle_reasons"] = decode_throttle_reasons(out.get("throttle_mask"))
        return out


def open_device(torch_index: int = 0, override_uuid: Optional[str] = None) -> NvmlDevice:
    """Initialize NVML (reference counted) and resolve the device.

    Callers pair this with ``nvml_shutdown()`` when finished.
    """
    nvml_init()
    return NvmlDevice(resolve_device(torch_index=torch_index, override_uuid=override_uuid))


# --------------------------------------------------------------------------- #
# Capability probing (G0 instrumentation gate)
# --------------------------------------------------------------------------- #

_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "TOMLML_NVML_UUID",
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
    "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_CPUS_PER_TASK",
    "HOSTNAME", "COMPUTERNAME",
)


def probe_power_samples(dev: NvmlDevice) -> Dict[str, Any]:
    """Read the sensor's own buffered power samples to estimate its native
    update period. Not all products expose this buffer."""
    stype = getattr(pynvml, "NVML_TOTAL_POWER_SAMPLES", None)
    if stype is None:
        return {"supported": False, "error": "BindingError:missing_symbol"}
    res, err = _call("nvmlDeviceGetSamples", dev.handle, stype, 0)
    if err:
        return {"supported": False, "error": err}
    try:
        _, samples = res
        ts = sorted(int(s.timeStamp) for s in samples)
    except Exception as err:  # noqa: BLE001
        return {"supported": False, "error": f"ParseError:{type(err).__name__}"}
    diffs = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    return {
        "supported": True,
        "n_samples": len(ts),
        "native_period_ms_median": statistics.median(diffs) / 1000.0 if diffs else None,
        "native_period_ms_min": min(diffs) / 1000.0 if diffs else None,
        "native_period_ms_max": max(diffs) / 1000.0 if diffs else None,
        "span_s": (ts[-1] - ts[0]) / 1e6 if len(ts) > 1 else 0.0,
    }


def probe_capabilities(dev: NvmlDevice, include_samples_api: bool = True) -> Dict[str, Any]:
    """Record what this device and binding support. Every field is either a
    value or an error name; nothing raises."""
    h = dev.handle
    cap: Dict[str, Any] = {"system": system_info(), "device": dev.resolved.to_dict()}

    def put(key: str, api: str, *args: Any, transform: Optional[Callable[[Any], Any]] = None) -> None:
        val, err = _call(api, h, *args)
        if err is None and transform is not None and val is not None:
            try:
                val = transform(val)
            except Exception as ex:  # noqa: BLE001
                val, err = None, f"TransformError:{type(ex).__name__}"
        cap[key] = {"value": val, "error": err}

    mw = lambda v: v / 1000.0  # noqa: E731
    put("energy_counter_mj", "nvmlDeviceGetTotalEnergyConsumption", transform=int)
    put("power_w", "nvmlDeviceGetPowerUsage", transform=mw)
    put("power_limit_management_w", "nvmlDeviceGetPowerManagementLimit", transform=mw)
    put("power_limit_enforced_w", "nvmlDeviceGetEnforcedPowerLimit", transform=mw)
    put("power_limit_default_w", "nvmlDeviceGetPowerManagementDefaultLimit", transform=mw)
    put("power_limit_constraints_w", "nvmlDeviceGetPowerManagementLimitConstraints",
        transform=lambda v: [v[0] / 1000.0, v[1] / 1000.0])
    put("temperature_c", "nvmlDeviceGetTemperature", pynvml.NVML_TEMPERATURE_GPU, transform=int)
    put("sm_clock_mhz", "nvmlDeviceGetClockInfo", pynvml.NVML_CLOCK_SM, transform=int)
    put("mem_clock_mhz", "nvmlDeviceGetClockInfo", pynvml.NVML_CLOCK_MEM, transform=int)
    put("sm_clock_max_mhz", "nvmlDeviceGetMaxClockInfo", pynvml.NVML_CLOCK_SM, transform=int)
    put("mem_clock_max_mhz", "nvmlDeviceGetMaxClockInfo", pynvml.NVML_CLOCK_MEM, transform=int)
    put("applications_clock_sm_mhz", "nvmlDeviceGetApplicationsClock", pynvml.NVML_CLOCK_SM, transform=int)
    put("applications_clock_mem_mhz", "nvmlDeviceGetApplicationsClock", pynvml.NVML_CLOCK_MEM, transform=int)
    put("persistence_mode", "nvmlDeviceGetPersistenceMode", transform=int)
    put("compute_mode", "nvmlDeviceGetComputeMode", transform=int)
    put("memory_total_bytes", "nvmlDeviceGetMemoryInfo", transform=lambda m: int(m.total))
    put("utilization", "nvmlDeviceGetUtilizationRates",
        transform=lambda u: {"gpu": int(u.gpu), "memory": int(u.memory)})

    throttle: Dict[str, Any] = {"api": None, "mask": None, "reasons": [], "error": None}
    for api in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
        val, err = _call(api, h)
        if err is None:
            throttle.update(api=api, mask=int(val), reasons=decode_throttle_reasons(int(val)), error=None)
            break
        throttle["error"] = err
    cap["throttle"] = throttle
    cap["running_processes"] = dev.running_processes(exclude_self=True)
    if include_samples_api:
        cap["power_samples_api"] = probe_power_samples(dev)
    cap["environment"] = {k: os.environ.get(k) for k in _ENV_KEYS}
    return cap


def probe_power_limit_write_permission(dev: NvmlDevice) -> Dict[str, Any]:
    """Write the current management power limit back to itself.

    This is the only state write in the module. It is off by default in every
    config (DECISIONS.md D-003) and exists so the G0 gate can record whether
    power-cap control would be available without changing anything.
    """
    limit_mw, err = _call("nvmlDeviceGetPowerManagementLimit", dev.handle)
    if err or limit_mw is None:
        return {"probed": False, "reason": err or "no_limit"}
    _, err2 = _call("nvmlDeviceSetPowerManagementLimit", dev.handle, int(limit_mw))
    return {"probed": True, "permitted": err2 is None, "error": err2,
            "limit_mw_written_back": int(limit_mw)}
