"""Shared fixtures.

``fake_nvml`` swaps the ``pynvml`` binding inside ``tomlml.measure.nvml`` for
an in-memory fake with two devices, so device resolution, capability probing,
and energy windows are exercised without a GPU. The fake energy counter
advances with wall time at the device's configured power, which lets the
counter-versus-integral consistency check be tested for real.
"""

from __future__ import annotations

import time
import types
from typing import Any, Dict, List

import pytest


class NVMLError(Exception):
    def __init__(self, value: int = 1):
        super().__init__(f"fake NVML error {value}")
        self.value = value


class NVMLError_NotSupported(NVMLError):
    pass


class NVMLError_NoPermission(NVMLError):
    pass


class _Handle:
    def __init__(self, idx: int):
        self.idx = idx


class _Pci:
    def __init__(self, bus: str):
        self.busId = bus


class _Mem:
    def __init__(self, total: int):
        self.total = total
        self.used = 0
        self.free = total


class _Util:
    gpu = 3
    memory = 1


class _Sample:
    def __init__(self, ts: int, mw: int):
        self.timeStamp = ts
        self.sampleValue = types.SimpleNamespace(uiVal=mw)


class Proc:
    def __init__(self, pid: int, mem: int, kind: str = "compute"):
        self.pid = pid
        self.usedGpuMemory = mem
        self.kind = kind


DEVICES: List[Dict[str, Any]] = [
    {"uuid": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "name": "NVIDIA A100-SXM4-40GB",
     "bus": "00000000:07:00.0", "power_w": 55.0, "temp": 33, "sm": 1410, "mem": 1215},
    {"uuid": "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "name": "NVIDIA A100-SXM4-40GB",
     "bus": "00000000:0f:00.0", "power_w": 60.0, "temp": 35, "sm": 1410, "mem": 1215},
]


class FakePynvml:
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_GRAPHICS = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_TOTAL_POWER_SAMPLES = 0
    nvmlClocksThrottleReasonGpuIdle = 0x1
    nvmlClocksThrottleReasonApplicationsClocksSetting = 0x2
    nvmlClocksThrottleReasonSwPowerCap = 0x4
    nvmlClocksThrottleReasonHwSlowdown = 0x8
    nvmlClocksThrottleReasonNone = 0x0
    nvmlClocksThrottleReasonAll = 0xF
    NVMLError = NVMLError
    NVMLError_NotSupported = NVMLError_NotSupported
    NVMLError_NoPermission = NVMLError_NoPermission

    def __init__(self, devices: List[Dict[str, Any]], energy_supported: bool = True,
                 throttle_supported: bool = True, samples_supported: bool = True):
        self.devices = devices
        self.energy_supported = energy_supported
        self.throttle_supported = throttle_supported
        self.samples_supported = samples_supported
        self.procs: Dict[int, List[Proc]] = {}
        self.init_calls = 0
        self.shutdown_calls = 0
        self.energy_base_mj = 1_000_000
        self.throttle_mask = 0
        self.set_power_limit_calls: List[int] = []
        self.management_limit_mw = None  # None -> NotSupported
        self._t0 = time.perf_counter()

    # lifecycle / system
    def nvmlInit(self) -> None:
        self.init_calls += 1

    def nvmlShutdown(self) -> None:
        self.shutdown_calls += 1

    def nvmlSystemGetDriverVersion(self) -> str:
        return "595.79"

    def nvmlSystemGetNVMLVersion(self) -> bytes:
        return b"13.595.45"  # bytes on purpose, to exercise decoding

    def nvmlSystemGetCudaDriverVersion_v2(self) -> int:
        return 13000

    # enumeration
    def nvmlDeviceGetCount(self) -> int:
        return len(self.devices)

    def nvmlDeviceGetHandleByIndex(self, i: int) -> _Handle:
        if self.devices[i].get("inaccessible"):
            raise NVMLError_NoPermission()
        return _Handle(i)

    def _d(self, h: _Handle) -> Dict[str, Any]:
        return self.devices[h.idx]

    def nvmlDeviceGetUUID(self, h: _Handle) -> str:
        return self._d(h)["uuid"]

    def nvmlDeviceGetName(self, h: _Handle) -> str:
        return self._d(h)["name"]

    def nvmlDeviceGetPciInfo(self, h: _Handle) -> _Pci:
        return _Pci(self._d(h)["bus"])

    # hot path
    def nvmlDeviceGetTotalEnergyConsumption(self, h: _Handle) -> int:
        if not self.energy_supported:
            raise NVMLError_NotSupported()
        elapsed = time.perf_counter() - self._t0
        return int(self.energy_base_mj + self._d(h)["power_w"] * 1000.0 * elapsed)

    def nvmlDeviceGetPowerUsage(self, h: _Handle) -> int:
        return int(self._d(h)["power_w"] * 1000)

    def nvmlDeviceGetTemperature(self, h: _Handle, kind: int) -> int:
        return self._d(h)["temp"]

    def nvmlDeviceGetClockInfo(self, h: _Handle, kind: int) -> int:
        return self._d(h)["sm"] if kind == self.NVML_CLOCK_SM else self._d(h)["mem"]

    def nvmlDeviceGetMaxClockInfo(self, h: _Handle, kind: int) -> int:
        return 3105 if kind == self.NVML_CLOCK_SM else 10501

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, h: _Handle) -> int:
        if not self.throttle_supported:
            raise NVMLError_NotSupported()
        return self.throttle_mask

    # probes
    def nvmlDeviceGetApplicationsClock(self, h: _Handle, kind: int) -> int:
        raise NVMLError_NotSupported()

    def nvmlDeviceGetPowerManagementLimit(self, h: _Handle) -> int:
        if self.management_limit_mw is None:
            raise NVMLError_NotSupported()
        return self.management_limit_mw

    def nvmlDeviceGetEnforcedPowerLimit(self, h: _Handle) -> int:
        return 150_000

    def nvmlDeviceGetPowerManagementDefaultLimit(self, h: _Handle) -> int:
        return 150_000

    def nvmlDeviceGetPowerManagementLimitConstraints(self, h: _Handle):
        return (5_000, 175_000)

    def nvmlDeviceGetPersistenceMode(self, h: _Handle) -> int:
        raise NVMLError_NotSupported()

    def nvmlDeviceGetComputeMode(self, h: _Handle) -> int:
        return 0

    def nvmlDeviceGetMemoryInfo(self, h: _Handle) -> _Mem:
        return _Mem(40 * 1024 ** 3)

    def nvmlDeviceGetUtilizationRates(self, h: _Handle) -> _Util:
        return _Util()

    def nvmlDeviceGetComputeRunningProcesses(self, h: _Handle) -> List[Proc]:
        return [p for p in self.procs.get(h.idx, []) if p.kind == "compute"]

    def nvmlDeviceGetGraphicsRunningProcesses(self, h: _Handle) -> List[Proc]:
        return [p for p in self.procs.get(h.idx, []) if p.kind == "graphics"]

    def nvmlDeviceGetSamples(self, h: _Handle, stype: int, last_ts: int):
        if not self.samples_supported:
            raise NVMLError_NotSupported()
        return (1, [_Sample(1_000_000 + i * 100_000, 150_000) for i in range(10)])

    def nvmlDeviceSetPowerManagementLimit(self, h: _Handle, mw: int) -> None:
        self.set_power_limit_calls.append(mw)
        raise NVMLError_NoPermission()


TORCH_IDENTITY_DEVICE1 = {
    "available": True, "index": 0, "name": DEVICES[1]["name"],
    "uuid": DEVICES[1]["uuid"].lower(), "pci_bus_id": DEVICES[1]["bus"],
    "torch_version": "fake", "cuda_visible_devices": None,
}


@pytest.fixture
def fake_nvml(monkeypatch):
    """Install the fake binding. torch device 0 is pretended to be physical
    device 1, so index-based resolution would pick the wrong GPU."""
    import tomlml.measure.nvml as nv

    fake = FakePynvml([dict(d) for d in DEVICES])
    monkeypatch.setattr(nv, "pynvml", fake)
    monkeypatch.setattr(nv, "HAS_NVML", True)
    monkeypatch.setattr(nv, "_init_depth", 0)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("TOMLML_NVML_UUID", raising=False)
    monkeypatch.setattr(nv, "torch_device_identity",
                        lambda index=0: dict(TORCH_IDENTITY_DEVICE1, index=index))
    return fake


@pytest.fixture
def fake_device(fake_nvml):
    """An opened NvmlDevice bound to the fake binding (resolves to device 1)."""
    from tomlml.measure.nvml import open_device, nvml_shutdown

    dev = open_device(torch_index=0)
    yield dev
    nvml_shutdown()
