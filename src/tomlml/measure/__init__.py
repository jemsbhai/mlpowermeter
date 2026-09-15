"""Measurement: NVML access and energy windows."""

from .nvml import (
    HAS_NVML,
    DeviceInfo,
    NvmlDevice,
    NvmlUnavailable,
    ResolvedDevice,
    decode_throttle_reasons,
    list_devices,
    nvml_init,
    nvml_shutdown,
    open_device,
    probe_capabilities,
    probe_power_limit_write_permission,
    probe_power_samples,
    resolve_device,
    system_info,
    torch_device_identity,
)
from .meter import (
    EnergyMeter,
    MeasurementWindow,
    StateSample,
    StateSampler,
    integrate_power,
    make_cuda_sync,
    measure_idle,
    no_sync,
    summarize_window,
    wait_for_thermal_settle,
    write_samples_csv,
)

__all__ = [
    "HAS_NVML", "DeviceInfo", "NvmlDevice", "NvmlUnavailable", "ResolvedDevice",
    "decode_throttle_reasons", "list_devices", "nvml_init", "nvml_shutdown",
    "open_device", "probe_capabilities", "probe_power_limit_write_permission",
    "probe_power_samples", "resolve_device", "system_info", "torch_device_identity",
    "EnergyMeter", "MeasurementWindow", "StateSample", "StateSampler",
    "integrate_power", "make_cuda_sync", "measure_idle", "no_sync",
    "summarize_window", "wait_for_thermal_settle", "write_samples_csv",
]
