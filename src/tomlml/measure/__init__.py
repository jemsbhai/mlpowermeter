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
    ENERGY_SOURCES,
    EnergyMeter,
    MeasurementWindow,
    StateSample,
    StateSampler,
    integrate_power,
    make_cuda_sync,
    mask_implausible_power,
    measure_idle,
    no_sync,
    summarize_window,
    wait_for_thermal_settle,
    write_samples_csv,
)
from .protocol import (
    ProtocolSettings,
    measure_window,
    plausibility_ceiling,
    regime_label,
    settle_under_load,
)
from .census import classify_command, command_census, count_commands_from_events

__all__ = [
    "HAS_NVML", "DeviceInfo", "NvmlDevice", "NvmlUnavailable", "ResolvedDevice",
    "decode_throttle_reasons", "list_devices", "nvml_init", "nvml_shutdown",
    "open_device", "probe_capabilities", "probe_power_limit_write_permission",
    "probe_power_samples", "resolve_device", "system_info", "torch_device_identity",
    "EnergyMeter", "ENERGY_SOURCES", "MeasurementWindow", "StateSample", "StateSampler",
    "integrate_power", "make_cuda_sync", "mask_implausible_power", "measure_idle", "no_sync",
    "summarize_window", "wait_for_thermal_settle", "write_samples_csv",
    "ProtocolSettings", "measure_window", "plausibility_ceiling", "regime_label",
    "settle_under_load", "classify_command", "command_census", "count_commands_from_events",
]
