"""Tests for tomlml.measure.nvml against the fake binding (no GPU needed)."""

from __future__ import annotations

import os

import pytest

from tests.conftest import DEVICES, Proc


def test_init_is_reference_counted(fake_nvml):
    from tomlml.measure import nvml as nv

    nv.nvml_init()
    nv.nvml_init()
    assert fake_nvml.init_calls == 1
    nv.nvml_shutdown()
    assert fake_nvml.shutdown_calls == 0
    nv.nvml_shutdown()
    assert fake_nvml.shutdown_calls == 1
    nv.nvml_shutdown()  # extra shutdown is a no-op
    assert fake_nvml.shutdown_calls == 1


def test_system_info_decodes_bytes_and_cuda_version(fake_nvml):
    from tomlml.measure.nvml import system_info

    info = system_info()
    assert info["driver_version"] == "595.79"
    assert info["nvml_version"] == "13.595.45"
    assert info["cuda_driver_version"] == 13000
    assert info["cuda_driver_version_str"] == "13.0"


def test_list_devices_normalizes_uuid_and_bus(fake_nvml):
    from tomlml.measure.nvml import list_devices

    devs = list_devices()
    assert [d.nvml_index for d in devs] == [0, 1]
    assert devs[0].uuid == DEVICES[0]["uuid"].lower()
    assert devs[1].pci_bus_id == DEVICES[1]["bus"]
    assert all(d.error is None for d in devs)


def test_resolve_prefers_torch_uuid_over_index(fake_nvml):
    from tomlml.measure.nvml import resolve_device

    r = resolve_device(torch_index=0)
    assert r.nvml_index == 1
    assert r.method == "torch_uuid"
    assert r.uuid_verified is True
    assert r.notes == []


def test_resolve_override_uuid_flags_mismatch(fake_nvml, monkeypatch):
    from tomlml.measure.nvml import resolve_device

    monkeypatch.setenv("TOMLML_NVML_UUID", DEVICES[0]["uuid"])
    r = resolve_device(torch_index=0)
    assert r.nvml_index == 0
    assert r.method == "override_uuid"
    assert r.uuid_verified is False
    assert any("MISMATCH" in n for n in r.notes)


def test_resolve_uses_cuda_visible_devices_without_torch(fake_nvml, monkeypatch):
    from tomlml.measure import nvml as nv

    monkeypatch.setattr(nv, "torch_device_identity", lambda index=0: {"available": False})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    r = nv.resolve_device(torch_index=0)
    assert r.nvml_index == 1
    assert r.method == "cuda_visible_devices_index"
    assert r.uuid_verified is None

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbbbbbb")
    r = nv.resolve_device(torch_index=0)
    assert r.nvml_index == 1
    assert r.method == "cuda_visible_devices_uuid"


def test_resolve_skips_inaccessible_devices(fake_nvml, monkeypatch):
    from tomlml.measure import nvml as nv

    fake_nvml.devices[0]["inaccessible"] = True
    monkeypatch.setattr(nv, "torch_device_identity", lambda index=0: {"available": False})
    r = nv.resolve_device(torch_index=0)
    assert r.nvml_index == 1
    assert r.method == "first_accessible"
    assert any("first accessible" in n for n in r.notes)


def test_resolve_raises_when_nothing_accessible(fake_nvml):
    from tomlml.measure.nvml import NvmlUnavailable, resolve_device

    for d in fake_nvml.devices:
        d["inaccessible"] = True
    with pytest.raises(NvmlUnavailable):
        resolve_device()


def test_decode_throttle_reasons(fake_nvml):
    from tomlml.measure.nvml import decode_throttle_reasons

    assert decode_throttle_reasons(0) == []
    assert decode_throttle_reasons(None) == []
    assert decode_throttle_reasons(0x5) == ["GpuIdle", "SwPowerCap"]
    assert decode_throttle_reasons(0x10) == ["unknown_bits:0x10"]


def test_device_hot_path_reads(fake_device):
    assert fake_device.index == 1
    assert fake_device.power_w() == pytest.approx(60.0)
    assert fake_device.temperature_c() == 35
    assert fake_device.sm_clock_mhz() == 1410
    assert fake_device.mem_clock_mhz() == 1215
    assert isinstance(fake_device.energy_mj(), int)
    assert fake_device.throttle_mask() == 0


def test_running_processes_excludes_self(fake_nvml, fake_device):
    fake_nvml.procs[1] = [Proc(os.getpid(), 1 << 20), Proc(4242, 2 << 20),
                          Proc(777, 1 << 10, kind="graphics")]
    procs = fake_device.running_processes()
    pids = sorted(p["pid"] for p in procs)
    assert pids == [777, 4242]
    kinds = {p["pid"]: p["kind"] for p in procs}
    assert kinds[4242] == "compute" and kinds[777] == "graphics"
    assert len(fake_device.running_processes(exclude_self=False)) == 3


def test_probe_capabilities_records_values_and_errors(fake_device):
    from tomlml.measure.nvml import probe_capabilities

    cap = probe_capabilities(fake_device)
    assert cap["device"]["nvml_index"] == 1
    assert cap["device"]["uuid_verified"] is True
    assert isinstance(cap["energy_counter_mj"]["value"], int)
    assert cap["energy_counter_mj"]["error"] is None
    assert cap["power_w"]["value"] == pytest.approx(60.0)
    assert cap["power_limit_management_w"]["value"] is None
    assert cap["power_limit_management_w"]["error"] == "NotSupported"
    assert cap["power_limit_constraints_w"]["value"] == [5.0, 175.0]
    assert cap["persistence_mode"]["error"] == "NotSupported"
    assert cap["memory_total_bytes"]["value"] == 40 * 1024 ** 3
    assert cap["throttle"]["api"] == "nvmlDeviceGetCurrentClocksThrottleReasons"
    assert cap["throttle"]["reasons"] == []
    assert cap["power_samples_api"]["supported"] is True
    assert cap["power_samples_api"]["native_period_ms_median"] == pytest.approx(100.0)
    assert cap["running_processes"] == []
    assert "CUDA_VISIBLE_DEVICES" in cap["environment"]


def test_probe_capabilities_when_samples_api_unsupported(fake_nvml, fake_device):
    from tomlml.measure.nvml import probe_capabilities

    fake_nvml.samples_supported = False
    cap = probe_capabilities(fake_device)
    assert cap["power_samples_api"] == {"supported": False, "error": "NotSupported"}


def test_state_snapshot_never_raises(fake_nvml, fake_device):
    fake_nvml.throttle_supported = False
    st = fake_device.state()
    assert st["power_w"] == pytest.approx(60.0)
    assert st["throttle_mask"] is None
    assert st["throttle_mask_error"] == "NotSupported"
    assert st["throttle_reasons"] == []


def test_write_permission_probe(fake_nvml, fake_device):
    from tomlml.measure.nvml import probe_power_limit_write_permission

    res = probe_power_limit_write_permission(fake_device)
    assert res == {"probed": False, "reason": "NotSupported"}
    assert fake_nvml.set_power_limit_calls == []

    fake_nvml.management_limit_mw = 150_000
    res = probe_power_limit_write_permission(fake_device)
    assert res["probed"] is True
    assert res["permitted"] is False
    assert res["error"] == "NoPermission"
    assert fake_nvml.set_power_limit_calls == [150_000]
