"""Tests for the D-010 protocol helpers and the command census aggregation."""

from __future__ import annotations

import time
import types

import pytest

from tomlml.measure.census import classify_command, count_commands_from_events
from tomlml.measure.protocol import (
    ProtocolSettings, capped_fraction, measure_window, plausibility_ceiling, regime_label,
    settle_under_load,
)


def test_protocol_settings_from_config_and_quick_scaling():
    cfg = {"sample_interval_s": 0.05, "energy_source": "power_integral",
           "measurement": {"window_s": 20, "warmup_s": 2, "thermal_settle_timeout_s": 300,
                           "thermal_settle_window_s": 5, "census_calls": 4}}
    s = ProtocolSettings.from_config(cfg)
    assert s.window_s == 20 and s.warmup_s == 2 and s.energy_source == "power_integral"
    assert s.thermal_settle_timeout_s == 300 and s.census_calls == 4 and s.notes == []
    q = ProtocolSettings.from_config(cfg, quick_factor=0.05, power_max_plausible_w=350.0,
                                     power_limit_enforced_w=175.0)
    assert q.window_s == pytest.approx(1.0) and q.warmup_s == pytest.approx(0.1)
    assert q.thermal_settle_timeout_s == pytest.approx(15.0)
    assert q.power_max_plausible_w == 350.0 and q.power_limit_enforced_w == 175.0
    assert q.notes and "quick factor" in q.notes[0]
    assert ProtocolSettings.from_config({}).window_s == 20.0
    assert "window_s" in q.to_dict()


def test_plausibility_ceiling():
    assert plausibility_ceiling({"power_limit_constraints_w": {"value": [5.0, 175.0]}}) == 350.0
    assert plausibility_ceiling({"power_limit_constraints_w": {"value": None, "error": "NotSupported"}}) is None
    assert plausibility_ceiling({}) is None


def test_settle_under_load_runs_workload_until_stable(fake_device):
    calls = {"n": 0}

    def run_once():
        calls["n"] += 1
        time.sleep(0.001)

    s = ProtocolSettings(thermal_settle_window_s=0.15, thermal_settle_poll_s=0.05,
                         thermal_settle_timeout_s=2.0)
    res = settle_under_load(fake_device, run_once, s, sync_fn=lambda: None)
    assert res["timed_out"] is False and res["settled_temp_c"] == 35.0
    assert res["calls"] == calls["n"] > 0

    temps = iter(range(35, 500))
    fake_device.temperature_c = lambda: next(temps)  # type: ignore[assignment]
    res = settle_under_load(fake_device, run_once, s, sync_fn=lambda: None)
    assert res["timed_out"] is True and res["wait_time_s"] >= 2.0


def test_measure_window_calibrates_and_labels_regime(fake_device):
    s = ProtocolSettings(window_s=0.2, warmup_s=0.02, sample_interval_s=0.02,
                         energy_source="power_integral", power_limit_enforced_w=61.0)
    w = measure_window(fake_device, lambda: time.sleep(0.002), s, sync_fn=lambda: None, label="x",
                       check_processes=False)
    assert w.n_calls >= 50 and 0.15 <= w.duration_s <= 0.6
    assert w.energy_source == "power_integral"
    assert w.energy_per_call_j() == pytest.approx(60.0 * w.duration_s / w.n_calls, rel=0.15)
    assert regime_label(w, 61.0) == "capped"          # 60 W against a 61 W limit is within 3 percent
    assert regime_label(w, 175.0) == "uncapped"
    assert regime_label(w, None) == "uncapped"
    assert capped_fraction(w) == 0.0
    w2 = measure_window(fake_device, lambda: time.sleep(0.002), s, sync_fn=lambda: None, label="y",
                        per_call_s=0.004, check_processes=False)
    assert w2.n_calls == 50


def test_regime_label_uses_capped_sample_fraction(fake_nvml, fake_device):
    from tomlml.measure.meter import StateSample, summarize_window

    def s(t, mask):
        return StateSample(t_wall=t, t_perf=t, power_w=60.0, temp_c=30, sm_clock_mhz=1000,
                           mem_clock_mhz=1000, energy_mj=None, throttle_mask=mask)

    # one carried-over SwPowerCap (0x4) sample out of ten: not capped
    samples = [s(0.1 * i, 0x4 if i == 0 else 0x0) for i in range(10)]
    w = summarize_window("x", 0.0, 1.0, 0.0, 1.0, None, None, samples, [], [], 0.1)
    assert capped_fraction(w) == pytest.approx(0.1)
    assert "SwPowerCap" in w.throttle_reasons            # the union still records it
    assert regime_label(w, 175.0) == "uncapped"
    # majority at the cap: capped
    samples = [s(0.1 * i, 0x4 if i < 6 else 0x0) for i in range(10)]
    w = summarize_window("y", 0.0, 1.0, 0.0, 1.0, None, None, samples, [], [], 0.1)
    assert regime_label(w, 175.0) == "capped"
    # no masks at all: fall back to the power criterion
    samples = [s(0.1 * i, None) for i in range(10)]
    w = summarize_window("z", 0.0, 1.0, 0.0, 1.0, None, None, samples, [], [], 0.1)
    assert capped_fraction(w) is None
    assert regime_label(w, 61.0) == "capped" and regime_label(w, 175.0) == "uncapped"


def test_count_commands_from_events():
    cuda, cpu = "CUDA", "CPU"

    def ev(name, dt):
        return types.SimpleNamespace(name=name, device_type=dt)

    events = [ev("ampere_sgemm_128x64_nn", cuda), ev("ampere_sgemm_128x64_nn", cuda),
              ev("gemv2N_kernel", cuda), ev("Memcpy DtoD (Device -> Device)", cuda),
              ev("Memset (Device)", cuda), ev("aten::mm", cpu), ev("cudaLaunchKernel", cpu)]
    out = count_commands_from_events(events, n_calls=2, is_device_event=lambda e: e.device_type == cuda)
    assert out["kernels_per_call"] == 1.5 and out["memcpy_per_call"] == 0.5 and out["memset_per_call"] == 0.5
    assert out["commands_per_call"] == 2.5 and out["distinct_kernels"] == 2
    assert out["kernel_names_per_call"] == {"ampere_sgemm_128x64_nn": 1.0, "gemv2N_kernel": 0.5}
    assert classify_command("Memcpy HtoD") == "memcpy" and classify_command("Memset (Device)") == "memset"
    assert classify_command("volta_sgemm") == "kernel"
    # no device events at all: unknown, never zero
    out = count_commands_from_events([ev("aten::mm", cpu), ev("cudaLaunchKernel", cpu)], n_calls=2,
                                     is_device_event=lambda e: e.device_type == cuda)
    assert out["commands_per_call"] is None and out["no_device_events"] is True
    assert out["kernels_per_call"] is None
