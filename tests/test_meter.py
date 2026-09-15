"""Tests for tomlml.measure.meter against the fake binding (no GPU needed)."""

from __future__ import annotations

import os
import time

import pytest

from tests.conftest import Proc


def _busy_sleep(seconds: float) -> None:
    time.sleep(seconds)


def test_window_counter_and_integral_agree(fake_device):
    from tomlml.measure.meter import EnergyMeter

    meter = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None)
    w = meter.measure(lambda: _busy_sleep(0.3), label="ref", n_calls=1)

    assert w.label == "ref"
    assert 0.25 <= w.duration_s <= 0.6
    expected = 60.0 * w.duration_s
    assert w.energy_counter_j == pytest.approx(expected, rel=0.10)
    assert w.energy_power_integral_j == pytest.approx(expected, rel=0.10)
    assert w.energy_mean_power_j == pytest.approx(expected, rel=0.10)
    assert w.energy_j == w.energy_counter_j
    assert w.counter_monotonic is True
    assert w.n_samples >= 5
    assert w.mean_power_w == pytest.approx(60.0)
    assert w.power_std_w == 0.0
    assert w.max_temp_c == 35
    assert w.min_sm_clock_mhz == 1410
    assert w.throttle_union_mask == 0
    assert w.throttle_reasons == []
    assert w.contaminated is False
    assert w.sampler_errors == {}
    assert w.sampler_disabled == {}
    assert w.n_calls == 1
    assert w.energy_per_call_j() == pytest.approx(w.energy_counter_j)


def test_window_without_energy_counter_falls_back_to_integral(fake_nvml, fake_device):
    from tomlml.measure.meter import EnergyMeter

    fake_nvml.energy_supported = False
    meter = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None)
    w = meter.measure(lambda: _busy_sleep(0.15))
    assert w.energy_counter_j is None
    assert w.counter_start_mj is None and w.counter_end_mj is None
    assert w.energy_power_integral_j == pytest.approx(60.0 * w.duration_s, rel=0.10)
    assert w.energy_j == w.energy_power_integral_j
    assert w.sampler_disabled.get("energy") == "NotSupported"
    assert w.sampler_disabled.get("counter_boundary") == "NotSupported"
    assert w.counter_monotonic is None


def test_throttle_reasons_are_unioned(fake_nvml, fake_device):
    from tomlml.measure.meter import EnergyMeter

    fake_nvml.throttle_mask = 0x4
    meter = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None)
    w = meter.measure(lambda: _busy_sleep(0.1))
    assert w.throttle_union_mask == 0x4
    assert w.throttle_reasons == ["SwPowerCap"]


def test_contamination_by_foreign_compute_process(fake_nvml, fake_device):
    from tomlml.measure.meter import EnergyMeter

    fake_nvml.procs[1] = [Proc(os.getpid(), 1 << 20), Proc(4242, 2 << 20)]
    w = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None).measure(
        lambda: _busy_sleep(0.05))
    assert w.contaminated is True
    assert [p["pid"] for p in w.processes_before] == [4242]

    fake_nvml.procs[1] = [Proc(777, 1 << 10, kind="graphics")]
    w = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None).measure(
        lambda: _busy_sleep(0.05))
    assert w.contaminated is False
    assert [p["kind"] for p in w.processes_before] == ["graphics"]


def test_sync_fn_is_called_at_both_boundaries(fake_device):
    from tomlml.measure.meter import EnergyMeter

    calls = []
    meter = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: calls.append(1))
    with meter:
        _busy_sleep(0.05)
    assert len(calls) == 2
    assert meter.window is not None and meter.window.duration_s > 0


def test_stop_before_start_raises(fake_device):
    from tomlml.measure.meter import EnergyMeter

    with pytest.raises(RuntimeError):
        EnergyMeter(fake_device).stop()


def test_integrate_power_holds_edges():
    from tomlml.measure.meter import StateSample, integrate_power

    def s(t, p):
        return StateSample(t_wall=t, t_perf=t, power_w=p, temp_c=None, sm_clock_mhz=None,
                           mem_clock_mhz=None, energy_mj=None, throttle_mask=None)

    # 100 W from t=1 to t=2, 200 W from t=2 to t=3 (trapezoid: 150 between)
    samples = [s(1.0, 100.0), s(2.0, 100.0), s(3.0, 200.0)]
    # window [0, 4]: hold 100 W on [0,1], 100 on [1,2], 150 avg on [2,3], hold 200 on [3,4]
    assert integrate_power(samples, 0.0, 4.0) == pytest.approx(100 + 100 + 150 + 200)
    assert integrate_power([], 0.0, 1.0) is None
    assert integrate_power([s(0.5, None)], 0.0, 1.0) is None


def test_summarize_window_detects_non_monotonic_counter():
    from tomlml.measure.meter import StateSample, summarize_window

    def s(t, e):
        return StateSample(t_wall=t, t_perf=t, power_w=10.0, temp_c=30, sm_clock_mhz=100,
                           mem_clock_mhz=100, energy_mj=e, throttle_mask=0)

    w = summarize_window("x", 0.0, 1.0, 0.0, 1.0, 1000, 1500,
                         [s(0.2, 1200), s(0.5, 1100), s(0.8, 1400)], [], [], 0.1)
    assert w.counter_monotonic is False
    assert w.energy_counter_j == pytest.approx(0.5)

    w = summarize_window("y", 0.0, 1.0, 0.0, 1.0, 1000, 1500,
                         [s(0.2, 1100), s(0.5, 1200), s(0.8, 1400)], [], [], 0.1)
    assert w.counter_monotonic is True
    d = w.to_dict()
    assert "samples" not in d and d["label"] == "y"
    assert "samples" in w.to_dict(include_samples=True)


def test_wait_for_thermal_settle_with_fake_clock(fake_nvml, fake_device):
    from tomlml.measure.meter import wait_for_thermal_settle

    clock = {"t": 0.0}
    res = wait_for_thermal_settle(fake_device, threshold_c=1.0, window_s=5.0,
                                  timeout_s=60.0, poll_s=1.0,
                                  sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + s),
                                  clock=lambda: clock["t"])
    assert res["timed_out"] is False
    assert res["settled_temp_c"] == 35.0
    assert res["wait_time_s"] >= 5.0

    # temperature drifting one degree every poll never settles -> timeout
    temps = iter(range(35, 200))
    fake_device.temperature_c = lambda: next(temps)  # type: ignore[assignment]
    clock["t"] = 0.0
    res = wait_for_thermal_settle(fake_device, threshold_c=1.0, window_s=5.0,
                                  timeout_s=20.0, poll_s=1.0,
                                  sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + s),
                                  clock=lambda: clock["t"])
    assert res["timed_out"] is True
    assert res["wait_time_s"] >= 20.0


def test_measure_idle_discards_then_measures(fake_device):
    from tomlml.measure.meter import measure_idle

    slept = []
    w = measure_idle(fake_device, duration_s=0.3, discard_s=0.1, sample_interval_s=0.02,
                     sleep_fn=lambda s: (slept.append(s), time.sleep(s)))
    assert slept[0] == pytest.approx(0.1)
    assert slept[1] == pytest.approx(0.2)
    assert w.label == "idle"
    assert w.energy_counter_j == pytest.approx(60.0 * w.duration_s, rel=0.15)


def test_write_samples_csv(tmp_path, fake_device):
    from tomlml.measure.meter import EnergyMeter, write_samples_csv, SAMPLE_FIELDS

    w = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None).measure(
        lambda: _busy_sleep(0.1), label="csv")
    path = write_samples_csv(w.samples, tmp_path / "samples" / "csv.csv", window_label="csv")
    lines = path.read_text().splitlines()
    assert lines[0].split(",") == ["window"] + SAMPLE_FIELDS
    assert len(lines) == 1 + w.n_samples
    assert lines[1].startswith("csv,")


def test_quantized_counter_series_is_monotone_across_many_windows(fake_nvml, fake_device):
    """Regression for the EXP-001 quick-run race: with a counter that updates
    in steps and reads that take about a millisecond, reading the closing
    value before stopping the sampler let a sample taken mid-cycle exceed it.
    The fixed order must never flag a window."""
    from tomlml.measure.meter import EnergyMeter

    fake_nvml.counter_step_s = 0.02
    fake_nvml.read_latency_s = 0.002
    flags = []
    for _ in range(30):
        w = EnergyMeter(fake_device, sample_interval_s=0.005, sync_fn=lambda: None,
                        check_processes=False).measure(lambda: _busy_sleep(0.1))
        flags.append(w.counter_monotonic)
        assert w.counter_end_mj >= max(s.energy_mj for s in w.samples if s.energy_mj is not None)
    assert flags == [True] * 30


def test_implausible_power_samples_are_masked_and_counted(fake_nvml, fake_device):
    from tomlml.measure.meter import EnergyMeter, StateSample, mask_implausible_power

    def s(t, p):
        return StateSample(t_wall=t, t_perf=t, power_w=p, temp_c=None, sm_clock_mhz=None,
                           mem_clock_mhz=None, energy_mj=None, throttle_mask=None)

    masked, n = mask_implausible_power([s(0.0, 593.5), s(0.1, 60.0), s(0.2, None)], 350.0)
    assert n == 1
    assert [m.power_w for m in masked] == [None, 60.0, None]
    assert mask_implausible_power([s(0.0, 593.5)], None) == ([s(0.0, 593.5)], 0)

    # a window whose first reading is a glitch: stats and integral must ignore it
    readings = iter([593.5] + [60.0] * 1000)
    fake_device.power_w = lambda: next(readings)  # type: ignore[assignment]
    w = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None,
                    power_max_plausible_w=350.0).measure(lambda: _busy_sleep(0.2))
    assert w.power_implausible_samples == 1
    assert w.power_max_plausible_w == 350.0
    assert w.mean_power_w == pytest.approx(60.0)
    assert w.power_max_w == pytest.approx(60.0)
    assert w.energy_power_integral_j == pytest.approx(60.0 * w.duration_s, rel=0.10)
    assert w.samples[0].power_w == 593.5  # raw log keeps the reading


def test_energy_source_selection(fake_nvml, fake_device):
    from tomlml.measure.meter import EnergyMeter

    w_auto = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None).measure(
        lambda: _busy_sleep(0.1), n_calls=4)
    assert w_auto.energy_source == "auto" and w_auto.energy_j == w_auto.energy_counter_j
    w_int = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None,
                        energy_source="power_integral").measure(lambda: _busy_sleep(0.1), n_calls=4)
    assert w_int.energy_j == w_int.energy_power_integral_j
    assert w_int.energy_per_call_j() == pytest.approx(w_int.energy_power_integral_j / 4)
    assert w_int.energy_per_call_j("counter") == pytest.approx(w_int.energy_counter_j / 4)
    w_cnt = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None,
                        energy_source="counter").measure(lambda: _busy_sleep(0.1), n_calls=4)
    assert w_cnt.energy_j == w_cnt.energy_counter_j
    fake_nvml.energy_supported = False
    w_none = EnergyMeter(fake_device, sample_interval_s=0.02, sync_fn=lambda: None,
                         energy_source="counter").measure(lambda: _busy_sleep(0.05), n_calls=2)
    assert w_none.energy_j is None and w_none.energy_per_call_j() is None
    assert w_none.energy_per_call_j("power_integral") is not None
    with pytest.raises(ValueError):
        EnergyMeter(fake_device, energy_source="joules_from_thin_air")
