# Experimental Logbook

Single source of truth for what was done, when, why, and what happened.
Structured metadata plus narrative reasoning. Follows the lab-runner protocol.

## Rules

1. Append-only. Past entries are never edited or deleted. Corrections are dated
   addenda that reference the original entry.
2. Plan first. Hypothesis, variables, controls, protocol, and environment are
   written before execution. Results, observations, and interpretation are
   filled in afterwards.
3. Failures are data. Failed, aborted, and contaminated runs get entries.
4. Cross-reference. Every entry names the prior experiment or decision that
   motivated it.
5. One commit per logbook update: `docs(logbook): EXP-XXX planned|completed|failed`.
6. Every number reported anywhere (findings.md, paper) traces to an entry here
   by experiment ID, commit SHA, frozen config, and seed file.

## Conventions

- Experiment IDs: `EXP-NNN`, three digits, allocated sequentially and never reused.
  A re-run after a code change is a new ID that references the old one.
- Platform tags: `rtx4090-laptop` (Windows 11, RTX 4090 Laptop GPU, 64 GB RAM),
  `a100-sxm4-40gb` (Linux SLURM cluster). A new platform gets a new tag and a
  decision-log entry before its first measurement.
- Timestamps: local time with the timezone named. The laptop runs in
  America/New_York. Cluster entries state the cluster's timezone.
- Researcher field names who executed the run (Muntaser or Ameera), which also
  identifies the platform.
- Output directory: `experiments/exp_NNN_name/<platform-tag>/<run-id>/`, where
  run-id is a `YYYYMMDDTHHMMSS` timestamp assigned at start. Run directories
  are never overwritten; a repeated run gets a new run-id, and the logbook
  entry names which run-id is the one reported. Each run directory contains
  `config.yaml` (frozen), `environment.json`, `seed.json`, `manifest.json`,
  `results/`, `samples/`, `logs/`, `figures/`.
- Phase label on every entry: `exploratory`, `pilot`, or `confirmatory`.
  Only confirmatory results can appear in the paper as confirmatory evidence.

## Entry template

```markdown
---

## EXP-NNN: Descriptive title

**Date:** YYYY-MM-DD HH:MM (timezone)
**Researcher:** name
**Platform:** platform-tag
**Type:** computational | hardware | mixed
**Phase:** exploratory | pilot | confirmatory
**Status:** planned | running | completed | failed | paused
**Motivated by:** EXP-NNN or D-NNN

### Hypothesis
### Independent variables
### Dependent variables / metrics
### Control conditions
### Protocol
### Environment
- Hardware, software, git commit (full SHA), config file, seeds
### Pre-registered pass/fail criteria
### Results
### Observations
### Interpretation
### Artifacts
```

---

## Index

| ID | Title | Platform | Phase | Status |
|----|-------|----------|-------|--------|
| EXP-001 | Instrumentation validation (gate G0) | rtx4090-laptop, a100-sxm4-40gb | pilot | PASS on both: rtx4090-laptop (power integral), a100-sxm4-40gb (counter) |
| EXP-002 | Exact low-rank crossover, dense versus factorized GEMM (inference) | rtx4090-laptop, a100-sxm4-40gb | pilot | complete on both: rtx4090-laptop H2a NOT supported (v2); a100-sxm4-40gb H2a' NOT supported (v3), predictions A1, A2, A7 held, A3 to A6 failed |

---

## EXP-001: Instrumentation validation (gate G0)

**Date:** 2026-09-15 (America/New_York), planned
**Researcher:** Muntaser (rtx4090-laptop), Ameera (a100-sxm4-40gb)
**Platform:** both; one run directory per platform
**Type:** hardware (measurement validation)
**Phase:** pilot
**Status:** planned
**Motivated by:** D-003 (measurement primitives), D-006 (gate G0 before any pilot science); blueprint section 13 and gate G0

### Hypothesis

On each platform the NVML cumulative energy counter is supported and
monotonic, updates at a period no longer than 200 ms, and agrees with the
integrated 50 ms power samples within 5 percent over 10 s windows. After
thermal settling, a fixed compute-bound reference workload (FP32 4096x4096
GEMM) yields energy per call with a coefficient of variation below 3 percent
across 10 back-to-back 10 s blocks, and a dispatch-bound reference workload
(FP32 64x64 GEMM, one launch per call) below 5 percent. The smallest window
length at which the compute-bound workload's energy per call has CV at or
below 2 percent is at most 10 s. Idle power drifts by less than the block
noise over one minute, and the GPU returns to a thermally settled state
within 300 s after a 60 s heat load.

This is a gate, not a scientific claim. Its outcome decides whether the
measurement stack is fit to produce the study's numbers.

### Independent variables

- Platform: rtx4090-laptop, a100-sxm4-40gb
- Workload regime: matmul_large (compute bound), matmul_tiny_loop (dispatch bound), idle
- Measurement window length: 0.5, 1, 2, 5, 10, 20 s (window sufficiency section)
- Thermal state: settled cold (pre-heat) versus settled after a 60 s heat load

### Dependent variables / metrics

- Counter support, monotonicity, and update period (median interval between
  counter changes under tight polling, ms), idle and under load
- Power reading update period (ms) and, where the samples API is exposed, the
  sensor's native sample period
- Idle power (W): mean, standard deviation, linear drift (W/min), first-half
  versus second-half difference; pre-heat and post-heat, and their difference
  (hot-idle bias, W)
- Thermal settle time after the heat load (s) and whether it timed out
- Energy per call (J) from the counter, per block; CV across blocks
- Relative difference between counter energy and trapezoid-integrated power
  per block; maximum absolute value across blocks
- Per window length: CV of energy per call across 5 randomized repetitions;
  smallest length such that it and all longer lengths meet the 2 percent target
- Throttle reasons observed under load (union of bitmask over samples)
- Foreign processes on the GPU before and after every window
- Device resolution method and UUID verification

### Control conditions

- Same package versions on both platforms (D-004); OS and driver recorded.
- No clock or power-cap changes; operating state measured, not set (D-003).
- Workload inputs fixed by seed 1234 inside the workload; the experiment
  seed 42 drives only the randomized window order.
- Window sufficiency windows are run in a seeded random order so that
  thermal drift is not confounded with window length.
- Every measured window is preceded by an unmeasured warmup of the same
  workload (30 s before block sequences, 2 s before each sufficiency window),
  so no window includes a cold-start ramp.

### Protocol

All parameters are in `configs/exp_001_instrumentation.yaml`; the runner
freezes the effective config into the run directory.

1. `python scripts/run_experiment.py --config configs/exp_001_instrumentation.yaml --quick`
   once, as a pipeline smoke test. Quick results are labeled and never used
   for the gate.
2. `git status` clean, then the full run:
   laptop: `python scripts/run_experiment.py --config configs/exp_001_instrumentation.yaml`;
   cluster: `sbatch scripts/slurm/exp_001.sbatch` (see docs/QUICKSTART_A100.md).
3. S0: capability probe (versions, limits, clocks, throttle API, samples API,
   running processes) written to `results/capabilities.json`.
4. S1: tight polling of counter and power for 10 s at idle and 10 s under a
   background matmul_large load; change events written to `samples/`.
5. S2: wait for thermal settle (spread at most 1 C over 5 s, timeout 300 s),
   then 60 s idle baseline (first 5 s discarded).
6. S3: 60 s matmul_large heat load (measured), thermal settle again, then a
   second 60 s idle baseline.
7. S4, per workload: 30 s warmup, calibrate calls per 10 s block, then 10
   back-to-back measured blocks.
8. S5, per workload: randomized sequence of 6 lengths x 5 repetitions, each
   preceded by a 2 s warmup.
9. S6: final process check. S7: evaluate criteria, write `results/summary.json`.
10. Commit the run directory; on the cluster via branch `exp-001-a100` and a
    pull request. Record the run id, commit SHA, and outcome below as a dated
    addendum.

### Environment

- Hardware: rtx4090-laptop (Windows 11, RTX 4090 Laptop GPU, driver 595.79 at
  planning time, 64 GB RAM); a100-sxm4-40gb (Linux SLURM node, driver to be
  recorded)
- Software: Python 3.12, torch 2.6.0+cu124, cuDNN 9.1, numpy 2.0.2,
  nvidia-ml-py 13.595.45 (D-004); exact versions in each run's `environment.json`
- Git commit: recorded in each run's `manifest.json` (the runner refuses a
  dirty tree for full runs); the protocol commit is tagged `exp-001-protocol`
- Config file: `configs/exp_001_instrumentation.yaml` (with `configs/base.yaml`)
- Seeds: master 42 (window order); workload inputs 1234

### Pre-registered pass/fail criteria

Evaluated automatically by `tomlml.experiments.exp_001_instrumentation.evaluate_criteria`.

| ID | Criterion | Threshold |
|----|-----------|-----------|
| C1 | Counter supported and monotonic in every window | true |
| C2 | Median counter update interval under load | at most 200 ms |
| C3 | Max abs relative difference, counter vs power integral, over repeatability blocks | at most 0.05 |
| C4 | CV of energy per call across blocks | matmul_large at most 0.03; matmul_tiny_loop at most 0.05 |
| C5 | Thermal settle after heat load completes before timeout | 300 s |
| C6 | No foreign compute process in any window | 0 contaminated windows |
| C7 | Smallest sufficient window (CV at or below 0.02, all longer lengths too), matmul_large | at most 10 s |
| C8 | No forbidden throttle reason under load (HwSlowdown, HwThermalSlowdown, SwThermalSlowdown, HwPowerBrakeSlowdown) | none observed; SwPowerCap is allowed and recorded |
| C9 | torch device UUID equals the NVML device UUID | true |

Gate G0 passes on a platform only if every criterion passes on that
platform's full (non-quick) run. A failure means the measurement code is
repaired and the gate is re-run under a new experiment ID that references
this one. Graphics processes (display) on the laptop are recorded but do not
count as contamination; any compute process that is not ours does.

### Results

See the addenda below (rtx4090-laptop, 2026-09-15; a100-sxm4-40gb pending).

### Observations

See the addenda below.

### Interpretation

See the addenda below.

### Artifacts

- `experiments/exp_001_instrumentation-gate/<platform-tag>/<run-id>/`
  (`results/summary.json`, `results/capabilities.json`, `results/windows.json`,
  `samples/*.csv`, `logs/run.log`)

### Addendum 2026-09-15 16:20 (America/New_York): quick-run observations and protocol amendment v2

Written before any full run. The original entry above is unchanged.

**Quick-run observations (rtx4090-laptop, run `20260915T161501`, quick
factor 0.05, not valid for the gate, committed as data).**

- Device resolved by torch UUID, verified (`GPU-a19463b0-...`). NVML
  reports the enforced power limit as 175.0 W although `nvidia-smi` shows
  N/A; constraints 5 to 175 W.
- Counter update period: median 100.1 ms under load, 100.0 ms at idle; NVML
  poll rate on Windows about 760 Hz for two reads per poll.
- Counter anomaly: the counter advanced at about 500 W-equivalent during the
  settled idle window (power reading 4.0 W, SM 210 MHz, `GpuIdle`), about
  250 W-equivalent during 175 W power-capped GEMM blocks, and about 164
  W-equivalent during the dispatch-bound loop (power reading 63 W). Counter
  versus integral: 0.57 (large) and 2.25 (tiny loop) maximum absolute
  relative difference against a 0.05 threshold. Not a constant factor and
  not physically possible for a 175 W part; the counter is not this GPU's
  energy on this platform. Raw series in `samples/idle_pre.csv` and
  `samples/repeatability_matmul_large.csv`.
- The first power reading after initialization was 593.5 W
  (`samples/resolution_idle.csv`); all later readings were plausible.
- One block reported a non-monotonic counter series; traced to the closing
  counter read preceding the sampler stop in `EnergyMeter.stop()` (race,
  fixed and covered by a regression test).
- Thermal: heat load reached 65 C at 151 W mean (ramp included), settled in
  10 s at quick scale; post-heat idle bias +0.13 W.
- Throttle reasons under load: `SwPowerCap` (allowed), `GpuIdle` during
  the dispatch-bound loop (expected: the GPU is mostly idle there).
- Window-sufficiency and repeatability numbers at quick scale are not
  interpretable (blocks were 0.5 s against a 100 ms counter period) and are
  not used.

**Protocol amendment v2 (DECISIONS.md D-009).** Both energy quantities are
recorded in every window. C1 to C3 qualify the counter per platform and
select the platform's energy source; they no longer gate. C4 and C7 are
evaluated on the selected source. New C10 bounds the fraction of power
samples masked by the plausibility ceiling. The closing counter read now
follows the sampler stop. Per-length target durations are recorded in the
window-sufficiency summary so quick runs cannot be misread.

| ID | Criterion (v2) | Threshold | Gating |
|----|----------------|-----------|--------|
| C1 | Counter supported and monotonic in every window | true | no (selects source) |
| C2 | Median counter update interval under load | at most 200 ms | no (selects source) |
| C3 | Max abs relative difference, counter vs power integral, repeatability blocks | at most 0.05 | no (selects source) |
| C4 | CV of energy per call across blocks, on the selected source | matmul_large 0.03; matmul_tiny_loop 0.05 | yes |
| C5 | Thermal settle after heat load before timeout | 300 s | yes |
| C6 | No foreign compute process in any window | 0 | yes |
| C7 | Smallest sufficient window on the selected source, matmul_large | at most 10 s | yes |
| C8 | No forbidden throttle reason under load | none | yes |
| C9 | torch device UUID equals NVML device UUID | true | yes |
| C10 | Fraction of power samples above the plausibility ceiling | at most 0.01 | yes |

Energy source = counter if C1, C2, and C3 all pass, else power integral.
The full runs on both platforms execute under v2 (tag `exp-001-protocol-v2`).

### Addendum 2026-09-15 17:30 (America/New_York): rtx4090-laptop full run, results

**Run:** `20260915T164645`, 16:46 to 17:06 local, 1182 s. Commit
`8c99e38c5aa4362458ab62be846ba28d33ffee84` (protocol v2), clean tree,
config hashes in the manifest. Laptop on AC power, display on the integrated
GPU, no other GPU applications; a second quick run (`20260915T164421`, also
committed) immediately preceded it, so the GPU started warm (52 C at the
first settle).

**Status:** completed. **Gate G0 on rtx4090-laptop: PASS (valid).**
Energy source for this platform: **power integral** (counter rejected on C3).

### Results (rtx4090-laptop)

| Quantity | Value |
|----------|-------|
| Device resolution | torch UUID, verified; NVML index 0, PCI 00000000:01:00.0 |
| Driver / CUDA driver | 595.79 / 13.2; throttle API `nvmlDeviceGetCurrentClocksEventReasons`; samples API not supported |
| Power limit | enforced 175.0 W (NVML), constraints 5.0 to 175.0 W; plausibility ceiling 350 W |
| Counter update period | 99.97 ms median (idle 99.5 to 100.5; load 71.7 to 127.7, p90 100.7), 100 updates in 10 s |
| Power reading update period | 500 ms (idle 498.5 to 500.8; load 486.9 to 512.7), 20 updates in 10 s |
| NVML poll rate (2 reads per poll) | 1296 Hz idle, 822 Hz under load |
| Idle pre-heat (55 s after 5 s discard) | 4.396 W, std 0.177, drift -0.62 W/min (r2 0.86), 45.4 C mean, SM 210 MHz |
| Idle post-heat | 4.401 W, std 0.206, drift -0.72 W/min; hot-idle bias +0.006 W |
| Thermal settle (1 C over 5 s) | 13 s at 52 C before; 16 s at 52 C after the heat load |
| Heat load (65.9 s, 11428 calls) | 171.7 W mean, 74 C max, SM 1704 MHz mean; `SwPowerCap` (plus `GpuIdle` at the start) |
| C1 / C2 / C3 (counter qualification) | pass / pass (100 ms) / **fail** (max abs rel 1.996) |
| Repeatability matmul_large, 10 x 10 s, 1739 calls per block | integral 1.0147 J/call, CV 0.32% (1.0090 to 1.0177, rising monotonically); counter 1.514 J/call, CV 4.2%; power 174.6 to 175.1 W; 73 to 79 C; SM 1711 to 1689 MHz; block duration 10.023 to 10.134 s |
| Repeatability matmul_tiny_loop, 10 x about 9.5 s, 525530 calls per block | integral 1.107 mJ/call, CV 1.99% (1.074 to 1.139); counter 2.89 mJ/call, CV 6.9%; power 62.6 to 59.8 W falling; 64 to 57 C falling; SM 2325 MHz; block duration 9.26 to 9.91 s |
| Window sufficiency matmul_large (integral CV, 5 reps) | 0.5 s 0.23%, 1 s 0.13%, 2 s 0.29%, 5 s 0.13%, 10 s 0.26%, 20 s 0.26%; smallest sufficient window 0.5 s |
| Window sufficiency matmul_tiny_loop (integral CV, 5 reps) | 0.5 s 8.6%, 1 s 3.2%, 2 s 1.8%, 5 s 3.8%, 10 s 2.2%, 20 s 1.4%; smallest sufficient window 20 s |
| Isolation | 0 foreign processes; 0 of 83 windows contaminated |
| C10 | 0 of 14877 samples masked |
| Gating criteria C4 to C10 | all pass (C4 0.32% and 1.99%; C5 16 s; C7 0.5 s; C8 none; C9 true) |

Full detail: `results/summary.json`, `results/windows.json` (83 windows),
`samples/*.csv` (14877 sampler rows plus change-event logs).

### Observations (rtx4090-laptop)

1. The counter behaves as in the quick run: about 500 W-equivalent at idle
   (26.0 kJ across the 55 s idle window against 242 J from the power
   integral), 1.4 to 1.6 J/call against 1.01 J/call under the cap, 2.6 to
   3.4 mJ/call against 1.1 mJ/call for the dispatch loop. Monotonic and
   regular (100 ms), just not energy.
2. Under the 175 W cap, energy per call is a timing measurement: power is
   pinned, so the integral equals 175 W times the block duration to well
   within a percent, and 0.5 s windows already give 0.2% CV. The 0.9% rise
   across the ten blocks is a thermal effect (79 C by the last block, SM
   clock down 22 MHz), not sensor noise. The pass on C7 at 0.5 s is therefore
   specific to power-capped workloads and must not be generalized.
3. The dispatch-bound loop is the informative regime. Its block-to-block
   variation is in the block duration (9.26 to 9.91 s for a fixed call
   count), that is, CPU-side dispatch timing, with power nearly constant near
   60 W and the SM at its 2325 MHz boost clock. Twenty-second windows are
   needed for 2% CV. This is the regime reinforcement learning agents live in.
4. Confound in the tiny-loop blocks: the GPU was cooling from the preceding
   175 W sequence (64 to 57 C, power 62.6 to 59.8 W across the blocks). A
   fixed 30 s warmup at 60 W does not reach thermal steady state after a
   175 W sequence; the 1.99% CV includes that trend.
5. Idle power drifts about -0.6 W/min while the temperature criterion is
   already met (the settle criterion is on temperature spread, not on idle
   power); the half-window difference was -0.29 W on a 4.4 W baseline. For
   idle-subtracted sensitivity analyses the idle mean over 55 s is used with
   this drift recorded.
6. Hot-idle bias is negligible (+0.006 W) once temperature has settled,
   which validates the settle-before-baseline rule inherited from
   TOMLSignals on this platform.
7. The power reading updates every 500 ms. For steady workloads that is
   immaterial; for windows under about 5 s the integral rests on fewer than
   ten distinct readings, and window-edge readings may reflect the preceding
   half second. Warming up with the same workload before each window keeps
   the edges representative.
8. No 593 W glitch this time (the quick run's glitch was the first read after
   initialization, in the polling section, not in a window); C10 masked zero
   samples.

### Interpretation (rtx4090-laptop)

The measurement stack is fit for purpose on this platform with the power
integral as the energy source. Consequences for the study protocol on
rtx4090-laptop, to be written into the next experiment's decision entry:

- Energy source: power integral (D-009), recorded in
  `configs/platform/rtx4090-laptop.yaml`.
- Default measurement window 20 s for any workload that is not pinned at the
  power cap; 10 s is acceptable only for power-capped compute-bound workloads,
  and the paper reports which regime each measurement was in.
- Before each workload's block sequence, warm up under that workload until
  its temperature is stable (the settle criterion applied under load), not
  for a fixed 30 s; the tiny-loop cooling confound is the reason.
- Report SM clock and temperature per window; under the cap, clock drift is
  the energy drift.
- Randomize condition order within blocks (already the rule) and keep the
  sampler at 50 ms.

The A100 is judged by its own full run. Nothing here transfers to it except
the protocol.

### Addendum 2026-09-26 (America/New_York): a100-sxm4-40gb full run, results

**Run:** `20260925T064203` by Ameera, node gpu01 (SLURM job 78865, partition
gpu1, whole node: four A100-SXM4-40GB allocated, 8 CPUs per task), 06:42 to
07:02 cluster local time, 1244 s. Commit `7087068e` (protocol v2), clean
tree. Stack matched to D-004: Python 3.12.3, torch 2.6.0+cu124, cuDNN 9.1,
nvidia-ml-py 13.595.45; driver 580.173.02 (CUDA driver 13.0), Linux 6.8,
AMD EPYC 7402, 504 GB. Delivered as pull request #1 (branch
`exp-001-a100`), reviewed and merged 2026-09-26.

**Status:** completed. **Gate G0 on a100-sxm4-40gb: PASS (valid).**
Energy source for this platform: **counter** (C1, C2, C3 all pass).

### Results (a100-sxm4-40gb)

| Quantity | Value |
|----------|-------|
| Device resolution | torch UUID, verified; NVML index 0 of 4, PCI 00000000:01:00.0 |
| Power limit | enforced 400.0 W, constraints 100 to 400 W; plausibility ceiling 800 W |
| Counter update period | 100.96 ms idle (98.6 to 103.4), 101.15 ms under load (78.1 to 125.5); 99 updates in 10 s |
| Power reading update period | about 103 to 108 ms median; native sample buffer 20.2 ms (samples API supported) |
| NVML poll rate (2 reads per poll) | 346 Hz idle, 304 Hz under load |
| Idle pre-heat (55 s) | 79.08 W, std 0.60, drift -2.19 W/min, 35.6 C, SM 1095 MHz; counter 4345.5 J against integral 4349.6 J |
| Idle post-heat | 79.96 W, std 0.99, drift -3.60 W/min; hot-idle bias +0.88 W |
| Thermal settle (1 C over 5 s) | 9 s at 40 C before; 14 s after the heat load |
| Heat load (60 s) | 379.5 W mean, 66 C max, SM 1409 MHz, no throttle; 2.743 J/call (counter), 2.741 (integral) |
| C1 / C2 / C3 | pass / pass (101 ms) / pass (max abs rel 0.0101) |
| Repeatability matmul_large, 10 x 10 s, 1386 calls per block | counter 2.853 J/call, CV 0.80 percent (2.816 to 2.879, rising with temperature 64 to 69 C at a fixed 1410 MHz; power 393 to 399 W); integral 2.852, CV 0.91 percent; SwPowerCap in the last block only |
| Repeatability matmul_tiny_loop, 10 x 10 s, 1,005,647 calls per block | counter 1.159 mJ/call, CV 1.10 percent (1.178 falling to 1.150 while cooling 45 to 38 C); 9.94 microseconds per call; 116 W; SM 1410 MHz |
| Window sufficiency matmul_large (counter CV, 5 reps) | 0.5 s 0.80 percent, 1 s 0.18, 2 s 0.13, 5 s 0.06, 10 s 0.54, 20 s 0.10; smallest sufficient window 0.5 s |
| Window sufficiency matmul_tiny_loop (counter CV, 5 reps) | 0.5 s 0.29 percent, 1 s 4.4, 2 s 5.3, 5 s 3.2, 10 s 3.3, 20 s 3.2; no sufficient window (drift dominated, see observations) |
| Isolation | 0 foreign processes; 0 of 83 windows contaminated |
| C10 | 0 of 15,436 samples masked |
| Gating criteria C4 to C10 | all pass (C4 0.80 and 1.10 percent; C5 14 s; C7 0.5 s; C8 none; C9 true) |

### Observations (a100-sxm4-40gb)

1. The counter is a valid energy source here: 1.0 percent agreement with
   the integral on every 10 s block, 0.1 percent on the 55 s idle windows,
   monotonic throughout. The same API that reports about 500 W-equivalent
   at a 4 W idle on the laptop reports 79 W at idle here against a 79 W
   power reading. Platform-specific validation of the counter is required;
   the gate did its job on both.
2. Temperature enters energy differently on the two platforms. Under the
   laptop's cap, clocks fall at fixed power and time per call rises; on
   the A100 the clock stays at 1410 MHz and power rises 393 to 399 W as
   the die warms 64 to 69 C, so energy per call rises 1.6 percent over 100
   s at fixed time per call. Both are leakage; one shows up as time, the
   other as power.
3. The dispatch floor is about 10 microseconds per call (Linux, EPYC)
   against about 14 to 19 on the Windows laptop, at nearly twice the
   power (116 W against 60 W), so the energy of one tiny call is almost
   the same on both platforms (1.16 against 1.11 mJ). The A100's higher
   floor power comes with a 79 W idle.
4. The tiny-loop blocks and windows ran while the GPU cooled from the 395
   W sequence (45 to 38 C), and their block-to-block trend (2.4 percent
   over 100 s) is that cooling, as on the laptop; hence no window length
   reached the 2 percent target in the sufficiency sweep except the 0.5 s
   windows that happened to fall close together. D-010's settle-under-load
   rule is the remedy and applies to every run after the gate.
5. Persistence mode is off on the node (recorded). All four GPUs were
   allocated to the job, which gives whole-node isolation.

### Interpretation (a100-sxm4-40gb)

The measurement stack is fit for purpose on the A100 with the counter as
the energy source, recorded in `configs/platform/a100-sxm4-40gb.yaml`. The
cross-platform facts feed the A100 predictions filed for EXP-002 on
2026-09-18: a launch floor near 10 microseconds and the published 1555 GB/s
put the transition size near 1970, inside the filed 1400 to 2800.

---

## EXP-002: Exact low-rank crossover, dense versus factorized GEMM (inference)

**Date:** 2026-09-15 (America/New_York), planned
**Researcher:** Muntaser (rtx4090-laptop), Ameera (a100-sxm4-40gb, after the laptop result and the filed A100 predictions)
**Platform:** both; one run directory per platform
**Type:** computational (measured energy of exact-function pairs)
**Phase:** pilot (spine experiment; thresholds provisional per D-011)
**Status:** planned
**Motivated by:** STUDY_DESIGN.md H2a and H3c; D-010 (protocol); EXP-001 (windows, warmup, energy source)

### Hypothesis

For a linear map y = Wx with W = VU of exact rank r (V is d by r, U is r by
d, d_in = d_out = d), the dense realization executes B d^2 MACs in one GEMM
and the factorized realization executes B r (2d) MACs in two GEMMs with an
intermediate B by r tensor. FLOPs predicts that the factorized realization
is cheaper exactly when r < d/2, for every batch size B. The physical model
predicts a crossover rank r*(B, d) that lies below d/2 at small B, because
the second launch and the poorly shaped GEMMs cost energy that does not
scale with MACs, and that rises toward d/2 as B grows and compute dominates.
H2a: at B = 1 the measured r* is at most d/4 (or no crossover exists in the
tested range, dense winning throughout), r*(B) is non-decreasing in B, and a
three-term model (compute, memory, dispatch) calibrated on two shapes
predicts r*(B) on two held-out shapes within its bootstrap interval. The
FLOPs rule is wrong by at least a factor of 2 at B = 1.

Honesty note: with one operation type, the fixed op-cost ratios of the TO
table do not affect this fit; this experiment tests a three-term physical
model against FLOPs, not the table (H1c is tested by the catalog).

### Independent variables

- Shape d: 512, 1024, 2048, 4096 (square; d_in = d_out)
- Rank fraction f = r/d: 1/64, 1/32, 1/16, 1/8, 1/4, 3/8, 1/2, 3/4
- Batch B: 1, 4, 16, 64, 256, 1024, 4096
- Realization: dense (one GEMM, W precomputed) versus factorized (two GEMMs)
- Platform: rtx4090-laptop, a100-sxm4-40gb
- Calibration shapes: d in {512, 2048}; held-out shapes: d in {1024, 4096}

224 configurations per platform, two realizations each, three randomized
repetitions of 20 s windows: 1344 windows, about 8.5 hours per platform.

### Dependent variables / metrics

- Energy per call (J) from the platform's energy source (D-009), per
  window; mean and standard deviation over the three repetitions; the other
  energy quantity recorded alongside
- Mean power, temperature, SM clock, throttle reasons per window (regime
  label: capped or uncapped, D-010 item 1)
- Command census per configuration and realization: kernel, memcpy, and
  memset counts with kernel names (torch.profiler)
- TO descriptors per configuration: MACs, HBM and SRAM words under the
  declared rules, command count
- Measured crossover r*(B, d): the rank at which the repetition-mean energy
  ratio E_fact / E_dense crosses 1, by log-linear interpolation between
  tested ranks; "none, dense wins" if the ratio exceeds 1 at every tested
  rank; "none, factorized wins" if below 1 at every tested rank
- Predicted crossover r*(B, d) from the selected model on held-out shapes,
  with a 95 percent bootstrap interval (1000 resamples of calibration
  windows, refit, recompute)
- Held-out prediction error: median and maximum absolute percentage error
  of energy per call; the same for FLOPs (P0)
- Decision regret: extra energy over the measured-optimal realization when
  choosing by the model, and when choosing by FLOPs, summed over the
  held-out grid
- Exactness: relative Frobenius difference between factorized and dense
  outputs per configuration

### Control conditions

- FP32 throughout; TF32 disabled; `float32_matmul_precision` highest;
  recorded in `environment.json` and in the workload description
- Same inputs for both realizations (X per batch size, seed 1234); weights
  V, U drawn with seed 1234, W = fp32(fp64(V) fp64(U)); the same weight
  sets are used by both realizations
- Weight-set cycling: K = ceil(2 L2 / (4 d^2)) independent weight sets per
  shape, cycled call by call, so the dense weight footprint is at least
  twice the L2 cache (D-010 item 5); factorized sets cycle identically. K and
  the L2 size are written to the results
- Preallocated outputs and intermediates (`torch.mm(..., out=)`); no
  allocation in the measured loop
- Thermal settle under a mid-grid load (d, f = 1/4, B = 256, dense) before
  each shape's block sequence; 2 s same-configuration warmup before each
  window (D-010 item 2)
- Condition order within each shape randomized with the run seed; order
  written to the results (D-010 item 7)
- Energy source and window length from the platform overlay; 20 s windows
  for every configuration regardless of regime, so that capped and uncapped
  cells are treated alike
- Model selection (M1 versus M2, below) by leave-one-cell-out
  cross-validation over the 14 calibration (d, B) cells only; the held-out
  shapes are scored once by the analysis script

### Models compared

- P0, FLOPs: E = k MACs, k fitted on calibration. Predicts r* = d/2 for
  every B and d.
- M1 (P4 form): E = a_c TO_compute + a_m TO_memory + a_o S_o, with TO_compute
  = 5000 MACs, TO_memory = 10000 HBM words for every operand read or written
  once per call (X, weights, intermediate written and read, Y), S_o the
  measured command count per configuration; a_c, a_m, a_o at least 0,
  fitted by non-negative least squares on calibration windows.
- M2: M1 with the L2-residency rule: weights whose cycled footprint fits in
  L2 are charged 192 TOs per word (SRAM) instead of 10000. With the cycling
  rule this affects only factorized weights at small r.
- The fused-sequential term is zero for these workloads by construction.

### Protocol

Parameters are in `configs/exp_002_lowrank_crossover.yaml`.

1. Quick run (`--quick`) on a reduced grid (2 shapes, 3 fractions, 3 batches,
   2 repetitions, 1 s windows) as a pipeline check; not used for any result.
2. Full run with a clean tree, laptop on AC power, display on the integrated
   GPU, no other GPU applications, machine left alone.
3. Per shape d: allocate inputs, K weight sets, outputs and intermediates;
   verify exactness for every (f, B) and record it; command census for every
   (f, B, realization) with torch.profiler after 5 warmup calls; thermal
   settle under the mid-grid load; randomized sequence of 8 x 7 x 2 x 3 = 336
   windows, each 2 s warmup then 20 s measured with calibrated call count.
4. Analysis script (`scripts/analyze_exp_002.py`, run after the measurement
   run, output into the same run directory): repetition means, descriptors,
   fits of P0, M1, M2 on calibration shapes, cross-validated selection,
   held-out predictions with bootstrap intervals, measured and predicted
   crossovers, regret, criteria evaluation, figures.
5. Laptop cross-check of the census: Nsight Systems on 8 configurations
   (both realizations, 4 shapes at f = 1/8, B in {1, 4096}), counts compared
   with the profiler census and logged as an addendum.
6. Before Ameera's run: file the A100 predictions (H3c) as an addendum with
   a commit hash.

### Environment

- Hardware: as EXP-001 per platform; L2 cache size from torch device
  properties recorded per run
- Software: reference stack (D-004); torch.profiler for the census
- Git commit: in each run's `manifest.json`; the protocol commit is tagged
  `exp-002-protocol`
- Config file: `configs/exp_002_lowrank_crossover.yaml` with `configs/base.yaml`
  and the platform overlay
- Seeds: master 42 (condition order, bootstrap); workload data 1234

### Pre-registered criteria (provisional numbers, D-011)

| ID | Criterion | Threshold |
|----|-----------|-----------|
| E1 | Held-out energy prediction, selected model, median absolute percentage error | at most 20 percent (P0 reported alongside) |
| E2 | Measured r* inside the 95 percent bootstrap interval of the predicted r*, over the 14 held-out (d, B) cells; matching "none" verdicts count as hits | at least 12 of 14 |
| E3 | At B = 1 on both held-out shapes: measured r* at most d/4 or no crossover with dense winning; the selected model predicts the same side | both shapes |
| E4 | Measured r*(B) non-decreasing in B on both held-out shapes (Spearman rank correlation, "none, dense wins" ranked lowest, "none, factorized wins" ranked highest) | at least 0.8 |
| E5 | Exactness: configurations within the 1e-4 relative Frobenius tolerance | all; a run with more than 5 percent exclusions is invalid |
| E6 | Decision regret of the model over the held-out grid (reported; FLOPs regret alongside) | at most 5 percent |
| E7 | Isolation and plausibility as in EXP-001 (C6, C10) | 0 contaminated windows; at most 1 percent masked samples |

H2a is supported on a platform if E1 to E5 and E7 pass; E6 is reported.
A failure is reported as such; no re-run of the same protocol to chase a
pass.

### Addendum 2026-09-15 22:20 (America/New_York): analysis pre-registered; E2 amended before any full-run data was analyzed

The analysis (`src/tomlml/analysis/exp_002.py`, `scripts/analyze_exp_002.py`)
was written and tested on synthetic runs generated from a known three-term
model while the rtx4090-laptop full run was in progress and before any of
its windows were inspected. Two declarations and one amendment:

1. Fitting convention: non-negative least squares on calibration windows
   with relative weighting (rows scaled by 1/energy), so the fit minimizes
   squared relative error, matching the percentage-error criteria.
2. Model selection: leave-one-(d, B)-cell-out cross-validation over the 14
   calibration cells, scored by mean per-fold median APE; ties go to M1.
3. E2 amended. As registered, E2 asked whether the measured r* lies inside
   the bootstrap interval of the predicted r*. That interval carries only
   prediction uncertainty; on a synthetic run with 1 percent window noise
   the calibration fit was so tight that the interval was 0.1 percent wide
   while the measured r* has about 1 percent repetition noise, so a model
   predicting r* within 1 to 2 percent failed every cell. Amended rule: two
   bootstrap intervals per held-out cell, one from resampling calibration
   windows within configuration (prediction uncertainty) and one from
   resampling the held-out repetitions within configuration (measurement
   uncertainty), 1000 resamples each; a cell is a hit when the two 95
   percent intervals overlap on the encoded scale (numeric r*, none_dense as
   0, none_factorized as infinity). Threshold unchanged: at least 12 of 14.
   Reported alongside, not gating: the point rule (same verdict kind and
   numeric r* within a factor of 1.5), and whether the measured point lies
   inside the prediction interval (the original wording).

Also noted from the synthetic tests: the three-term model can produce r*
falling with batch size when the memory coefficient is large relative to the
compute coefficient (the intermediate tensor's traffic scales with B r), so
E3 and E4 are physical predictions about this hardware, not consequences of
the model's form.

### Results

#### rtx4090-laptop, run `20260915T180708` (addendum 2026-09-18, America/New_York)

**Run.** Started 2026-09-15 18:07 at commit `70d7c61e` (clean tree), 20 s
windows, energy source power integral, L2 64 MiB from the device, 128, 32,
8, 2 weight sets for d = 512, 1024, 2048, 4096. The first session measured
261 windows and was killed from outside at about 19:54 (manifest status
still `running` at the resume; no exception, no traceback; cause not
established, sleep or a forced stop suspected; AC sleep and hibernate
timeouts had been set to never, lid action not changed). Resumed at 20:19
at commit `7087068e` with `--resume-any-commit` (analysis and docs commits
only since the start), 1083 windows measured in the second session, all
1344 present, completed 2026-09-16 03:14. Total measurement time about 8.7
hours. 0 configurations excluded (exactness max 1e-6 to 1e-5 relative
Frobenius), 0 contaminated windows, 0 implausible power samples in 537,600.
Repetition CV of energy per call: median 0.76 percent, p90 4.7 percent,
maximum 15.8 percent. Dense energy at fixed (d, B) varies by at most 2
percent across the eight weight sets of different rank: data dependence of
GEMM energy is negligible here.

**Deviation: the command census produced no device events.** Every
configuration's `commands_per_call` is 0.0. `torch.profiler` with the CUDA
activity recorded no kernel events on this Windows machine (attribution
pending; `scripts/diag_profiler.py` added), and the census wrote 0 instead
of unknown. The analysis as first executed on 2026-09-16 therefore fitted
every model with a dispatch feature identically zero (E1 fail at 23.4
percent median APE, dispatch coefficient zero by construction). The
corrected analysis, executed 2026-09-18 from the unchanged run files,
treats a non-positive census as unknown and uses the declared count of one
command per GEMM (1 dense, 2 factorized), which is the D-010 item 4
fallback and remains to be validated against Nsight Systems on a subset.
Both executions are reported; the corrected one is primary and is labeled
"command counts declared, not measured" everywhere it is used.

**Criteria (v2, the run's frozen config), corrected analysis.**

| ID | Result | Value |
|----|--------|-------|
| E1 | PASS | selected model M1 (leave-one-cell-out CV 0.331 versus M2 0.338); held-out median APE 0.124, p90 0.261; P0 0.596 |
| E2 | FAIL | interval overlap 3 of 14 (measurement intervals about 2 percent wide, prediction intervals under 1 percent); point rule (factor 1.5) 13 of 14, the miss d = 1024, B = 4 (measured 374, predicted 220) |
| E3 | FAIL | d = 1024: measured r*(B=1) = 202 (0.198 d), predicted 206, pass; d = 4096: measured 2034 (0.497 d), fail on the measurement |
| E4 | FAIL | Spearman 0.79 at d = 1024 (threshold 0.8), -0.04 at d = 4096 (flat between 0.41 and 0.54 d) |
| E5 | PASS | 0 excluded |
| E6 | PASS (reported) | summed regret over the held-out grid 0.06 percent (model) versus 0.09 percent (FLOPs), dominated by the joule-scale cells; per-cell mean 0.7 versus 1.2 percent; wrong choices 3 versus 7 of 112; worst cell 62 percent (model, d = 1024, B = 4) versus 51 percent (FLOPs) |
| E7 | PASS | 0 contaminated, 0 implausible |
| E8 (exploratory) | 11 of 14 | M3 point rule |
| E9 (exploratory) | FAIL | M3 held-out median APE 0.240 against M1's 0.124 |
| E10 (exploratory) | PASS | measured r*(B=1)/d 0.198 at 1024 (M3 0.174), 0.497 at 4096 (M3 0.500) |

**H2a on rtx4090-laptop: NOT SUPPORTED as stated** (E2, E3, E4 fail).

Fitted coefficients, M1 on d in {512, 2048}: 0.637 mJ per command, 15.3 pJ
per MAC, 1.03 nJ per HBM word (95 percent bootstrap intervals within 1
percent). M3, joint fit on energy and per-call time: 16.6 microseconds per
command, 443 GB/s, 17.4 TFLOPS effective, floor power 46 W, memory-bound
power 175 W, compute-bound 184 W (cap-bounded), transition size 1358;
calibration median APE 0.209 (energy), 0.139 (time).

Measured crossover fraction r*/d (rows d, columns B = 1, 4, 16, 64, 256, 1024, 4096):
512: 0.021, dense always, 0.169, 0.157, 0.219, 0.368, 0.503;
1024: 0.198, 0.366, 0.262, 0.327, 0.336, 0.443, 0.473;
2048: 0.461, 0.459, 0.365, 0.327, 0.444, 0.405, 0.463;
4096: 0.497, 0.421, 0.466, 0.538, 0.415, 0.467, 0.491.
FLOPs predicts 0.5 everywhere: wrong by a factor of 24 at (512, 1), 2.5 at
(1024, 1), 1.1 at (2048, 1), 1.0 at (4096, 1).

### Observations (rtx4090-laptop)

1. Three regimes are visible in the per-call times and powers. Dispatch
   bound (d = 512, B = 1): dense 14.0 microseconds per call at 64 W;
   factorized with two launches 25.4 microseconds at 33 W and 1455 MHz. Twice
   the launches, 1.8 times the time, half the power, equal energy (0.89
   against 0.84 mJ): the GPU downclocks while it waits, so a launch's
   energy is not a constant. Memory bound (d = 4096, B = 1): dense reads 64
   MB in 123 microseconds, 520 GB/s on a 576 GB/s part, at 149 W, uncapped.
   Compute bound (B = 4096): everything at the 175 W cap, time proportional
   to MACs at 23.6 TFLOPS, factorized at f = 1/2 about 2 percent slower.
2. The B = 1 crossover fraction rises with d (0.021, 0.198, 0.461, 0.497)
   because the dense GEMV moves from dispatch bound to memory bound; a
   GEMV's bytes scale like its MACs, so at large d the FLOPs rule becomes
   right. At fixed d the fraction rises with B where dispatch matters (512,
   1024) and is flat near d/2 for d at or above 2048. H2a stated the
   dependence on B alone; the dependence on d is the larger effect.
3. The additive TOML model with command counts captures the bulk: 12.4
   percent median error on held-out shapes, crossovers at B = 1 within 3
   percent, 13 of 14 cells within a factor of 1.5. Its largest errors are
   the small-rank factorized cells at B = 1 (about 50 percent over), where
   the downclocked two-launch floor costs no more energy than one launch;
   an additive per-command energy cannot represent that.
4. The max-of-times model captures the regimes and the transition size but
   predicts energy worse than the linear model, because one throughput
   constant cannot represent cuBLAS's shape-dependent kernels (2 to 24
   TFLOPS across the grid) and its floor power is fixed while the real one
   moves with DVFS.
5. E2 as an interval-overlap test is a test of exact agreement given
   measurement intervals of about 2 percent; a model with 12 percent error
   cannot pass it while still making every decision correctly. The
   decision-relevant statement is the point rule and the regret.
6. The registered regret metric sums energy over the held-out grid and is
   dominated by the joule-scale cells; it says nothing about the small
   cells where FLOPs is wrong by 24x. A per-cell relative regret is added
   (D-012) as a reported metric.

### Interpretation (rtx4090-laptop)

H2a is refuted in the form registered: the B = 1 gap below d/4 holds for d
at or below 1024 and not for memory-bound sizes, and monotonicity in B
holds only where dispatch matters. The measurement stack, the exact-pair
design, and the calibrate-then-predict protocol all worked; the hypothesis
was mis-specified by leaving d out, and the census failed silently. The
additive TOML form, given command counts, is a usable predictor of the
crossover on this platform. The mechanism (a dispatch floor at reduced
power, a memory-bound regime, a compute-bound regime at the cap, with the
B = 1 transition size set by launch time and bandwidth) is the content to
carry forward, restated as H2a' with d in it and tested prospectively on
the A100 (addendum below and D-012). The nsys cross-check on the laptop is
now required to validate the declared command counts.

### Addendum 2026-09-18: A100 predictions filed before the A100 run (H3c), and criteria v3

Filed at the commit containing this addendum (tag `exp-002-a100-predictions`),
before any EXP-002 measurement on a100-sxm4-40gb. Basis: the laptop
mechanism above, the A100's published bandwidth (1555 GB/s HBM2e) and FP32
throughput (19.5 TFLOPS), and the expectation of a similar or larger
CPU-side launch floor on a Linux node (5 to 20 microseconds). Each item is
falsifiable by the A100 run alone.

- A1 (structure in d). r*(B=1)/d rises with d: at most 0.30 at d = 1024
  and at least 0.45 at d = 4096 (E10 form).
- A2 (transition size larger than the laptop's). d_t = sqrt(t_launch x BW /
  4 bytes) with the A100's bandwidth is between 1400 and 2800 for a floor
  of 5 to 20 microseconds, against 1358 fitted on the laptop. Prediction:
  at d = 2048, B = 1, the A100's crossover fraction is below 0.40 (the
  laptop's is 0.461); at d = 4096, B = 1, it is at least 0.45.
- A3 (large batch). At B = 4096 every shape's r*/d is between 0.42 and
  0.50 (factorized at f = 1/2 costs 0 to 15 percent more than dense).
- A4 (FLOPs errors). At (512, 1) FLOPs is wrong by at least a factor of
  10 (or dense wins throughout); at (1024, 1) by at least 2; at (4096, 1)
  by at most 1.2.
- A5 (coefficients). M1 on the A100 calibration shapes: 0.5 to 2.0 mJ per
  command (laptop 0.64), 5 to 10 pJ per MAC (laptop 15.3), 0.3 to 1.0 nJ
  per HBM word (laptop 1.03).
- A6 (predictor). M1 with measured command counts: held-out median APE at
  most 20 percent and at least 12 of 14 held-out cells within a factor of
  1.5 (E1 and E2 under v3).
- A7 (counter). The A100's NVML energy counter passes C1 to C3 in EXP-001
  and becomes that platform's energy source; if not, the power integral is
  used and A1 to A6 stand unchanged.

Criteria v3 for the A100 run (`configs/exp_002_lowrank_crossover.yaml`,
`criteria.version: 3`, D-012): E2 gates on the point rule (factor 1.5),
interval overlap reported; E3 and E4 apply to d = 1024 (E4 threshold 0.7,
set from the laptop's 0.79), with d = 4096 reported under E10 instead; E8
and E10 gate (M3 point rule at least 12 of 14; the d-structure); E6 and E9
reported. Gate: E1, E2, E3, E4, E5, E7, E8, E10. The A100 run's census is
expected to work (Linux CUPTI); if it also returns no device events, the
run is analyzed with declared counts and labeled as here.

### Artifacts

- `experiments/exp_002_lowrank-crossover/<platform-tag>/<run-id>/`
  (`results/windows.json`, `results/census.json`, `results/exactness.json`,
  `results/summary.json`, `results/analysis.json`, `samples/*.csv`,
  `figures/`, `logs/run.log`)

### Addendum 2026-10-30 (America/New_York): a100-sxm4-40gb full run, results and the test of H2a' and A1 to A7

**Run.** `20261030T001246` by Ameera, one A100-SXM4-40GB of a whole-node
SLURM allocation (four GPUs allocated, 8 CPUs per task), 00:12 to 08:40
cluster local time, 30464 s, started at commit `e1358fcbe2e9b7fca52e30dc63c6d30b2e170cb9`
(main plus the partition edit to `scripts/slurm/exp_002.sbatch`), clean tree,
config hashes in the manifest, criteria v3 frozen in `config.yaml`
(`criteria.version: 3`, D-012). Delivered as pull request #2 (branch
`exp-002-a100`), merged 2026-10-30 (`97857d5`). Energy source setting `auto`
(the overlay `configs/platform/a100-sxm4-40gb.yaml` was not in her clone;
`auto` resolves to the counter when it is available, so the data are the
counter's, as D-009 prescribes for this platform; the counter stayed within
0.59 percent of the power integral in every window, median 0.18 percent).
All 1344 windows measured in one session, 538,803 samples, 0 above the
plausibility ceiling, 0 contaminated windows, 0 exactness exclusions
(224 configurations), L2 40 MiB, weight sets 80, 20, 5, 2. Repetition CV of
energy per call: median 1.05 percent, p90 2.0 percent, max 4.3 percent.
Dense energy varies by at most 4.9 percent across the eight weight sets at
fixed (d, B). 93 of 1344 windows capped (SwPowerCap), all large-batch GEMMs
at d = 2048.

**Census worked.** `torch.profiler` returned device events for every
configuration (Linux CUPTI): 1 to 4 commands per call (72, 206, 115, 55 of
the 448 configuration-realization pairs), kernel names recorded. cuBLAS runs
many dense B = 1 calls as a split-K GEMV plus `splitKreduce` (2 commands)
and the factorized form in 2 to 4 kernels. `census_fallback_used: false`;
the analysis used measured counts.

**Analysis.** `scripts/analyze_exp_002.py` at commit `e1358fc` (Ameera,
committed with the run) and reproduced independently on a second machine
from the committed run files (identical to floating point). Criteria v3:

| ID | Result | Value |
|----|--------|-------|
| E1 | PASS | M1 selected (CV 0.121 vs M2 0.123); held-out median APE 0.081, p90 0.213, max 0.627; P0 0.450 |
| E2 | FAIL | point rule 11 of 14 (threshold 12); misses (1024, 1) measured 293 vs predicted none_dense, (1024, 4) measured none_dense vs 54, (1024, 16) measured none_dense vs 92; interval overlap 4 of 14 |
| E3 | FAIL | d = 1024: measured r*(B=1) = 292.7 (0.286 d) above d/4 = 256; model predicts none_dense |
| E4 | FAIL | Spearman 0.667 at d = 1024 (threshold 0.7); d = 4096 reported 0.25 |
| E5 | PASS | 0 excluded |
| E6 | reported | summed regret 0.92 percent (FLOPs 0.95); per-cell mean 3.9 percent (FLOPs 5.2); wrong choices 12 vs 17 of 112; worst cell 88 percent for both |
| E7 | PASS | 0 contaminated, 0 implausible |
| E8 | FAIL | M3 point rule 11 of 14 |
| E9 | reported | M3 held-out median APE 0.146 (M1 0.081) |
| E10 | PASS | r*(B=1)/d 0.286 at 1024 (at most 0.30), 0.496 at 4096 (at least 0.45) |

**H2a' on a100-sxm4-40gb: NOT SUPPORTED** (E2, E3, E4, E8 fail). No
criterion was amended after the data were seen.

Fitted coefficients, M1 on d in {512, 2048}: 1.170 mJ per command (1.168 to
1.173), 40.0 pJ per MAC (39.9 to 40.1), 0.741 nJ per HBM word (0.738 to
0.744); P0 55.7 pJ per MAC. M3 (registered for this platform): 13.2
microseconds per command, 1071 GB/s, 7.37 TMAC/s (14.7 TFLOPS), floor 100.8 W,
memory-bound 362.4 W, compute-bound 364.5 W, transition size 1883;
calibration median APE 0.132 (energy), 0.192 (time).

Measured crossover fraction r*/d (rows d, columns B = 1, 4, 16, 64, 256, 1024, 4096):
512: dense always x4, 0.228, 0.446, 0.468;
1024: 0.286, dense always, dense always, 0.203, 0.481, 0.438, 0.444;
2048: 0.157, 0.334, 0.307, 0.378, 0.403, 0.523, 0.521;
4096: 0.496, 0.456, 0.458, 0.475, 0.503, 0.444, 0.521.
FLOPs (0.5 everywhere) is wrong at B = 1 by: dense always at 512 (no finite
factor), 1.75x at 1024, 3.2x at 2048, 1.01x at 4096.

**Predictions A1 to A7 (filed 2026-09-18, tag `exp-002-a100-predictions`), scored as filed:**

- A1 HELD as registered (0.286 at most 0.30 at d = 1024; 0.496 at least 0.45
  at d = 4096). The monotone wording "rises with d" does not hold across the
  four sizes: the B = 1 column is dense always, 0.286, 0.157, 0.496.
- A2 HELD: M3 transition size 1883 (filed range 1400 to 2800); r*/d at
  (2048, 1) = 0.157 < 0.40; at (4096, 1) = 0.496 >= 0.45.
- A3 FAILED: at B = 4096, r*/d = 0.468, 0.444, 0.521, 0.521; at d >= 2048
  the factorized form at r = d/2 costs 4 to 5 percent LESS than dense
  (dense 2048^3 kernel at the 400 W cap at 8.6 TFLOPS; at 4096 dense 19.0
  TFLOPS at 381 W vs factorized at 353 W).
- A4 FAILED on the middle clause: dense always at (512, 1) (clause met);
  1.75x at (1024, 1) (clause requires at least 2x); 1.01x at (4096, 1)
  (clause met).
- A5 FAILED on the MAC term: 1.17 mJ per command (filed 0.5 to 2.0, met);
  40.0 pJ per MAC (filed 5 to 10, failed); 0.74 nJ per word (filed 0.3 to
  1.0, met). The filed MAC range ignored the datasheet: 400 W over 9.75
  TMAC/s FP32 is 41 pJ per MAC. The laptop's 175 W over 11.8 TMAC/s is 14.8
  pJ (fit 15.3). On both platforms a_c is power over FP32 throughput within
  4 percent.
- A6 FAILED on coverage: median APE 0.081 (at most 0.20, met); 11 of 14
  within a factor of 1.5 (at least 12 required).
- A7 HELD: counter passed C1 to C3 in EXP-001 (C3 0.0101) and is the
  platform's energy source; EXP-002 confirms it window by window.

Score: 3 of 7 held as filed (A1, A2, A7).

### Observations (a100-sxm4-40gb)

1. The B = 1 column is set by cuBLAS kernel selection, not by bytes alone.
   d = 512: dense is one `gemvNSP` kernel, 25.9 us at 82 W (the gate's
   launch floor is 9.9 us, so the kernel's own latency is the floor);
   factorized is two such kernels, 38.7 us at 77 W; ratio 1.39 at every
   rank. d = 1024: dense becomes split-K GEMV plus reduce (2 commands,
   30.8 us at 130 W, 4.0 mJ) while factorized at small ranks runs three
   kernels at floor power (30.6 to 43.8 us at 78 to 94 W, 2.9 to 3.4 mJ),
   so factorized is 16 to 28 percent cheaper and r* = 0.286 d. d = 2048:
   dense is one memory-bound `gemvx` kernel (16.8 us at 262 W, 1.0 TB/s,
   4.4 mJ); factorized costs 3.4 to 3.7 mJ in three kernels for f <= 1/8,
   then 6.1 and 8.8 mJ at f = 1/4 and 3/8 (four-kernel path, 60.8 us at
   101 to 145 W), then 5.4 mJ in two kernels at f = 1/2 and 7.0 mJ at 3/4;
   the ratio is not monotone in r and r* = 0.157 d marks the kernel
   boundary. d = 4096: dense `gemvx` plus reduce, 58.5 us at 287 W (1.15
   TB/s, 16.7 mJ); factorized at f = 1/16 4.5 mJ at 103 W (ratio 0.27 vs
   FLOPs 0.125, latency-bound at floor power); the ratio meets the FLOPs
   line at f = 1/4; r* = 0.496 d.
2. At B = 4096 all sizes follow the FLOPs line; at d >= 2048 the
   kernel-shape penalty has the opposite sign from the laptop's (factorized
   at f = 1/2 cheaper by 4 to 5 percent), which is the failure of A3.
3. M1 with measured command counts predicts held-out energy better on the
   A100 (8.1 percent) than on the laptop with declared counts (12.4
   percent). Its worst cells are again the small-rank factorized calls at
   small batch (d = 1024, B = 4 and 16: 62 percent over), where three or
   four kernels at floor power cost less than the additive per-command
   energy.
4. Coverage misses (point rule) are all three in the kernel-selection
   corner d = 1024, B <= 16; all seven d = 4096 cells and (1024, B >= 64)
   hit.
5. The A100's coefficients against the laptop's: command 1.8x (1.17 vs
   0.64 mJ; floor power 101 vs 46 W in the M3 fits), MAC 2.6x (40.0 vs
   15.3 pJ; power over FP32 throughput on both), word 0.72x (0.74 vs 1.03
   nJ; HBM2e vs GDDR6). A tiny call costs nearly the same energy on both
   (EXP-001).
6. Correction to the 2026-09-18 addendum: the leave-one-cell-out
   cross-validation score for the laptop run was recorded there as 0.331
   (M1) versus 0.338 (M2); the committed `analysis.json` of run
   `20260915T180708`, reproduced on 2026-10-30, holds 0.152 versus 0.160
   (same selection, M1). The file of record is the analysis output; the
   earlier figure was from a superseded execution.

### Interpretation (a100-sxm4-40gb)

The structure the hypotheses were built on survives: the FLOPs rule fails
at small d (by 1.75x to no finite factor at B = 1 for d <= 2048) and holds
at large d and large B; the transition size from the launch floor and the
bandwidth lands where filed; the three-term additive model calibrated on
two shapes predicts the held-out grid at 8.1 percent and chooses better
than FLOPs (12 vs 17 wrong of 112). What does not survive is H2a' as a
smooth function of d with the dispatch floor as the only small-size
mechanism: on the A100 the library's kernel choice at each shape sets the
floor time, the power, and the command count, and the crossover map
inherits its discontinuities. The measured census is what makes the model
work there, and it marks the laptop's declared counts (one per GEMM) as a
lower bound. Reported in the MLSys 2027 submission as it stands, failures
included.

### Artifacts (a100-sxm4-40gb)

- `experiments/exp_002_lowrank-crossover/a100-sxm4-40gb/20261030T001246/`
  (all results files, `samples/*.csv.gz`, `logs/run.log`; analysis.json
  committed with the run and reproduced independently)
- `paper/figures/*_a100.pdf`, `paper/figures/*_both.pdf`,
  `paper/figures/heldout_scatter_row.pdf` from `scripts/paper_figures.py`

## Addendum 2026-10-30: MLSys 2027 submission filed

Submitted to MLSys 2027 on 2026-10-30, about 15:20 ET (deadline 16:00 ET,
20:00 UTC), on OpenReview (venue MLSys.org/2027/Conference), double blind.

- Title: "When FLOPs Point the Wrong Way: Pre-Registered Energy Crossovers
  for Low-Rank Inference on GPUs".
- Main paper: `paper/main.tex` built on Overleaf with `\def\NOAPPENDIX{1}`
  (mlsys2025 style, review mode). 12 pages: body on pages 1 to 10,
  references on 11, AI usage disclosure on 12. All fonts Type 1. Author
  field "Anonymous Authors". The only self-reference is the third-person
  citation of the FLAIRS TOML paper.
- Supplementary material: `paper/supplementary.tex` (the appendix compiled
  on its own), 2 pages, with the anonymized repository link
  https://anonymous.4open.science/r/mlpowermeter-1FA5/ (mirror of this
  repository at the submission state).
- Data in the paper: laptop run `20260915T180708` and A100 run
  `20261030T001246`, both analyses reproduced independently before
  submission (see the addendum above). Hypotheses and predictions reported
  as filed: H2a refuted on the laptop, H2a' not supported on the A100 under
  criteria v3, predictions A1, A2, A7 held, A3 to A6 failed.

Two upload rounds were needed, both before the deadline:

1. The first main PDF had four unresolved cross-references printed as
   "??" (three `\ref`s from the body to appendix labels, which do not
   exist when the appendix is built separately, and one from the appendix
   to the main gate table). Fixed by replacing them with plain text
   ("Supplementary Figure 1", "Supplementary Figure 2", "in the
   supplement", "Table 1 of the main paper") and shortening one clause so
   the body still ends on page 10.
2. The rebuilt main PDF came out at 14 pages because `\NOAPPENDIX` was
   commented out in the copied `main.tex`, so the appendix was compiled
   into the main file; the CFP requires the appendix as a separate upload.
   Rebuilt with the define uncommented (12 pages) and re-uploaded.

Lesson for the next submission: build the main paper and the supplement
from one checklist that includes `grep '??'` on the PDF text, the page
count, and the presence or absence of the appendix, and run it on the
exact files to be uploaded, not on a local build.

Files changed for the submission that remain to be committed at the time
of writing: `paper/main.tex`, `paper/appendix.tex`, `paper/supplementary.tex`
(mirror link), `paper/SUBMISSION_FORM.md`, `paper/figures/` (Type 1
rebuilds), `README.md`, and this logbook.
