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
| EXP-001 | Instrumentation validation (gate G0) | rtx4090-laptop, a100-sxm4-40gb | pilot | rtx4090-laptop PASS (power integral); a100 pending |
| EXP-002 | Exact low-rank crossover, dense versus factorized GEMM (inference) | rtx4090-laptop, a100-sxm4-40gb | pilot | planned |

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

(filled in per platform after completion)

### Observations

(filled in after completion)

### Interpretation

(filled in after completion)

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

### Results

(filled in per platform after completion)

### Observations

(filled in after completion)

### Interpretation

(filled in after completion)

### Artifacts

- `experiments/exp_002_lowrank-crossover/<platform-tag>/<run-id>/`
  (`results/windows.json`, `results/census.json`, `results/exactness.json`,
  `results/summary.json`, `results/analysis.json`, `samples/*.csv`,
  `figures/`, `logs/run.log`)
