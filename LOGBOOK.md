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
| EXP-001 | Instrumentation validation (gate G0) | rtx4090-laptop, a100-sxm4-40gb | pilot | planned |

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
