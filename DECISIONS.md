# Decision Log

Dated record of scope, protocol, and modeling decisions. A decision that
changes how confirmatory data will be collected or analyzed must be entered
here before the affected data is collected. Superseded decisions stay in place
with a pointer to the decision that replaced them.

Format: `D-NNN (YYYY-MM-DD) Title`, then Decision, Rationale, Consequences,
Supersedes (if any).

---

## D-001 (2026-09-15) Paper framing: catalog backbone, crossover-prediction spine, RL agents as a family, RL controller gated

**Decision.** The study is organized as follows.

1. Empirical backbone: a catalog of the major ML algorithm families (linear and
   kernel models, tree ensembles, clustering and decomposition, MLP, CNN,
   recurrent, transformer, graph, and reinforcement learning agents), training
   and inference, measured on two GPU platforms with a device-calibrated
   four-term TOML energy model.
2. Scientific spine: a short list of pre-registered crossover predictions that
   FLOPs-based reasoning cannot make (dense versus low-rank, dense versus
   sparse, dispatch-dominated small-batch execution, GPU versus CPU),
   registered before the held-out platform is measured.
3. Reinforcement learning agents are a first-class algorithm family in the
   catalog, chosen because their small-batch, environment-stepping execution is
   the regime where dispatch overhead dominates.
4. The RL energy controller described in the blueprint (docs/blueprint,
   section 11) is a gated extension. It enters the paper only if it beats a
   tuned fixed schedule, a contextual bandit, a greedy progress-per-joule
   controller, and model-predictive control under matched budgets (blueprint
   gate G4). Otherwise the characterization result stands alone.

**Rationale.** The blueprint's rigor is adopted in full; its breadth as written
(six hypotheses, seven experiment classes, eight predictor baselines, nine
controller baselines) is a multi-year program. A best-paper-competitive
submission needs one memorable falsifiable finding with airtight evidence.
Breadth becomes the evidence; the crossover predictions are the claim.

**Consequences.** Repository and package are organized around algorithm
families and measurement; the controller code, if built, lives in its own
subpackage and its experiments carry their own gate. The blueprint remains the
reference for claim discipline, measurement protocol, and statistical design.

---

## D-002 (2026-09-15) Repository and package naming

**Decision.** Public GitHub repository `jemsbhai/mlpowermeter`. Python package
`tomlml` under `src/`. Local working folder renamed from `TOMLICLR` to
`mlpowermeter` to match the repository.

**Rationale.** A bare `toml` name collides with the TOML configuration-format
package on PyPI. The repository name is provisional and may change before
submission; the package name is expected to persist. The earlier
`nexus-ml-metrics` PyPI package (E:\data\code\claudecode\nexus-ml) is
rudimentary and is not a dependency; PyPI publication of `tomlml` is deferred
until after the experiments are running.

**Consequences.** ICLR review is double-blind. The public repository is the
working repository; an anonymized mirror will be prepared at submission time.
Nothing in commit messages, configs, or results files should be required to
change for that mirror beyond author identifiers.

---

## D-003 (2026-09-15) Measurement primitives

**Decision.**

1. Primary energy quantity: NVML cumulative energy counter
   (`nvmlDeviceGetTotalEnergyConsumption`, millijoules) read at
   CUDA-synchronized measurement boundaries.
2. Secondary: 50 ms power sampling (`nvmlDeviceGetPowerUsage`) integrated over
   the same window, reported alongside the counter as a same-sensor
   consistency check, never as independent validation.
3. Device resolution: the torch CUDA device is mapped to its NVML handle by
   UUID; the mapping is verified (UUIDs compared) and written to the manifest.
   Index-based fallback is permitted only when the UUID route is unavailable,
   and is flagged in the manifest.
4. Gross device energy is the main operational measure. Idle-subtracted dynamic
   energy is a sensitivity analysis; the idle measurement method and its
   uncertainty are logged.
5. Isolation: running compute processes on the target GPU are recorded before
   and after every measurement window. A window with foreign processes is
   excluded and its exclusion logged.
6. Clock and power-cap control are not exercised by default. Operating state
   (SM and memory clocks, temperature, throttle reasons, enforced power limit
   where readable) is measured and recorded as part of the operating context.

**Rationale.** The TOMLSignals stack measured mean power times duration. The
counter integrates on-device at the sensor's native rate and does not depend on
host sampling jitter. Blueprint section 13 requires the isolation and thermal
checks and explicitly warns that two APIs reading one sensor are not two
measurements.

**Consequences.** EXP-001 must establish, per platform, that the counter is
supported and monotonic, its update period, agreement with power integration,
idle drift, thermal settling time, and the minimum measurement window.

---

## D-004 (2026-09-15) Reference software stack

**Decision.** Pinned to the laptop environment on this date: Python 3.12.2,
torch 2.6.0 with CUDA 12.4 wheels (cuDNN 9.1.0), numpy 2.0.2, scipy 1.17.1,
scikit-learn 1.5.2, nvidia-ml-py 13.595.45, driver 595.79 on the laptop. The
cluster installs the same torch build. Driver and OS necessarily differ and
are recorded in `environment.json` for every run.

**Rationale.** Blueprint section 12: match software across platforms where
possible so that hardware transfer can be isolated from deployment-platform
transfer.

**Consequences.** Any package upgrade during the study is a new decision entry
and a new experiment ID for anything re-measured.

---

## D-005 (2026-09-15) TO cost table inherited as the FP32 baseline

**Decision.** The transistor-operation cost table from FLAIRS-39 and MLSP 2026
(MAC 5,000; division and square root 15,000; exp, sin, cos, sigmoid 18,000;
tanh 15,000; GELU 20,000; softmax 25,000 per element; ReLU 100; comparison 50;
SRAM word 192; HBM word 10,000) is inherited unchanged as the FP32 baseline.
Precision-specific costs (FP16, BF16, TF32, INT8) and any structural additions
require a dated decision here and, if they change a previously reported
number, an erratum in findings.md.

**Rationale.** Consistency across the program's papers; the table is the
published calibration anchor.

---

## D-006 (2026-09-15) Gate G0 before any pilot science

**Decision.** No pilot or confirmatory measurement begins on a platform until
EXP-001 (instrumentation validation) has passed on that platform against its
pre-registered criteria. Failure means repairing the measurement code and
re-running under a new experiment ID.

**Rationale.** Blueprint gate G0. Every downstream claim inherits the
measurement's validity.

---

## D-007 (2026-09-15) Deadlines do not scope the study

**Decision.** Venue deadlines are not used to gate or cut scope. Scope
decisions are made on evidence and the gates above.

**Rationale.** Standing rule for the program.

---

## D-008 (2026-09-15) Deterministic algorithm selection is not forced in measured workloads

**Decision.** `set_all_seeds` seeds Python, NumPy, and torch (CPU and CUDA)
from one master seed but does not enable
`torch.use_deterministic_algorithms(True)` by default. Runs that need bitwise
reproducible outputs (not energy) opt in with `deterministic: true` in their
config, and the flag's state is recorded in `seed.json` and
`environment.json` for every run.

**Rationale.** Forcing deterministic algorithms changes which kernels run,
and therefore the physical work being measured. The quantity under study is
the energy of the algorithm as it is normally executed. Reproducibility of
energy results comes from repeated measurement with reported uncertainty,
not from bitwise-identical outputs.

**Consequences.** Any experiment that compares outputs numerically (exact
function tests, such as dense versus factorized maps) declares its numerical
tolerance in the logbook entry rather than relying on determinism.

---

## D-009 (2026-09-15) Energy source is selected per platform by the instrumentation gate

**Decision.**

1. Every measurement window records both quantities: the cumulative-counter
   delta (`energy_counter_j`) and the trapezoid integral of the 50 ms power
   readings (`energy_power_integral_j`). Neither is discarded.
2. EXP-001 qualifies the counter per platform with three criteria: C1
   (supported and monotonic in every window), C2 (median update interval at
   most 200 ms under load), C3 (agreement with the power integral within 5
   percent on every repeatability block). A platform uses the counter as its
   energy source only if all three pass; otherwise it uses the power integral.
   C1 to C3 are reported but do not gate; C4 (repeatability) and C7 (window
   sufficiency) are evaluated on the selected source and do gate.
3. The selection is written into the run's `summary.json` (`energy_source`)
   and, after the full run, into a per-platform config override that every
   later experiment on that platform inherits. The paper's methods section
   states the source per platform.
4. Power readings above a plausibility ceiling (twice the device's maximum
   power-limit constraint from the capability probe) are kept in the raw
   sample log but masked from power statistics and the integral, and
   counted. The masked fraction is gate criterion C10 (at most 1 percent).
5. The closing counter value of a window is read after the sampler thread
   has stopped, so the counter series of a window is monotone by
   construction if the hardware counter is.

**Rationale.** The rtx4090-laptop quick run of EXP-001 (run
`20260915T161501`, committed with this decision) showed the NVML counter
advancing at about 500 W-equivalent while the GPU idled at 4 W and 210 MHz,
about 250 W-equivalent under a 175 W power-capped GEMM load, and about 164
W-equivalent for the dispatch-bound loop whose power reading was 63 W.
Whatever that counter integrates on this platform, it is not this GPU's
energy in millijoules, and it is not off by a constant factor either. The
power reading was physically consistent throughout (4 W idle, 175 W pinned at
the cap with `SwPowerCap`, 63 W for the dispatch-bound loop). The power
integral is also the method behind the published TOMLSignals results on both
GPUs, so using it where the counter fails keeps the study's measurements
comparable with the program's prior work. The same run produced one 593.5 W
power reading (the first after initialization) and one spurious
non-monotonic counter series caused by the read ordering in `stop()`.

**Consequences.** EXP-001 protocol v2 (LOGBOOK.md addendum of 2026-09-15),
tagged `exp-001-protocol-v2`; all full runs execute under v2. Supersedes the
wording of D-003 item 1 ("primary energy quantity: the counter"); D-003
items 2 to 6 stand. The A100 gets its own verdict from its own full run.

---

## D-010 (2026-09-15) Measurement protocol for all experiments after the gate

**Decision.**

1. Measurement windows are 20 s by default. A 10 s window is permitted only
   for workloads pinned at the power cap, and every reported number states
   which regime it was measured in (the window's `throttle_reasons` and mean
   power against the enforced limit decide).
2. Before each block sequence, the workload is run until the GPU
   temperature is stable under it (spread at most 1 C over 5 s, timeout
   300 s, the same criterion as the idle settle), not for a fixed time. Each
   measured window is additionally preceded by 2 s of the same configuration
   unmeasured.
3. Both energy quantities are recorded in every window; the platform overlay
   in `configs/platform/<tag>.yaml` selects the energy source (D-009).
4. GPU command counts (the dispatch term's S_o) are measured with
   `torch.profiler` (CUPTI) per configuration on both platforms: one profiled
   call after warmup, counting kernels, memcpy, and memset, with kernel names
   recorded. On the laptop a subset of configurations is cross-checked once
   against Nsight Systems (the TOMLSignals method, F-022) and the comparison
   is logged.
5. Inference microbenchmarks cycle through enough independent weight sets
   that the total weight footprint is at least twice the device's L2 cache
   (from torch device properties), so that per-call weight reads are not
   served from L2 in a way a multi-layer model would not enjoy. The count and
   the L2 size are recorded per run.
6. Exact-function pairs declare a numerical tolerance before running; for
   FP32 the default is a relative Frobenius difference of at most 1e-4 on
   the output. A configuration that violates it is excluded with its reason
   logged, never silently.
7. Condition order within a block sequence is randomized with the run seed;
   the order is written to the results.

**Rationale.** Items 1, 2 and 3 are the EXP-001 laptop findings (dispatch-
bound work needs 20 s for 2 percent CV; a fixed 30 s warmup did not reach
steady state after a change of load level; the counter is not usable there).
Item 4 makes the launch census portable to the cluster while keeping the
validated method as a check. Item 5 keeps the memory term honest for weights
that would otherwise sit in a 40 to 64 MB L2. Items 6 and 7 are blueprint
sections 9 and 14.

**Consequences.** Implemented as shared protocol helpers used by every
experiment module from EXP-002 on; EXP-001 stays as run (fixed warmups, its
protocol said so).

---

## D-011 (2026-09-15) Study design adopted as canon

**Decision.** `docs/STUDY_DESIGN.md` is the pre-registration: platforms and
the held-out-platform protocol, the Tier 1 and Tier 2 algorithm catalog,
the primitive experiments EXP-002 to EXP-007, tasks, predictors P0 to P5,
research questions RQ1 to RQ5 with hypotheses and refutation conditions,
gates G0 to G6, the compute plan, and the claim discipline. The blueprint
remains the reference for measurement and statistical protocol; where the
two differ, the study design governs.

**Rationale.** Framing C (D-001) needed its concrete object list and its
falsifiable statements written down before the first spine experiment.

**Consequences.** Numeric thresholds are provisional until a dated freeze
entry after EXP-002 to EXP-004. Changes to the catalog or hypotheses are
dated entries here, before the affected data is collected. A100 predictions
for each spine experiment are filed in LOGBOOK.md with a commit hash before
the corresponding job is submitted.
