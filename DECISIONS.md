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
