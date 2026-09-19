# Findings

Two sections. The curated summary at the top is a living results section,
organized thematically and refined as understanding evolves (history in git).
The raw findings log at the bottom is chronological and append-only.

Rules:

- Every number carries an experiment ID (EXP-NNN). No number appears here
  without one.
- Raw entries are never edited. Corrections are dated errata that reference the
  original entry.
- Negative results are recorded in both sections.
- Commits: `docs(findings): add EXP-NNN results` or
  `docs(findings): update curated summary`.

## Curated summary

### Measurement (gate G0)

- rtx4090-laptop passed gate G0 under protocol v2 (EXP-001, run
  `20260915T164645`) with the power integral as its energy source. The NVML
  cumulative energy counter on this platform is monotonic and updates every
  100 ms but does not measure the GPU's energy (about 500 W-equivalent at a
  4.4 W idle); it is recorded and never used there (D-009). The power reading
  updates every 500 ms.
- Two measurement regimes with different precision requirements on the
  laptop: power-capped compute-bound work (energy per call is a timing
  measurement; CV 0.3% at 10 s, 0.2% at 0.5 s) and dispatch-bound work near
  60 W at boost clock (CV 2.0% at 10 s, 1.4% at 20 s; the variance is
  CPU-side dispatch timing). Twenty-second windows are the default for
  anything not pinned at the cap (EXP-001).
- Under the cap, energy per call drifts with temperature through the SM
  clock (+0.9% over 100 s while 73 to 79 C, 1711 to 1689 MHz). Hot-idle bias
  after a temperature settle is negligible (+0.006 W). A fixed warmup does
  not reach steady state after a change of load level; warm up until the
  temperature is stable under the workload (EXP-001).
- `torch.profiler` with the CUDA activity recorded no device events on the
  Windows laptop in EXP-002 (attribution pending); the census must report
  unknown, never zero, and the laptop's command counts are declared (one
  per GEMM) until the Nsight Systems cross-check confirms them (D-012).
- a100-sxm4-40gb: pending its own gate run.

### Low-rank crossover (EXP-002, rtx4090-laptop, declared command counts)

- H2a as registered is refuted on the laptop. The crossover rank fraction
  at B = 1 rises with map size d: 0.021, 0.198, 0.461, 0.497 of d for d =
  512, 1024, 2048, 4096, because the dense GEMV moves from dispatch bound
  (14 microseconds per call at 64 W) to memory bound (64 MB in 123
  microseconds at 520 GB/s, 149 W). At fixed d it rises with batch only
  where dispatch matters (d at or below 1024) and is flat near d/2 above.
  FLOPs (r* = d/2 always) is wrong by a factor of 24 at (512, 1), 2.5 at
  (1024, 1), and right within 10 percent at d at or above 2048.
- The additive TOML model with one command per GEMM, fitted on d in {512,
  2048}, predicts held-out (1024, 4096) energy at 12.4 percent median error
  (P0 FLOPs: 60 percent), the B = 1 crossovers within 3 percent (206 versus
  202 at 1024; 1972 versus 2034 at 4096), and 13 of 14 held-out cells
  within a factor of 1.5. Fitted costs: 0.64 mJ per command, 15.3 pJ per
  MAC, 1.03 nJ per HBM word.
- Dispatch overhead is a time floor at reduced power, not a constant energy:
  two launches at d = 512, B = 1 take 1.8x the time of one at half the
  power (33 W, SM downclocked to 1455 MHz) for equal energy. An additive
  per-command energy overpredicts those cells by about 50 percent; the
  max-of-times model (M3, 16.6 microseconds per command, 443 GB/s, 46 W
  floor, transition size 1358) captures the regimes but predicts energy
  worse (24 percent) because cuBLAS throughput is shape dependent (2 to 24
  TFLOPS across the grid).
- Dense GEMM energy is independent of the weight data to within 2 percent
  across eight weight sets of different rank; repetition CV of energy per
  call over 448 configurations: median 0.76 percent, p90 4.7 percent.
- The A100 run is the prospective test of the restated H2a' (predictions
  A1 to A7 filed 2026-09-18 in LOGBOOK.md).

## Inherited prior results (context only, not confirmatory for this study)

These results come from the earlier TOML papers and their repositories. They
motivate the design but are not re-validated here. Any use in the paper is
labeled retrospective and cited to the original work.

- TO cost table (FLAIRS-39, MLSP 2026): MAC 5,000; division 15,000;
  square root 15,000; exp 18,000; HBM word 10,000; SRAM word 192; comparison 50
  transistor operations at the 45 nm reference node, derived from Horowitz
  (ISSCC 2014). The table is FP32 only. Precision-specific extensions are a
  new-decision item for this study (see DECISIONS.md, D-005).
- Four-term energy model from TOMLSignals: compute, memory, command dispatch
  overhead (alpha_o), and fused sequential execution (alpha_f) are physically
  distinct mechanisms. alpha_o is CPU-side dispatch dominated; alpha_f is
  GPU-side serial execution at low SM utilization. Their ratio differs across
  GPUs (about 14x on the RTX 4090 versus about 1x on the A100 in that study).
- Command count S_o must be measured (kernel launches plus memcpy plus memset
  per invocation, via Nsight Systems), not derived from source inspection.
- Sequential algorithms favor the CPU; parallel algorithms favor the GPU; this
  regime distinction is invisible to FLOPs-based analysis. That observation is
  the seed of this study's crossover hypotheses.

## Raw findings log

### 2026-09-18 -- EXP-002: Exact low-rank crossover, rtx4090-laptop (run 20260915T180708)

**Key result:** H2a NOT supported as registered (E2, E3, E4 fail under
criteria v2); the crossover rank at B = 1 rises with d from 0.021 d (512)
to 0.497 d (4096); the additive TOML model with declared command counts
predicts held-out energy at 12.4 percent median error and the B = 1
crossovers within 3 percent.

**Details:**
- 1344 windows of 20 s, 3 repetitions, 224 configurations x 2
  realizations; 0 excluded, 0 contaminated; repetition CV median 0.76
  percent. Run interrupted once (external kill, cause not established) and
  resumed with no loss.
- Census failure: profiler saw no device events; commands recorded as 0;
  first analysis (2026-09-16) inverted by it (E1 fail 23.4 percent);
  corrected analysis uses one command per GEMM, labeled declared.
- Criteria v2: E1 pass 0.124; E2 fail 3 of 14 (point rule 13 of 14); E3
  fail at 4096 (measured 0.497 d); E4 fail (0.79 at 1024, -0.04 at 4096);
  E5, E7 pass; E6 0.06 percent (summed); E8 11 of 14, E9 fail, E10 pass
  (exploratory).
- Coefficients (M1): 0.637 mJ per command, 15.3 pJ per MAC, 1.03 nJ per
  HBM word. M3 (joint energy and time): 16.6 microseconds per command, 443
  GB/s, 17.4 TFLOPS effective, 46 W floor, d_t = 1358.
- Per-call regimes: d = 512, B = 1 dense 14.0 microseconds at 64 W;
  factorized (two launches) 25.4 microseconds at 33 W, 1455 MHz; d = 4096,
  B = 1 dense 123 microseconds at 149 W (520 GB/s); B = 4096 at the 175 W
  cap, 23.6 TFLOPS.

**Statistical tests:** relative-weighted NNLS fits; leave-one-cell-out
selection (M1 0.331 versus M2 0.338); 1000-resample bootstrap intervals on
both sides of E2; Spearman for E4.

**Notes:** H2a restated as H2a' with d in it (D-012); A100 predictions A1
to A7 filed before the A100 run. The nsys cross-check of the declared
counts is pending.

### 2026-09-15 -- EXP-001: Instrumentation gate, rtx4090-laptop (run 20260915T164645)

**Key result:** Gate G0 passed on rtx4090-laptop with the power integral as
the energy source; the NVML energy counter was rejected (counter versus
integral max abs relative difference 1.996 against a 0.05 threshold).

**Details:**
- Counter update period: 99.97 ms median under load; power reading update
  period: 500 ms (both from 10 s of tight polling, 822 Hz).
- Idle: 4.396 W (std 0.177) pre-heat, 4.401 W post-heat; hot-idle bias
  +0.006 W; idle drift about -0.6 W/min with the temperature criterion met.
- Thermal settle after a 60 s, 172 W heat load: 16 s (criterion 1 C over 5 s).
- matmul_large (FP32 4096x4096 GEMM, 175 W cap): 1.0147 J/call, CV 0.32%
  over 10 x 10 s blocks; monotone rise 1.0090 to 1.0177 J/call with
  temperature 73 to 79 C and SM clock 1711 to 1689 MHz.
- matmul_tiny_loop (FP32 64x64 GEMM per call): 1.107 mJ/call, CV 1.99% over
  10 blocks of about 9.5 s, at 60 W and 2325 MHz, while cooling 64 to 57 C.
- Window sufficiency (integral, CV at or below 2% for the length and all
  longer lengths): matmul_large 0.5 s (power-capped, degenerate);
  matmul_tiny_loop 20 s.
- Isolation clean; zero implausible power samples in 14877.

**Statistical tests:** none (descriptive gate; 10 blocks per workload, 5
repetitions per window length).

**Notes:** The 0.5 s sufficiency for the capped workload is an artifact of
the cap (the integral of a pinned 175 W is exact); it does not transfer to
uncapped work. Quick runs `20260915T161501` (protocol v1 code, counter
anomaly first seen, race, 593.5 W first power reading) and `20260915T164421`
(v2 code) are committed as data; neither is a gate result.

### 2026-09-15 -- EXP-001 quick run: counter anomaly on rtx4090-laptop (run 20260915T161501)

**Key result:** `nvmlDeviceGetTotalEnergyConsumption` on the RTX 4090
Laptop GPU (Windows, driver 595.79) advanced by 1388 J over a 2.75 s idle
window at 4 W (about 500 W-equivalent) and by about 250 W-equivalent under
a 175 W power-capped load; the ratio to the power reading is not constant.

**Details:** raw series in that run's `samples/idle_pre.csv` and
`samples/repeatability_matmul_large.csv`. First NVML power reading after
initialization: 593.5 W (`samples/resolution_idle.csv`).

**Notes:** Motivated DECISIONS.md D-009 (per-platform energy source) and the
plausibility guard. Not a gate result.
