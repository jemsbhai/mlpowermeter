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

No confirmatory results yet. The instrumentation gate (G0, EXP-001) is the
current phase; see LOGBOOK.md.

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

(none yet)
