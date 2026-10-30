# MLSys 2027 submission form: paste-ready text

Deadline: 2026-10-30 20:00 UTC (16:00 ET). OpenReview venue: MLSys.org/2027/Conference, research track.

## Title

When FLOPs Point the Wrong Way: Pre-Registered Energy Crossovers for Low-Rank Inference on GPUs

## Abstract (plain text, identical in content to the PDF)

A linear map of exact rank r can be executed as one dense GEMM or as two thin GEMMs through a rank-r intermediate, and operation counting says the factorized form saves energy exactly when r < d/2, for every batch size. We measure where that crossover actually falls on a consumer laptop GPU (RTX 4090 Laptop, 175 W cap) and on a data-center A100, over 224 shape, rank, and batch configurations per platform with 20 s windows and three repetitions (1344 windows per platform). Before measuring, an instrumentation gate found that the NVML cumulative energy counter reports about 500 W-equivalent while the laptop GPU idles at 4 W, and agrees with integrated power within 1 percent on the A100, so the energy source is qualified per platform rather than assumed. At batch size one the measured crossover rank is 0.021d, 0.198d, 0.461d, and 0.497d for d = 512 to 4096: the FLOPs rule is wrong by a factor of 24 where the dense matrix-vector product is dispatch bound and right where it is memory bound, with the transition set by the launch floor and the memory bandwidth. An additive model with three physical terms (multiply-accumulates, memory words, GPU commands) calibrated on two shapes predicts held-out energy with 12.4 percent median error (FLOPs: 59.6 percent) and chooses the wrong realization in 3 of 112 held-out cells against 7 for FLOPs. The hypothesis registered before the run was refuted by the data, restated with the map size in it, and seven quantitative predictions for the A100 were filed with a commit hash before that platform was measured: three held (the rise of the crossover with map size, the transition size, the validity of the counter) and four did not, including a per-MAC energy four times the filed range, and the paper reports why. All code, configurations, raw samples, and the pre-registration trail are released.

## Keywords

energy measurement; GPU energy; FLOPs; low-rank factorization; dispatch overhead; pre-registration; roofline

## Primary area (if the form asks)

Measurement, benchmarking, and tooling for ML systems (alternatives: efficient inference and serving; hardware-efficient ML methods).

## Uploads

- Paper PDF: build of main.tex with `\def\NOAPPENDIX{1}` active (body ends on page 10; references and the AI Usage Disclosure follow).
- Supplementary PDF: build of supplementary.tex (the appendix: gate details, held-out cells, additional figures).

## AI Usage Disclosure

Present in the PDF after the references (section "AI Usage Disclosure"); if the form has its own field, paste that paragraph.

## Anonymized artifact link

Make the mirror at https://anonymous.4open.science from https://github.com/jemsbhai/mlpowermeter at commit 26f7cbd or later (after the README line change). Terms to anonymize (the service replaces them throughout the mirror; the run directories contain cluster paths and author handles):

- Muntaser
- Syed
- Silaghi
- Marius
- Ameera
- aalsulami2019
- jemsbhai
- msyed2011
- Florida Institute of Technology
- Florida Tech
- my.fit.edu
- gmail.com
- mlpowermeter (optional; keeps the repository name out of the mirror)
- TOMLICLR
- home1 (cluster home path prefix in run logs and environment snapshots)
- gpu01 (cluster node name in the EXP-001 logbook addendum)

Leave `TOML`, `tomlml`, and `Transistor Operations for Machine Learning` as they are: the framework is published and is cited in the third person.

Paste the generated anonymous URL into the artifact field; the paper's conclusion says the repository is linked from the submission form, so nothing in the PDF needs to change.

## Author list and quota

Author lists are frozen at the deadline and the Generalized Harmonic Quota Rule (N1 = 2, N10 = 20) applies to every author; run the Author Quota Checker before submitting if anyone on the list has other MLSys 2027 submissions.

## Conflicts

Declare domain conflicts for every author's institution and any recent collaborators (undeclared conflicts can be a desk reject).
