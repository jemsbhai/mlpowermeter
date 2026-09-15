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
- Output directory: `experiments/exp_NNN_name/<platform-tag>/`, containing
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
| (none yet) | | | | |
