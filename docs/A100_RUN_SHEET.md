# A100 run sheet: EXP-001 gate, then EXP-002 crossover

For: Ameera, on the a100-sxm4-40gb SLURM cluster.
Deadline context: these runs feed the MLSys 2027 submission of the crossover
paper, due Friday Oct 30, 2026, 4:00 PM ET. EXP-002 data must be back by
Thursday Oct 29 evening ET. The full EXP-002 run takes about 9 hours, so it
needs to start Wednesday night.

The A100 predictions for EXP-002 were filed in LOGBOOK.md on 2026-09-18
(addendum, tag `exp-002-a100-predictions`, criteria v3), before any A100
measurement. Nothing about the run waits on Muntaser. This run sheet is the
go-ahead the sbatch header asks for.

## 0. Pull and set up

```bash
git clone https://github.com/jemsbhai/mlpowermeter.git   # or: git pull on an existing clone
cd mlpowermeter
python3 --version                                        # 3.10 or newer
python3 -m venv ~/venvs/tomlml
source ~/venvs/tomlml/bin/activate
python -m pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
pytest tests -v                                          # all tests must pass; no GPU needed
```

If torch 2.6.0 is unavailable for the cluster's Python, stop and report the
Python version rather than substituting. Full setup detail:
docs/QUICKSTART_A100.md.

## 1. EXP-001, the instrumentation gate (run first)

Edit `scripts/slurm/exp_001.sbatch` once: set `--partition` (and `--account`
if the cluster requires one). Keep `--exclusive` if the scheduler accepts it;
if the job is rejected because of it, remove that line and note it when
reporting.

```bash
git status                                               # must be clean; the runner refuses a dirty tree for full runs
sbatch scripts/slurm/exp_001.sbatch                      # full run, 15 to 25 minutes
```

Optional pipeline check first (1 to 2 minutes, not valid for the gate):
`sbatch scripts/slurm/exp_001.sbatch --quick`.

The end of `slurm-<jobid>.out` prints PASS/FAIL per criterion and GATE PASS or
GATE FAIL. Either outcome is a result. Do not re-run to chase a pass. A
rejected NVML energy counter is not a failure: the gate then evaluates the
power integral (that is what happened on the laptop, DECISIONS.md D-009).

Report from EXP-001: GATE PASS or FAIL, the run id, the energy source line,
the driver version, whether persistence mode was on, whether `--exclusive`
was granted, any other users' processes listed under `isolation` in
`results/summary.json`, and any `throttle` line with a reason other than
SwPowerCap.

Do not start EXP-002 until EXP-001 finishes. If the gate FAILs, still send
the output back (step 3) and wait for Muntaser before running EXP-002.

## 2. EXP-002, the low-rank crossover (about 9 hours)

Same one-time edit in `scripts/slurm/exp_002.sbatch`: `--partition` (and
`--account` if required). The script requests 12 hours of wall time.

```bash
sbatch scripts/slurm/exp_002.sbatch --quick              # about 3 minutes, pipeline check
sbatch scripts/slurm/exp_002.sbatch                      # full run, about 9 hours
```

Grid (configs/exp_002_lowrank_crossover.yaml): shapes {512, 1024, 2048,
4096}, rank fractions {0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5,
0.75}, batches {1, 4, 16, 64, 256, 1024, 4096}, dense and factorized, 3 reps,
20 s windows. Calibration shapes {512, 2048}; held-out shapes {1024, 4096}.

## 3. Send the results back

Commit each run directory on a branch and open a pull request, following the
QUICKSTART pattern:

```bash
git checkout -b exp-001-a100
git add experiments/exp_001_instrumentation-gate/
git commit -m "data(exp-001): a100 run <run_id>"
git push -u origin exp-001-a100
# after the EXP-002 run, same pattern on a second branch:
git checkout -b exp-002-a100
git add experiments/exp_002_lowrank-crossover/
git commit -m "data(exp-002): a100 run <run_id>"
git push -u origin exp-002-a100
```

The `slurm-<jobid>.out` files are gitignored; attach them to the message to
Muntaser rather than committing them. For EXP-002, report the run id and
confirm the run directory is in the pull request. Analysis
(scripts/analyze_exp_002.py, criteria v3, E1 to E10) runs on Muntaser's side
once the data lands.
