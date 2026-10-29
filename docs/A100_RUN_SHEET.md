# A100 run sheet: EXP-002 crossover (EXP-001 gate already passed)

For: Ameera, on the a100-sxm4-40gb SLURM cluster.
Deadline context: this run feeds the MLSys 2027 submission of the crossover
paper, due Friday Oct 30, 2026, 4:00 PM ET. EXP-002 data must be back by
Thursday Oct 29 evening ET. The full run takes about 9 hours, so it needs to
start Wednesday night.

Status: EXP-001 is DONE on the A100. Ameera's run 20260925T064203 (Sep 25)
passed the gate (gate_pass true, all 11 criteria, energy source: the NVML
counter, C1 to C3 passed as predicted in A7) and was merged in PR #1 on
Sep 26. Only EXP-002 remains.

The A100 predictions for EXP-002 were filed in LOGBOOK.md on 2026-09-18
(addendum, tag `exp-002-a100-predictions`, criteria v3), before any A100
measurement of EXP-002. This run sheet is the go-ahead the sbatch header
asks for.

## Commands

Run from the existing clone (~/mlpowermeter on the cluster):

```bash
cd ~/mlpowermeter
git checkout main && git pull
source ~/venvs/tomlml/bin/activate

# One-time: put your partition (and account, if required) into the sbatch,
# then commit the edit. A full run refuses a dirty tree, so the tree must be
# clean at submit time.
sed -i 's/<PARTITION>/<YOUR PARTITION>/' scripts/slurm/exp_002.sbatch
# If the cluster requires an account, also uncomment and set it:
# sed -i 's/#SBATCH --account=<ACCOUNT>/#SBATCH --account=<YOUR ACCOUNT>/' scripts/slurm/exp_002.sbatch
git add scripts/slurm/exp_002.sbatch
git commit -m "chore: set cluster partition for exp-002 sbatch"
git status    # must be clean

sbatch scripts/slurm/exp_002.sbatch --quick    # about 3 minutes, pipeline check
sbatch scripts/slurm/exp_002.sbatch            # full run, about 9 hours (12 h requested)
squeue -u $USER
```

If `git pull` brings setup changes, re-run the test suite before submitting
(no GPU needed): `pytest tests -v`.

Grid (configs/exp_002_lowrank_crossover.yaml): shapes {512, 1024, 2048,
4096}, rank fractions {0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5,
0.75}, batches {1, 4, 16, 64, 256, 1024, 4096}, dense and factorized, 3 reps,
20 s windows. Calibration shapes {512, 2048}; held-out shapes {1024, 4096}.

## Send the results back

```bash
git checkout -b exp-002-a100
git add experiments/exp_002_lowrank-crossover/
git commit -m "data(exp-002): a100 run $(ls experiments/exp_002_lowrank-crossover/a100-sxm4-40gb/ | tail -1)"
git push -u origin exp-002-a100
```

Then open a pull request against main (same pattern as PR #1). The
`slurm-<jobid>.out` file is gitignored; attach it to the message to Muntaser
instead of committing it. Report the run id when the job finishes. Analysis
(scripts/analyze_exp_002.py, criteria v3, E1 to E10) runs on Muntaser's side
once the data lands.
