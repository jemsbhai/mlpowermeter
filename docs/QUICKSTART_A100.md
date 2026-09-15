# Quickstart: A100 cluster (SLURM)

This walks through cloning the repository, verifying the software stack, and
running EXP-001 (the instrumentation gate) on one A100. Everything the paper
will claim rests on this gate passing on both platforms, so the full run is
worth doing carefully once rather than quickly twice.

Nothing here changes GPU state. The code only reads NVML counters; the one
optional write (a power-limit permission probe) is off in every config.

## 1. Clone and install

On a login node:

```bash
git clone https://github.com/jemsbhai/mlpowermeter.git
cd mlpowermeter
# If the cluster uses environment modules, load a Python 3.10 or newer first, e.g.
#   module avail python; module load python/3.12
python3 --version
python3 -m venv ~/venvs/tomlml
source ~/venvs/tomlml/bin/activate
python -m pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

torch 2.6.0 with CUDA 12.4 wheels is the reference build (DECISIONS.md D-004).
The wheels bundle the CUDA runtime; only the node's driver matters, and
EXP-001 records it. If `torch==2.6.0` is unavailable for the cluster's Python,
stop and report the Python version rather than substituting.

## 2. Run the test suite (no GPU needed)

```bash
pytest tests -v
```

All tests must pass. They exercise the measurement code against a fake NVML
binding, so a login node without a GPU is fine. If anything fails, paste the
output into the group chat before going further.

## 3. Smoke test on a GPU (about 2 minutes)

Get an interactive GPU shell (adjust partition and account):

```bash
srun --partition=<PARTITION> --gres=gpu:1 --cpus-per-task=8 --time=00:20:00 --pty bash
source ~/venvs/tomlml/bin/activate
cd mlpowermeter
nvidia-smi
python scripts/run_experiment.py --config configs/exp_001_instrumentation.yaml --quick
```

The quick run scales every duration down by 20x. It is a pipeline check
only: its results are labeled `quick` and are not valid for the gate. Look
for these lines in the output and note them:

- `device: nvml index N, NVIDIA A100-SXM4-40GB, uuid GPU-..., resolved by torch_uuid, uuid_verified=True`
- `counter supported=True`
- `counter update period under load: median X ms`
- `energy source for this platform: counter` or `power_integral (counter rejected: ...)`
- `GATE ... (QUICK RUN, NOT VALID FOR THE GATE)`

If `uuid_verified` is not `True`, stop and report the full output; that is
exactly the condition this gate exists to find, and the fix belongs in the
code, not in a workaround. A rejected counter is not a failure: on the RTX
4090 laptop the counter turned out not to measure the GPU's energy at all
(DECISIONS.md D-009), and the gate then evaluates the power integral instead.
Either way the log line is worth quoting when you report.

## 4. Full run (15 to 25 minutes)

Edit `scripts/slurm/exp_001.sbatch` once: set `--partition` (and `--account`
if required). Keep `--exclusive` if the scheduler accepts it; if the job is
rejected because of it, remove that line and mention it when reporting. Then:

```bash
git status            # must be clean: the runner refuses a dirty tree for full runs
sbatch scripts/slurm/exp_001.sbatch
squeue -u $USER
```

When it finishes, `slurm-<jobid>.out` ends with a criteria list
(`PASS`/`FAIL` per criterion) and `GATE PASS` or `GATE FAIL`. Either outcome
is a result; do not re-run to chase a pass.

## 5. Send the results back

The run directory is committed to the repository (raw samples included; they
are small). Use a branch and a pull request so nothing lands on `main`
without review:

```bash
git checkout -b exp-001-a100
git add experiments/exp_001_instrumentation-gate/
git commit -m "data(exp-001): a100 run $(ls experiments/exp_001_instrumentation-gate/a100-sxm4-40gb/ | tail -1)"
git push -u origin exp-001-a100
```

Then open a pull request on GitHub and post in the group chat: the run id,
`GATE PASS` or `GATE FAIL`, and the `slurm-<jobid>.out` file (attach it; it
is gitignored). Muntaser records the outcome in `LOGBOOK.md` and merges.

Quick runs can be committed the same way if something interesting happened
(a failure, a warning); otherwise leave them out of the pull request.

## 6. Things to report even if everything passes

- The node's driver version and whether `persistence_mode` was on (both are in
  the summary). Persistence mode off means the first CUDA context creation
  includes driver load time; it is recorded, not a failure.
- Whether `--exclusive` was granted.
- Any other users' processes listed under `isolation` in
  `results/summary.json`.
- Any line in the log mentioning `throttle` with a reason other than
  `SwPowerCap` or an empty list.

## Where things are

```
configs/exp_001_instrumentation.yaml   protocol parameters and pass criteria
LOGBOOK.md                             pre-registered entry EXP-001
experiments/exp_001_instrumentation-gate/a100-sxm4-40gb/<run_id>/
  config.yaml environment.json seed.json manifest.json
  results/capabilities.json results/windows.json results/summary.json
  samples/*.csv logs/run.log
```
