#!/bin/bash
# EXP-001 instrumentation gate on the A100 cluster.
#
# Submit from the repository root:
#     sbatch scripts/slurm/exp_001.sbatch            # full run (15 to 25 min)
#     sbatch scripts/slurm/exp_001.sbatch --quick    # 1 to 2 min pipeline smoke test
#
# Fill in the partition (and account, if the cluster requires one) before the
# first submission. Request the whole node with --exclusive if the scheduler
# allows it: the GPU is exclusive to the job under gres either way, but the
# dispatch-bound workload is sensitive to CPU contention from co-scheduled jobs.
# If --exclusive is refused, remove it and note that in the logbook entry.

#SBATCH --job-name=exp001-g0
#SBATCH --partition=<PARTITION>
##SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Environment: adjust to the cluster's module system if needed, e.g.
#   module load python/3.12
source ~/venvs/tomlml/bin/activate

echo "== host $(hostname) job ${SLURM_JOB_ID:-none} at $(date -Is)"
echo "== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,driver_version,persistence_mode,power.limit,clocks.max.sm --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

python scripts/run_experiment.py --config configs/exp_001_instrumentation.yaml "$@"
