#!/bin/bash
# EXP-002 exact low-rank crossover on the A100 cluster.
#
# Do not submit the full run before the A100 predictions for EXP-002 have
# been filed in LOGBOOK.md (STUDY_DESIGN.md section 2, H3c); Muntaser gives
# the go-ahead. The quick run may be submitted at any time.
#
#     sbatch scripts/slurm/exp_002.sbatch --quick     # about 3 minutes, pipeline check
#     sbatch scripts/slurm/exp_002.sbatch             # about 9 hours

#SBATCH --job-name=exp002-lowrank
#SBATCH --partition=<PARTITION>
##SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source ~/venvs/tomlml/bin/activate

echo "== host $(hostname) job ${SLURM_JOB_ID:-none} at $(date -Is)"
echo "== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,driver_version,persistence_mode,power.limit,clocks.max.sm --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

python scripts/run_experiment.py --config configs/exp_002_lowrank_crossover.yaml "$@"
