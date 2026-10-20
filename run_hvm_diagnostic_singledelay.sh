#!/bin/bash -l
#SBATCH --job-name=hvm_diag_sd
#SBATCH --output=/project/mx6/yh87/Yiran-Movie-Paper/reports/%x.%A_%a.out
#SBATCH --error=/project/mx6/yh87/Yiran-Movie-Paper/reports/%x.%A_%a.err
#SBATCH --partition=general   # CPU only -- sklearn ridge, no GPU needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --qos=standard
#SBATCH --account=mx6
#SBATCH --time=2-00:00:00
#
# Diagnostic re-run (single fixed 4s delay instead of FIR, near-leave-one-
# segment-out outer CV instead of 5 big folds) -- see
# run_hvm_diagnostic_singledelay.py docstring for the full rationale.
#
# One array task = one movie (3 models x 2 measures = 6 combos per task,
# run sequentially inside the python script itself). DM is the heaviest
# (outer_folds=36 leave-one-segment-out folds x 83 subjects), so it gets
# its own task/node rather than being bundled with TP/PC.
#
# Submit as a 3-task array:
#   sbatch --array=0-2 run_hvm_diagnostic_singledelay.sh
#
# task 0 -> tp, task 1 -> dm, task 2 -> pc

set -e
module load Miniforge3
conda activate /project/mx6/yh87/conda_env/movie

cd /project/mx6/yh87/Yiran-Movie-Paper/hvm

MOVIES=(tp dm pc)
MOVIE="${MOVIES[$SLURM_ARRAY_TASK_ID]}"
SCORE_METRIC="${1:-pearson_r}"   # pass "r2" as $1 to sbatch to run the r2 version

echo "===== Task $SLURM_ARRAY_TASK_ID on $(hostname): movie=$MOVIE  score_metric=$SCORE_METRIC ====="

python3 run_hvm_diagnostic_singledelay.py --movies "$MOVIE" --score-metric "$SCORE_METRIC"

echo "===== Task $SLURM_ARRAY_TASK_ID ($MOVIE) done ====="
