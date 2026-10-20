#!/bin/bash -l
#SBATCH --job-name=gemma_description
#SBATCH --output=/project/mx6/yh87/Yiran-Movie-Paper/reports/%x.%A_%a.out # %A=job id, %a=array index
#SBATCH --error=/project/mx6/yh87/Yiran-Movie-Paper/reports/%x.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000M # Maximum allowable memory per CPU 8G (12B model needs headroom during load)
#SBATCH --qos=standard
#SBATCH --account=mx6 # Replace PI_ucid which the NJIT UCID of PI
#SBATCH --time=0-24:00:00  # D-HH:MM:SS
#SBATCH --exclude=n0025,n0024,n0088,n0004
#SBATCH --array=0-2 # 0=TP, 1=DM, 2=PC -- each array task gets its own node/GPU

set -e

module load Miniforge3
conda activate /project/mx6/yh87/conda_env/movie

# Keep HF cache/token and temp files on project space, not home quota
export HF_HOME=/project/mx6/yh87/hf_cache
mkdir -p /project/mx6/yh87/tmp
export TMPDIR=/project/mx6/yh87/tmp

cd /project/mx6/yh87/Yiran-Movie-Paper/gemma

MODEL_ID="google/gemma-3-12b-it"

MOVIES=(TP DM PC)
FILES=(movie/tp.mp4 movie/dm.mp4 movie/pc.mov)
TRS=(0.8 0.8 1.0)

MOVIE=${MOVIES[$SLURM_ARRAY_TASK_ID]}
VIDEO=${FILES[$SLURM_ARRAY_TASK_ID]}
TR=${TRS[$SLURM_ARRAY_TASK_ID]}

OUTDIR="embedding/${MOVIE}_gemma"
mkdir -p "$OUTDIR"

echo "===== Task $SLURM_ARRAY_TASK_ID: $MOVIE on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ====="

echo "===== [$MOVIE] Description (tr=$TR) ====="
python gemma_description.py \
  --video "$VIDEO" \
  --srt "$OUTDIR/gemma_${MOVIE}_description.srt" \
  --csv "$OUTDIR/gemma_${MOVIE}_description.csv" \
  --model-id "$MODEL_ID" --device cuda:0 --tr "$TR"

echo "===== [$MOVIE] done ====="
