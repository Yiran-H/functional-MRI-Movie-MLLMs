#!/bin/bash -l
#SBATCH --job-name=gemma_context_ablation
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
#SBATCH --array=0-5 # 0=TP-tom,1=DM-tom,2=PC-tom,3=TP-val,4=DM-val,5=PC-val -- each task = its own node/GPU, 4 variants only

set -e

module load Miniforge3
conda activate /project/mx6/yh87/conda_env/movie

# Keep HF cache/token and temp files on project space, not home quota
export HF_HOME=/project/mx6/yh87/hf_cache
mkdir -p /project/mx6/yh87/tmp
export TMPDIR=/project/mx6/yh87/tmp

# NOTE: this dir must also contain context_variants.py (copy it alongside the
# gemma_*.py scripts when syncing rating_generation/ to the cluster).
cd /project/mx6/yh87/Yiran-Movie-Paper/gemma

MODEL_ID="google/gemma-3-12b-it"

MOVIES=(TP   DM   PC    TP      DM      PC)
FILES=(movie/tp.mp4 movie/dm.mp4 movie/pc.mov movie/tp.mp4 movie/dm.mp4 movie/pc.mov)
TRS=(0.8  0.8  1.0   0.8     0.8     1.0)
MEASURES=(tom tom tom valence valence valence)

MOVIE=${MOVIES[$SLURM_ARRAY_TASK_ID]}
VIDEO=${FILES[$SLURM_ARRAY_TASK_ID]}
TR=${TRS[$SLURM_ARRAY_TASK_ID]}
MEASURE=${MEASURES[$SLURM_ARRAY_TASK_ID]}

OUTDIR="embedding/${MOVIE}_gemma"
DESC_CSV="$OUTDIR/gemma_${MOVIE}_description.csv"
mkdir -p "$OUTDIR"

echo "===== Task $SLURM_ARRAY_TASK_ID: $MOVIE $MEASURE on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ====="

if [ ! -f "$DESC_CSV" ]; then
  echo "ERROR: missing description CSV: $DESC_CSV (run gemma_description.py / run_description.sh first)"
  exit 1
fi

if [ "$MEASURE" = "tom" ]; then
  SCRIPT="gemma_tom_context.py"
else
  SCRIPT="gemma_valence_context.py"
fi

# Single window-length axis (4s/8s/12s/16s of lookback, exact match across
# TP/DM/PC) -- see context_variants.py for the per-movie (context_frames,
# jump) mapping and why. base (no context) already exists from the
# no-context run.
VARIANTS=(ctx_4s ctx_8s ctx_12s ctx_16s)

for LABEL in "${VARIANTS[@]}"; do
  read -r CFRAMES JUMP <<< "$(python context_variants.py "$MOVIE" "$LABEL")"

  echo "===== [$MOVIE $MEASURE] $LABEL (context_frames=$CFRAMES, jump=$JUMP) ====="
  python "$SCRIPT" \
    --video "$VIDEO" \
    --srt "$OUTDIR/gemma_${MOVIE}_${MEASURE}_${LABEL}.srt" \
    --csv "$OUTDIR/gemma_${MOVIE}_${MEASURE}_${LABEL}.csv" \
    --model-id "$MODEL_ID" --device cuda:0 --tr "$TR" \
    --context-frames "$CFRAMES" --jump "$JUMP" \
    --description-csv "$DESC_CSV"
done

echo "===== [$MOVIE $MEASURE] all ablation variants done ====="
