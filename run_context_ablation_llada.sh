#!/bin/bash -l
#SBATCH --job-name=llada_context_ablation
#SBATCH --output=/project/mx6/yh87/Yiran-Movie-Paper/reports/%x.%A_%a.out # %A=job id, %a=array index
#SBATCH --error=/project/mx6/yh87/Yiran-Movie-Paper/reports/%x.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000M
#SBATCH --qos=standard
#SBATCH --account=mx6 # Replace PI_ucid which the NJIT UCID of PI
#SBATCH --time=0-24:00:00  # D-HH:MM:SS
#SBATCH --exclude=n0025,n0024,n0088,n0004
#SBATCH --array=0-5 # 0=TP-tom,1=DM-tom,2=PC-tom,3=TP-val,4=DM-val,5=PC-val -- each task = its own node/GPU, base + 4 variants

set -e

module load Miniforge3
# TODO: fill in the LLaDA-V conda env path (this is a DIFFERENT env from the
# "movie" env used for Gemma -- LLaDA needs llava + Fast-dLLM installed).
conda activate /project/mx6/yh87/conda_env/llada

export HF_HOME=/project/mx6/yh87/hf_cache
mkdir -p /project/mx6/yh87/tmp
export TMPDIR=/project/mx6/yh87/tmp

# Same fix as the Qwen array job: 6 tasks on 6 different compute nodes all
# calling load_pretrained_model("GSAI-ML/LLaDA-V") at once corrupt the shared
# HF cache on the network filesystem even with offline mode set (the
# corruption is from concurrent local file access/locking, not network
# calls). Pre-download + verify the model ONCE via a serial srun session
# BEFORE submitting this array (same recipe as for Qwen), THEN resolve the
# local snapshot dir and pass that instead of the repo id.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LLADA_MODEL_DIR="$HF_HOME/hub/models--GSAI-ML--LLaDA-V"
MODEL_PATH=$(ls -d "$LLADA_MODEL_DIR"/snapshots/*/ 2>/dev/null | head -n1)
MODEL_PATH="${MODEL_PATH%/}"
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: local GSAI-ML/LLaDA-V snapshot not found/complete under $LLADA_MODEL_DIR/snapshots/ -- pre-download and verify it first before submitting this array:"
  echo '  srun --partition=gpu --gres=gpu:1 --time=00:30:00 --qos=standard --account=mx6 --pty bash'
  echo '  # then inside: conda activate this env, export HF_HOME=..., and load the model once with'
  echo '  # load_pretrained_model("GSAI-ML/LLaDA-V", ...) to populate the cache, then verify config.json exists.'
  exit 1
fi
echo "Using local model snapshot: $MODEL_PATH"

# TODO: fill in the LLaDA-V repo dir on the cluster. Must also contain
# context_variants.py (copy it alongside video_analysis_tom_llada.py /
# video_analysis_valence_llada.py when syncing rating_generation/).
cd /project/mx6/yh87/Yiran-Movie-Paper/llada

MOVIES=(TP   DM   PC    TP      DM      PC)
FILES=(movie/tp.mp4 movie/dm.mp4 movie/pc.mov movie/tp.mp4 movie/dm.mp4 movie/pc.mov)
TRS=(0.8  0.8  1.0   0.8     0.8     1.0)
MEASURES=(tom tom tom valence valence valence)

MOVIE=${MOVIES[$SLURM_ARRAY_TASK_ID]}
VIDEO=${FILES[$SLURM_ARRAY_TASK_ID]}
TR=${TRS[$SLURM_ARRAY_TASK_ID]}
MEASURE=${MEASURES[$SLURM_ARRAY_TASK_ID]}

# Matches existing naming convention in embedding/{MOVIE}_final/ (no model
# prefix, unlike qwen_/gemma_).
OUTDIR="embedding/${MOVIE}_final"
DESC_CSV="$OUTDIR/${MOVIE}_description.csv"
mkdir -p "$OUTDIR"

echo "===== Task $SLURM_ARRAY_TASK_ID: $MOVIE $MEASURE on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ====="

if [ ! -f "$DESC_CSV" ]; then
  echo "ERROR: missing description CSV: $DESC_CSV (LLaDA-V descriptions should already exist from the original base run)"
  exit 1
fi

if [ "$MEASURE" = "tom" ]; then
  SCRIPT="video_analysis_tom_llada.py"
else
  SCRIPT="video_analysis_valence_llada.py"
fi

# Re-run base too (context_frames=0), OVERWRITING the existing
# {MOVIE}_{MEASURE}.csv -- for tom this also fixes the old frame-drift bug
# (extract_frames_per_second -> extract_frames_by_tr), and for both
# measures it keeps base + all variants on one consistent pipeline/model
# snapshot. Then the context variants: a single window-length axis (4s/8s/
# 12s/16s of lookback, exact match across TP/DM/PC) -- see context_variants.py.
echo "===== [$MOVIE $MEASURE] base (context_frames=0) ====="
python "$SCRIPT" \
  --video "$VIDEO" \
  --srt "$OUTDIR/${MOVIE}_${MEASURE}.srt" \
  --csv "$OUTDIR/${MOVIE}_${MEASURE}.csv" \
  --model-id "$MODEL_PATH" \
  --device cuda:0 --tr "$TR"

VARIANTS=(ctx_4s ctx_8s ctx_12s ctx_16s)

for LABEL in "${VARIANTS[@]}"; do
  read -r CFRAMES JUMP <<< "$(python context_variants.py "$MOVIE" "$LABEL")"

  echo "===== [$MOVIE $MEASURE] $LABEL (context_frames=$CFRAMES, jump=$JUMP) ====="
  python "$SCRIPT" \
    --video "$VIDEO" \
    --srt "$OUTDIR/${MOVIE}_${MEASURE}_${LABEL}.srt" \
    --csv "$OUTDIR/${MOVIE}_${MEASURE}_${LABEL}.csv" \
    --model-id "$MODEL_PATH" \
    --device cuda:0 --tr "$TR" \
    --context-frames "$CFRAMES" --jump "$JUMP" \
    --description-csv "$DESC_CSV"
done

echo "===== [$MOVIE $MEASURE] base + all ablation variants done ====="
