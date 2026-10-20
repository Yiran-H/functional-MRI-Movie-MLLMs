#!/bin/bash -l
#SBATCH --job-name=qwen_context_ablation
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
# Qwen3-VL shares the same modern transformers/torch stack as Gemma, so the
# "movie" env works -- just make sure it has transformers>=4.57.0 and
# qwen-vl-utils>=0.0.14 installed (see rating_generation/check_env.py).
conda activate /project/mx6/yh87/conda_env/movie

export HF_HOME=/project/mx6/yh87/hf_cache
mkdir -p /project/mx6/yh87/tmp
export TMPDIR=/project/mx6/yh87/tmp

# Defense in depth: force offline mode too, though the real fix is loading
# from a resolved local snapshot dir below (bypassing HF's cache-resolution
# code entirely -- offline mode alone was NOT enough: 6 array tasks on 6
# different compute nodes concurrently touching the same model cache on the
# network filesystem ($HF_HOME under /mmfs1) corrupted
# model.safetensors.index.json even with HF_HUB_OFFLINE=1 set, because the
# corruption comes from concurrent local file access/locking on that
# filesystem, not from network calls.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Resolve the already-downloaded local snapshot dir ONCE and pass that as
# --model-id instead of the repo id "Qwen/Qwen3-VL-8B-Instruct". Loading by
# repo id makes transformers re-run its HF-cache resolution logic in every
# one of the 6 parallel processes; loading by local path skips that
# entirely and just reads the files directly. Pre-download+verify the cache
# once BEFORE submitting this array (see chat history / check_env.py).
QWEN_MODEL_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct"
MODEL_PATH=$(ls -d "$QWEN_MODEL_DIR"/snapshots/*/ 2>/dev/null | head -n1)
MODEL_PATH="${MODEL_PATH%/}"
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
  echo "ERROR: local Qwen3-VL snapshot not found/complete under $QWEN_MODEL_DIR/snapshots/ -- pre-download and verify it first (see chat notes) before submitting this array."
  exit 1
fi
echo "Using local model snapshot: $MODEL_PATH"

# TODO: fill in the Qwen3-VL scripts dir on the cluster. Must also contain
# context_variants.py (copy it alongside qwen3vl_valence.py /
# qwen3vl_tom_context_jump.py when syncing rating_generation/).
cd /project/mx6/yh87/Yiran-Movie-Paper/qwen

MOVIES=(TP   DM   PC    TP      DM      PC)
FILES=(movie/tp.mp4 movie/dm.mp4 movie/pc.mov movie/tp.mp4 movie/dm.mp4 movie/pc.mov)
TRS=(0.8  0.8  1.0   0.8     0.8     1.0)
MEASURES=(tom tom tom valence valence valence)

MOVIE=${MOVIES[$SLURM_ARRAY_TASK_ID]}
VIDEO=${FILES[$SLURM_ARRAY_TASK_ID]}
TR=${TRS[$SLURM_ARRAY_TASK_ID]}
MEASURE=${MEASURES[$SLURM_ARRAY_TASK_ID]}

OUTDIR="embedding/${MOVIE}_qwen"
DESC_CSV="$OUTDIR/qwen_${MOVIE}_description.csv"
mkdir -p "$OUTDIR"

echo "===== Task $SLURM_ARRAY_TASK_ID: $MOVIE $MEASURE on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ====="

if [ ! -f "$DESC_CSV" ]; then
  echo "ERROR: missing description CSV: $DESC_CSV (Qwen3-VL descriptions should already exist from the original base run)"
  exit 1
fi

if [ "$MEASURE" = "tom" ]; then
  SCRIPT="qwen3vl_tom_context_jump.py"
else
  SCRIPT="qwen3vl_valence.py"
fi

# Re-run base too (context_frames=0), OVERWRITING the existing
# qwen_{MOVIE}_{MEASURE}.csv -- the old base was generated on a different
# machine/transformers version, so on-cluster greedy decoding may not
# reproduce it exactly; re-running keeps base + all variants on one
# consistent pipeline. Then the context variants: a single window-length
# axis (4s/8s/12s/16s of lookback, exact match across TP/DM/PC) -- see
# context_variants.py.
echo "===== [$MOVIE $MEASURE] base (context_frames=0) ====="
python "$SCRIPT" \
  --video "$VIDEO" \
  --srt "$OUTDIR/qwen_${MOVIE}_${MEASURE}.srt" \
  --csv "$OUTDIR/qwen_${MOVIE}_${MEASURE}.csv" \
  --tr "$TR" \
  --model-id "$MODEL_PATH"

VARIANTS=(ctx_4s ctx_8s ctx_12s ctx_16s)

for LABEL in "${VARIANTS[@]}"; do
  read -r CFRAMES JUMP <<< "$(python context_variants.py "$MOVIE" "$LABEL")"

  echo "===== [$MOVIE $MEASURE] $LABEL (context_frames=$CFRAMES, jump=$JUMP) ====="
  python "$SCRIPT" \
    --video "$VIDEO" \
    --srt "$OUTDIR/qwen_${MOVIE}_${MEASURE}_${LABEL}.srt" \
    --csv "$OUTDIR/qwen_${MOVIE}_${MEASURE}_${LABEL}.csv" \
    --tr "$TR" \
    --context-frames "$CFRAMES" --jump "$JUMP" \
    --description-csv "$DESC_CSV" \
    --model-id "$MODEL_PATH"
done

echo "===== [$MOVIE $MEASURE] base + all ablation variants done ====="
