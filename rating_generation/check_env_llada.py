"""
Quick check for LLaDA-V readiness (video_analysis_tom_llada.py /
video_analysis_valence_llada.py). Run this INSIDE the dedicated LLaDA-V
conda env -- NOT the "movie" env used for Gemma/Qwen3-VL, since LLaDA-V's
pinned deps (tokenizers~=0.15.2, accelerate==0.29.1, numpy<2, ...) conflict
with the modern transformers stack Gemma/Qwen3-VL need. See setup notes
below the checks.
"""

modules = [
    "os", "re", "warnings", "copy", "time", "csv", "argparse", "pandas",
    "torch", "torchvision", "PIL", "cv2", "transformers", "numpy",
]

print(f"{'module':<15} {'status':<10} version")
print("-" * 50)

for name in modules:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "")
        print(f"{name:<15} {'OK':<10} {version}")
    except ImportError as e:
        print(f"{name:<15} {'MISSING':<10} ({e})")

print()
llava_checks = [
    ("llava.model.builder", "load_pretrained_model"),
    ("llava.mm_utils", "process_images"),
    ("llava.mm_utils", "tokenizer_image_token"),
    ("llava.constants", "IMAGE_TOKEN_INDEX"),
    ("llava.conversation", "conv_templates"),
    ("llava.hooks.fast_dllm_hook", "register_fast_dllm_hook"),
]
for mod_name, attr in llava_checks:
    try:
        mod = __import__(mod_name, fromlist=[attr])
        getattr(mod, attr)
        print(f"{mod_name}.{attr}: OK")
    except (ImportError, AttributeError) as e:
        print(f"{mod_name}.{attr}: MISSING ({e})")

print()
try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"torch/cuda check failed: {e}")

print("""
--- Setup notes (if llava.* is MISSING) ---
LLaDA-V needs its OWN conda env, separate from "movie":

  conda create -p /project/mx6/yh87/conda_env/llada python=3.10 -y
  conda activate /project/mx6/yh87/conda_env/llada

  git clone https://github.com/ML-GSAI/LLaDA-V
  cd LLaDA-V
  bash init_env.sh
  # ^ this runs:
  #   pip install -e eval/lmms-eval
  #   pip install -e "eval/lmms-eval[metrics]"
  #   pip install -e "train[train]"          <- this installs the `llava` package
  #   pip install --upgrade --force-reinstall --no-cache-dir "numpy<2" scikit-learn pandas

  # torch/torchvision are NOT pinned by LLaDA-V's pyproject.toml (commented
  # out) -- install them separately to match the cluster's CUDA, e.g. the
  # same cu124 wheels already used for Gemma:
  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

  export HF_HOME=/project/mx6/yh87/hf_cache   # reuse the same cache as Gemma/Qwen
""")
