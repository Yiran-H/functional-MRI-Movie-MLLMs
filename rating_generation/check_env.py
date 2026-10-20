"""
Quick check for all modules used by the Gemma 3 AND Qwen3-VL generation
scripts (video_analysis_tom_v2.py / video_analysis_valence_tr.py /
gemma_*_context.py / qwen3vl_*.py). These two models share the same conda
env ("movie") since they both need a modern transformers/torch stack.

Do NOT run this to check LLaDA-V readiness -- LLaDA-V needs its OWN conda
env (see check_env_llada.py) because its pinned deps (tokenizers~=0.15.2,
accelerate==0.29.1, numpy<2, ...) conflict with what Gemma/Qwen3-VL need.
"""

modules = [
    "os", "re", "warnings", "time", "csv", "argparse", "pandas",
    "torch", "torchvision", "PIL", "cv2", "transformers", "qwen_vl_utils",
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
try:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    print("Gemma3ForConditionalGeneration: OK")
except ImportError as e:
    print(f"Gemma3ForConditionalGeneration: MISSING ({e})")

try:
    from transformers import Qwen3VLForConditionalGeneration
    print("Qwen3VLForConditionalGeneration: OK  (needs transformers>=4.57.0)")
except ImportError as e:
    print(f"Qwen3VLForConditionalGeneration: MISSING ({e})  -> pip install -U 'transformers>=4.57.0'")

try:
    from qwen_vl_utils import process_vision_info
    print("qwen_vl_utils.process_vision_info: OK")
except ImportError as e:
    print(f"qwen_vl_utils.process_vision_info: MISSING ({e})  -> pip install 'qwen-vl-utils>=0.0.14'")

print()
try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"torch/cuda check failed: {e}")
