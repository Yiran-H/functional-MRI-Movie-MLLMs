# -*- coding: utf-8 -*-
"""
gemma_description.py
Video frame description using Gemma 3.
Single-stage: description only (no keywords, no people count, no face detection).
Supports TR-based frame sampling (0.8s, 1.0s, etc.)
"""

import os
import re
import warnings
import time
import csv
import argparse
import torch
from PIL import Image
import cv2
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

def format_srt_time(seconds):
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def build_prompt_description():
    """Description-only prompt"""
    return (
        "[Answer ONLY in English]\n"
        "Describe this image in one sentence.\n\n"
        "Format:\n"
        "Description: <one clear sentence about what you see>\n\n"
        "Example:\n"
        "Description: A boy sitting on a couch looking into a cardboard box."
    )

def postprocess_description(text):
    """Extract description from model output"""
    t = text.strip().replace("\r\n", "\n")
    desc = ""

    lines = t.split('\n')
    for line in lines:
        line_clean = line.strip()
        if line_clean.lower().startswith('description:'):
            desc = line_clean.split(':', 1)[1].strip()
            if '.' in desc:
                desc = desc.split('.')[0] + '.'
            break

    if not desc and t:
        sentences = re.split(r'[.!?]', t)
        if sentences:
            desc = sentences[0].strip()
            if desc and not desc.endswith('.'):
                desc += '.'

    return desc or "Image description not available."

def extract_frames_by_tr(video_path, tr):
    """Extract frames at TR intervals (e.g., every 0.8s or 1.0s)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"\nVideo Info:")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f}s")
    print(f"  - fMRI TR: {tr}s")
    print(f"  - Sampling rate: 1 frame every {tr}s")

    frames = []
    current_time = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_video_time = frame_count / fps

        if current_video_time >= current_time:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append((len(frames), current_time, pil_image))
            current_time += tr

        frame_count += 1

    cap.release()

    expected_frames = int(duration / tr) + 1
    print(f"\nExtracted {len(frames)} frames (expected ~{expected_frames})")
    print(f"TR timestamps: 0.0s, {tr}s, {2*tr}s, ... {(len(frames)-1)*tr:.1f}s")

    return frames

def analyze_frame_description(image, model, processor, device):
    """Generate a one-sentence description for a single frame"""

    print("    [1/3] Building prompt...", end='', flush=True)
    t_start = time.time()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt_description()},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=model.dtype)
    input_len = inputs["input_ids"].shape[-1]
    print(f" Done ({time.time()-t_start:.2f}s)")

    print("    [2/3] Generating...", end='', flush=True)
    t_start = time.time()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    outputs = outputs[0][input_len:]

    print(f" Done ({time.time()-t_start:.2f}s)")

    print("    [3/3] Decoding...", end='', flush=True)
    t_start = time.time()
    raw = processor.decode(outputs, skip_special_tokens=True).strip()
    desc = postprocess_description(raw)
    print(f" Done ({time.time()-t_start:.2f}s)")

    return desc, raw

def main():
    parser = argparse.ArgumentParser(description='Gemma 3 Video Frame Description (description only)')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--csv', required=True, help='Output CSV file path')
    parser.add_argument('--srt', required=True, help='Output SRT file path')
    parser.add_argument('--tr', type=float, default=1.0, help='fMRI TR in seconds (default: 1.0)')
    parser.add_argument('--model-id', default="google/gemma-3-12b-it")
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--test-first-n', type=int, help='Test mode: process only first N frames')

    args = parser.parse_args()
    device = args.device

    print("=" * 70)
    print("GEMMA 3 VIDEO FRAME DESCRIPTION")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"CSV Output: {args.csv}")
    print(f"SRT Output: {args.srt}")
    print(f"TR: {args.tr}s")
    print("=" * 70)

    # ============ Step 1: Load Model ============
    print("Step 1/4: Loading Gemma 3 model...")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print("Model loaded\n")

    # ============ Step 2: Extract Frames ============
    print("=" * 70)
    print("Step 2/4: Extracting frames based on TR...")
    print("=" * 70)

    frames = extract_frames_by_tr(args.video, args.tr)

    if args.test_first_n:
        frames = frames[:args.test_first_n]
        print(f"\nTest mode: Processing only first {len(frames)} frames")

    print(f"\nReady to process: {len(frames)} frames\n")

    # ============ Step 3: Analyze Frames ============
    print("=" * 70)
    print("Step 3/4: Generating descriptions...")
    print("=" * 70)

    results = []
    total_time = 0

    for idx, (frame_idx, timestamp, image) in enumerate(frames):
        print(f"\n Frame {idx+1}/{len(frames)} (TR timepoint: {timestamp:.1f}s)")

        t0 = time.time()
        desc, raw_output = analyze_frame_description(image, model, processor, device)
        dt = time.time() - t0
        total_time += dt

        print(f"    Time: {dt:.1f}s")
        print(f"    Description: {desc[:60]}..." if len(desc) > 60 else f"    Description: {desc}")

        if idx % 10 == 0 and torch.cuda.is_available():
            print(f"    VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")

        results.append({
            'index': frame_idx,
            'timestamp': timestamp,
            'description': desc,
            'raw': raw_output
        })

    avg = total_time / len(frames) if frames else 0
    print(f"\nAnalysis complete!")
    print(f"  Avg: {avg:.1f}s/frame")
    print(f"  Total: {total_time/60:.1f} min")

    if frames:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        expected_full = int(duration / args.tr) + 1
        print(f"  Est. for full video ({duration:.1f}s): {expected_full} frames x {avg:.1f}s = {avg*expected_full/60:.1f} min\n")

    # ============ Step 4: Save Results ============
    print("=" * 70)
    print("Step 4/4: Saving results...")
    print("=" * 70)

    # Save SRT file
    with open(args.srt, 'w', encoding='utf-8') as f:
        for i, r in enumerate(results, 1):
            t1 = r['timestamp']
            t2 = results[i]['timestamp'] if i < len(results) else t1 + args.tr
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(t1)} --> {format_srt_time(t2)}\n")
            f.write(f"Description: {r['description']}\n\n")

    print(f"SRT: {args.srt}")

    # Save CSV file (description only, matching frame_index/timestamp_sec convention)
    with open(args.csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_index', 'timestamp_sec', 'description', 'raw_output'])
        for r in results:
            w.writerow([
                r['index'],
                f"{r['timestamp']:.3f}",
                r['description'],
                r['raw']
            ])

    print(f"CSV: {args.csv}")

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
