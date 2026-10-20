# -*- coding: utf-8 -*-
"""
qwen3vl_description.py
使用Qwen3-VL进行视频帧描述分析，功能与LLaDA-V的video_analysis_two_stage_tr.py相同
支持TR采样，双阶段提示，输出格式完全兼容

Features:
- TR-based frame sampling (0.8s, 1.0s, etc.)
- Two-stage prompting: Description+Keywords, then People count+Face detection
- Compatible output format with LLaDA-V results
- GPU acceleration with Qwen3-VL
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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

def format_srt_time(seconds):
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# ============ STAGE 1 PROMPT: Description + Keywords ============
def build_prompt_stage1():
    """Stage 1: Focus on semantic understanding"""
    return (
        "[Answer ONLY in English]\n"
        "Describe this image in one sentence and provide 3-5 keywords.\n\n"
        "Format:\n"
        "Description: <one clear sentence about what you see>\n"
        "Keywords: <word1, word2, word3, word4, word5>\n\n"
        "Example:\n"
        "Description: A boy sitting on a couch looking into a cardboard box.\n"
        "Keywords: boy, couch, box, living room, curious"
    )

# ============ STAGE 2 PROMPT: People Count + Face Detection ============
def build_prompt_stage2():
    """Stage 2: Focus on precise visual counting"""
    return (
        "[Answer ONLY in English. Be PRECISE and LITERAL.]\n\n"
        "Look at this image carefully and answer TWO questions:\n\n"
        "Question 1: How many PEOPLE can you see?\n"
        "- Count ONLY if you can see their body (full or partial)\n"
        "- Do NOT count if you only see a hand, shadow, or reflection\n"
        "- Output ONLY a number: 0, 1, 2, 3, 4...\n\n"
        "Question 2: Can you see any HUMAN FACE?\n"
        "- A face means you can clearly see eyes, nose, or mouth\n"
        "- If the face is turned away, answer 'no'\n"
        "- If the face is blurred or too dark, answer 'no'\n"
        "- If you can only see the back of head, answer 'no'\n"
        "- Output ONLY: 'yes' or 'no'\n\n"
        "Format:\n"
        "Number_of_people: <number>\n"
        "Face_present: <yes or no>\n\n"
        "CRITICAL: Answer based ONLY on what you can CLEARLY SEE. Do not guess or infer from context."
    )

# ============ Postprocess Functions ============
def postprocess_stage1(text):
    """Extract description and keywords from Stage 1 output"""
    t = text.strip().replace("\r\n", "\n")
    desc = ""
    keywords_raw = ""
    
    lines = text.split('\n')
    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()
        
        if line_lower.startswith('description:'):
            desc = line_clean.split(':', 1)[1].strip()
            if '.' in desc:
                desc = desc.split('.')[0] + '.'
        
        elif line_lower.startswith('keywords:'):
            keywords_raw = line_clean.split(':', 1)[1].strip()
            # Stop at explanation patterns
            stop_patterns = r'[.]|i used|the definition|the image'
            keywords_raw = re.split(stop_patterns, keywords_raw, flags=re.IGNORECASE)[0]
    
    # Parse keywords
    keywords = []
    if keywords_raw:
        raw_kws = re.split(r'[,;/|]', keywords_raw)
        for kw in raw_kws:
            clean = re.sub(r'[^a-z\s]', '', kw.lower()).strip()
            clean = ' '.join(clean.split())
            
            if len(clean) > 2 and clean not in {'the', 'and', 'for', 'with'}:
                if clean not in {'used', 'definition', 'image', 'description'}:
                    if clean not in keywords:
                        keywords.append(clean)
            if len(keywords) >= 5:
                break
    
    # Fallback handling
    if not desc and t:
        sentences = re.split(r'[.!?]', t)
        if sentences:
            desc = sentences[0].strip()
            if desc and not desc.endswith('.'):
                desc += '.'
    
    if not keywords:
        # Extract any meaningful words as fallback
        words = re.findall(r'\b[a-z]{3,}\b', t.lower())
        keywords = [w for w in words[:5] if w not in {'the', 'and', 'for', 'with', 'this', 'that'}]
    
    return desc or "Image description not available.", keywords

def postprocess_stage2(text):
    """Extract people count and face detection from Stage 2 output"""
    num_people = 0
    face_present = "no"
    
    lines = text.split('\n')
    for line in lines:
        line_clean = line.strip().lower()
        
        if 'number_of_people:' in line_clean:
            try:
                num_str = re.search(r'\d+', line_clean)
                if num_str:
                    num_people = int(num_str.group())
                    num_people = max(0, min(10, num_people))  # Clamp 0-10
            except:
                num_people = 0
        
        elif 'face_present:' in line_clean:
            if 'yes' in line_clean:
                face_present = "yes"
            else:
                face_present = "no"
    
    return num_people, face_present

# ============ TR-based Frame Extraction ============
def extract_frames_by_tr(video_path, tr):
    """Extract frames at TR intervals (e.g., every 0.8s or 1.0s)"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\n📹 Video Info:")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f}s")
    print(f"  - fMRI TR: {tr}s")
    print(f"  - Sampling rate: 1 frame every {tr}s")
    
    frames = []
    current_time = 0.0  # Start from 0 seconds
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Current timestamp of this video frame
        current_video_time = frame_count / fps
        
        # Check if we've reached or passed the next TR timepoint
        if current_video_time >= current_time:
            # Extract this frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Store with the exact TR timestamp (not video timestamp)
            frames.append((len(frames), current_time, pil_image))
            
            # Move to next TR timepoint
            current_time += tr
        
        frame_count += 1
    
    cap.release()
    
    expected_frames = int(duration / tr) + 1
    print(f"\nExtracted {len(frames)} frames (expected ~{expected_frames})")
    print(f"TR timestamps: 0.0s, {tr}s, {2*tr}s, ... {(len(frames)-1)*tr:.1f}s")
    
    return frames

# ============ Qwen3-VL Generation Functions ============
def generate_with_qwen3vl(image, prompt_text, processor, model):
    """Generate response using Qwen3-VL with given prompt"""
    
    # Build messages in Qwen3-VL format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]
    }]
    
    # Process input
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=150,  # Sufficient for our tasks
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0].strip()
    
    return output_text

def analyze_frame_two_stage(image, processor, model):
    """
    Two-stage analysis using Qwen3-VL:
    1. Generate description + keywords
    2. Generate people count + face detection
    """
    
    print("    [1/5] Stage 1: Description + Keywords...", end='', flush=True)
    t_start = time.time()
    
    stage1_prompt = build_prompt_stage1()
    stage1_output = generate_with_qwen3vl(image, stage1_prompt, processor, model)
    
    desc, keywords = postprocess_stage1(stage1_output)
    print(f" Done ({time.time()-t_start:.2f}s)")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # ============ STAGE 2: People Count + Face Detection ============
    print("    [2/5] Stage 2: People + Face Count...", end='', flush=True)
    t_start = time.time()
    
    stage2_prompt = build_prompt_stage2()
    stage2_output = generate_with_qwen3vl(image, stage2_prompt, processor, model)
    
    num_people, face_present = postprocess_stage2(stage2_output)
    print(f" Done ({time.time()-t_start:.2f}s)")
    
    # Combine raw outputs for debugging
    raw_combined = f"[Stage 1]\n{stage1_output}\n\n[Stage 2]\n{stage2_output}"
    
    return desc, keywords, num_people, face_present, raw_combined

# ============ Main Function ============
def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL Video Frame Analysis')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--csv', required=True, help='Output CSV file path')
    parser.add_argument('--srt', required=True, help='Output SRT file path')
    parser.add_argument('--tr', type=float, default=1.0, help='fMRI TR in seconds (default: 1.0)')
    parser.add_argument('--test-first-n', type=int, help='Test mode: process only first N frames')
    parser.add_argument('--cache-dir', default="C:\\Qwen3-VL\\models", help='Model cache directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("QWEN3-VL VIDEO FRAME ANALYSIS")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"CSV Output: {args.csv}")
    print(f"SRT Output: {args.srt}")
    print(f"TR: {args.tr}s")
    print(f"Model Cache: {args.cache_dir}")
    print("=" * 70)
    
    # ============ Step 1: Load Model ============
    print("Step 1/4: Loading Qwen3-VL model...")
    print("=" * 70)
    
    device = "cuda"
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    
    print(f"🔄 Loading {model_name}...")
    
    start_time = time.time()
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        cache_dir=args.cache_dir
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=args.cache_dir
    )
    
    load_time = time.time() - start_time
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🎮 Model VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    
    print(f"✅ Model loaded in {load_time:.1f}s\n")
    
    # ============ Step 2: Extract Frames ============
    print("=" * 70)
    print("Step 2/4: Extracting frames based on TR...")
    print("=" * 70)
    
    frames = extract_frames_by_tr(args.video, args.tr)
    
    if args.test_first_n:
        frames = frames[:args.test_first_n]
        print(f"\n⚠️  Test mode: Processing only first {len(frames)} frames")
    
    print(f"\n✅ Ready to process: {len(frames)} frames\n")
    
    # ============ Step 3: Analyze Frames ============
    print("=" * 70)
    print("Step 3/4: Analyzing frames...")
    print("=" * 70)
    
    results = []
    total_time = 0
    
    for idx, (frame_idx, timestamp, image) in enumerate(frames):
        print(f"\n▶ Frame {idx+1}/{len(frames)} (TR timepoint: {timestamp:.1f}s)")
        
        t0 = time.time()
        desc, keywords, num_people, face_present, raw_output = analyze_frame_two_stage(
            image, processor, model
        )
        dt = time.time() - t0
        total_time += dt
        
        print(f"    [3/5] Time: {dt:.1f}s")
        print(f"    [4/5] Description: {desc[:60]}..." if len(desc) > 60 else f"    [4/5] Description: {desc}")
        print(f"    [5/5] Keywords: {keywords}")
        print(f"          People: {num_people} | Face: {face_present}")
        
        if idx % 5 == 0 and torch.cuda.is_available():
            print(f"          💾 VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        results.append({
            'index': frame_idx,
            'timestamp': timestamp,
            'description': desc,
            'keywords': keywords,
            'num_people': num_people,
            'face_present': face_present,
            'raw': raw_output
        })
    
    # Performance stats
    avg = total_time / len(frames) if frames else 0
    print(f"\n✅ Analysis complete!")
    print(f"  Avg: {avg:.1f}s/frame")
    print(f"  Total: {total_time/60:.1f} min")
    
    if frames:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        expected_full = int(duration / args.tr) + 1
        print(f"  Est. for full video ({duration:.1f}s): {expected_full} frames × {avg:.1f}s = {avg*expected_full/60:.1f} min\n")
    
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
            f.write(f"Description: {r['description']}\n")
            f.write(f"Keywords: {', '.join(r['keywords'])}\n")
            f.write(f"People: {r['num_people']} | Face: {r['face_present']}\n\n")
    
    print(f"✅ SRT: {args.srt}")
    
    # Save CSV file (compatible with LLaDA-V format)
    with open(args.csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_index', 'timestamp_sec', 'description', 'keywords', 
                    'num_people', 'face_present_binary', 'raw_output'])
        for r in results:
            face_bin = 1 if r['face_present'].lower() == 'yes' else 0
            w.writerow([
                r['index'],
                f"{r['timestamp']:.3f}",
                r['description'],
                ', '.join(r['keywords']),
                r['num_people'],
                face_bin,
                r['raw']
            ])
    
    print(f"✅ CSV: {args.csv}")
    
    # Final stats
    ppl_frames = sum(1 for r in results if r['num_people'] > 0)
    face_frames = sum(1 for r in results if r['face_present'].lower() == 'yes')
    total_ppl = sum(r['num_people'] for r in results)
    
    print(f"\n📊 Summary:")
    print(f"  TR used: {args.tr}s")
    print(f"  Frames processed: {len(results)}")
    print(f"  Frames with people: {ppl_frames}/{len(results)} ({100*ppl_frames/len(results) if results else 0:.1f}%)")
    print(f"  Frames with faces: {face_frames}/{len(results)} ({100*face_frames/len(results) if results else 0:.1f}%)")
    print(f"  Total people: {total_ppl}")
    print(f"  Avg people/frame: {total_ppl/len(results) if results else 0:.2f}")
    
    print("\n" + "=" * 70)
    print("✅✅✅ Complete! ✅✅✅")
    print("=" * 70)

if __name__ == "__main__":
    main()
