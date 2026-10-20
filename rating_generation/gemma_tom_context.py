# -*- coding: utf-8 -*-
"""
gemma_tom_context.py
Theory of Mind (ToM) Analysis with Temporal Context Support using Gemma 3

EXACT COPY from qwen3vl_tom_context_jump.py with following adaptations:
- Uses Gemma 3 instead of Qwen3-VL
- Identical prompt structure and wording
- Same frame indexing logic (0,1,2,3... not video frame numbers)
- Compatible output format
"""

import os
import re
import warnings
import time
import csv
import argparse
import pandas as pd
import torch
from PIL import Image
import cv2
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

def format_srt_time(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def load_description_csv(csv_path):
    """
    Load description CSV file into pandas DataFrame

    Args:
        csv_path: Path to description CSV file

    Returns:
        pandas DataFrame with columns: frame_index, timestamp_sec, description, etc.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Description CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ['frame_index', 'timestamp_sec', 'description']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Description CSV missing required columns: {missing_cols}")

    print(f"Loaded description CSV: {len(df)} frames")
    return df

def build_context_text(current_sample_idx, context_frames, description_df, tr, jump=0):
    """
    Build temporal context text from previous sampled frames with optional jump sampling

    Args:
        current_sample_idx: Index of current sample point (row number in description_df)
        context_frames: Number of previous sampled frames to include (e.g., 5)
        description_df: DataFrame with frame descriptions (already sampled at TR intervals)
        tr: Repetition Time in seconds (e.g., 0.8 or 1.0)
        jump: Jump/skip parameter for sampling (default: 0)
              - jump=0: consecutive sampling (step=1)
              - jump=1: skip 1 between samples (step=2)
              - jump=2: skip 2 between samples (step=3)

    Returns:
        Formatted context string with clear instructions, or empty string if no context
    """
    if context_frames <= 0 or current_sample_idx == 0:
        return ""

    step = jump + 1

    prev_indices = []
    for i in range(1, context_frames + 1):
        prev_idx = current_sample_idx - (i * step)
        if prev_idx >= 0:
            prev_indices.append(prev_idx)
        else:
            break

    if len(prev_indices) == 0:
        return ""

    context_lines = []
    context_lines.append("═" * 77)
    context_lines.append("                         VIDEO STORY CONTEXT                            ")
    context_lines.append("═" * 77)
    context_lines.append("")
    context_lines.append("You are watching a continuous video story. Below are descriptions of key")
    context_lines.append("moments that happened BEFORE the current frame. These previous scenes provide")
    context_lines.append("narrative context to help you understand the character's emotional journey")
    context_lines.append("and current mental state.")
    context_lines.append("")
    context_lines.append("Use this story progression to inform your Theory of Mind rating.")
    context_lines.append("")
    context_lines.append("Previous Moments in the Story:")

    for prev_idx in reversed(prev_indices):
        if prev_idx < len(description_df):
            row = description_df.iloc[prev_idx]
            desc = row['description']
            time_ago = (current_sample_idx - prev_idx) * tr
            context_lines.append(f"• {time_ago:.1f} seconds ago: {desc}")
        else:
            time_ago = (current_sample_idx - prev_idx) * tr
            context_lines.append(f"• {time_ago:.1f} seconds ago: [Description not available]")

    context_lines.append("")
    context_lines.append("─" * 77)
    context_lines.append("")
    context_lines.append("NOW → Current Frame (the image you see above):")
    context_lines.append("Consider how the story has developed. Based on this progression,")
    context_lines.append("rate the Theory of Mind for THIS current moment.")
    context_lines.append("")
    context_lines.append("═" * 77)
    context_lines.append("")

    return "\n".join(context_lines)

def build_prompt_tom(context_text=""):
    """
    Theory of Mind Prompt - Focus on EMOTIONAL ENGAGEMENT and EMPATHY
    EXACT COPY from qwen3vl_tom_context_jump.py including special characters

    Args:
        context_text: Optional temporal context from previous frames

    Returns:
        Complete prompt string
    """
    context_section = ""
    if context_text:
        context_section = context_text + "\n"

    return (
        "[Answer ONLY in English]\n\n"
        + context_section +
        "═══════════════════════════════════════════════════════════════════════\n"
        "                    THEORY OF MIND (ToM) RATING TASK                   \n"
        "═══════════════════════════════════════════════════════════════════════\n\n"

        "DEFINITION:\n"
        "Theory of Mind is the ability to understand and infer what others might be \n"
        "thinking or feeling, and to predict their actions based on those inferences.\n\n"

        "YOUR TASK AS A VIEWER:\n"
        "Rate the extent to which YOU can perceive, understand, and emotionally connect \n"
        "with the character's emotions, thoughts, or intentions in this moment.\n\n"

        "🎯 KEY QUESTION:\n"
        "\"How much can I feel and understand what's going on in this character's mind?\"\n\n"

        "This is about YOUR SUBJECTIVE EXPERIENCE as a viewer:\n"
        "• Can you sense what they're feeling?\n"
        "• Can you infer what they're thinking?\n"
        "• Can you predict what they might do next?\n"
        "• Do you feel emotionally connected to their mental state?\n\n"

        "───────────────────────────────────────────────────────────────────────\n"
        "                    RATING SCALE: -10 to +10                            \n"
        "───────────────────────────────────────────────────────────────────────\n\n"

        "Start from 0 (neutral baseline) and move the rating based on your experience:\n\n"

        "POSITIVE DIRECTION (+1 to +10): INCREASING EMOTIONAL CONNECTION\n"
        "\"I CAN feel/sense/understand what the character is experiencing\"\n\n"

        "+1 to +3: WEAK CONNECTION\n"
        "• I can sense some emotional presence, but it's vague\n"
        "• Minimal understanding of their mental state\n"
        "• Limited emotional engagement\n"
        "Example: Character is present but their emotions are subtle/unclear\n\n"

        "+4 to +6: MODERATE CONNECTION\n"
        "• I can clearly identify their basic emotional state (happy, sad, angry, etc.)\n"
        "• I understand what they might be thinking about\n"
        "• Moderate level of emotional resonance - I \"get\" what they're going through\n"
        "Example: Character shows clear emotions; I can relate to their situation\n\n"

        "+7 to +8: STRONG CONNECTION\n"
        "• I deeply understand their complex emotional state\n"
        "• I can infer their thoughts, motivations, and likely next actions\n"
        "• Strong empathic response - I feel WITH them\n"
        "• Multiple emotional layers are perceivable\n"
        "Example: Character's internal struggle is palpable; I'm emotionally invested\n\n"

        "+9 to +10: MAXIMUM CONNECTION\n"
        "• Complete psychological transparency - I fully \"inhabit\" their mental state\n"
        "• Profound empathic resonance - their emotions become MY emotions\n"
        "• Crystal clear understanding of their thoughts, fears, hopes, intentions\n"
        "• Peak emotional engagement - transformative moment of connection\n"
        "Example: Breakthrough/epiphany moment; overwhelming sense of shared humanity\n\n"

        "┌───────────────────────────────────────────────────────────────────────┐\n\n"

        "0: NEUTRAL BASELINE\n"
        "• No emotional engagement or connection\n"
        "• Cannot perceive any mental states\n"
        "• Emotionally flat scene / transition moment\n"
        "Example: Black screen, title card, empty scene, or emotionally neutral moment\n\n"

        "└───────────────────────────────────────────────────────────────────────┘\n\n"

        "NEGATIVE DIRECTION (-1 to -10): DECREASING EMOTIONAL CONNECTION\n"
        "\"I CANNOT feel/sense/understand - there are barriers to connection\"\n\n"

        "-1 to -3: WEAK BARRIER\n"
        "• Character present but emotional access is limited\n"
        "• Difficulty reading their mental state (ambiguous expression, poor angle)\n"
        "• Weak negative feeling - something blocks full engagement\n"
        "Example: Side view, neutral expression, unclear context\n\n"

        "-4 to -6: MODERATE BARRIER\n"
        "• Significant obstacles to understanding their mental state\n"
        "• Cannot see face/eyes or body language clearly\n"
        "• Frustration in trying to connect - I want to understand but can't\n"
        "Example: Back turned, face covered, distance, darkness obscures emotions\n\n"

        "-7 to -8: STRONG BARRIER\n"
        "• Major blockage to emotional connection\n"
        "• Almost no access to mental states despite character presence\n"
        "• Strong sense of disconnect - complete emotional opacity\n"
        "Example: Heavily obscured, extreme distance, no visible cues\n\n"

        "-9 to -10: MAXIMUM BARRIER\n"
        "• Total inability to access mental states\n"
        "• Complete emotional disconnection - like trying to read a blank wall\n"
        "• Possible active misdirection or deception\n"
        "Example: Fully masked/disguised figure; intentionally hidden emotions\n\n"

        "───────────────────────────────────────────────────────────────────────\n"
        "                         IMPORTANT PRINCIPLES                           \n"
        "───────────────────────────────────────────────────────────────────────\n\n"

        "1. SUBJECTIVE EXPERIENCE:\n"
        "   This is about YOUR felt sense as a viewer, not objective description\n"
        "   • Focus on: \"What am I experiencing?\" not \"What is in the frame?\"\n\n"

        "2. EMOTIONAL RESONANCE:\n"
        "   • Positive = \"I feel connected to their inner world\"\n"
        "   • Negative = \"I feel blocked/disconnected from their inner world\"\n"
        "   • Zero = \"No emotional engagement happening\"\n\n"

        "3. DYNAMIC RATING:\n"
        "   • Your rating can change moment by moment\n"
        "   • Early in video: ratings may be lower (getting to know character)\n"
        "   • Climax moments: ratings may be higher (deep emotional peaks)\n\n"

        "4. CONTEXT MATTERS:\n"
        "   • Empty scenes / black screens → typically 0 (no one to connect with)\n"
        "   • But if the scene creates emotional atmosphere → can be positive\n"
        "   • Story progression affects your ability to understand character\n\n"

        "5. WHAT DRIVES RATINGS:\n"
        "   POSITIVE (+): Clear emotions, visible thoughts, relatable situations, \n"
        "                 expressive faces/bodies, emotional buildup, story climax\n"
        "   NEGATIVE (-): Physical obscurity, emotional ambiguity, lack of context,\n"
        "                 blocked facial cues, intentional hiding\n"
        "   ZERO (0): No character, or complete emotional neutrality\n\n"

        "───────────────────────────────────────────────────────────────────────\n"
        "                              OUTPUT FORMAT                             \n"
        "───────────────────────────────────────────────────────────────────────\n\n"

        "You MUST respond in EXACTLY this format:\n\n"

        "ToM_Rating: <integer from -10 to +10>\n"
        "Reasoning: <brief explanation in 1-2 sentences>\n\n"

        "CRITICAL RULES:\n"
        "• ToM_Rating MUST be a single integer (-10 to +10)\n"
        "• Reasoning MUST be concise (maximum 2 sentences)\n"
        "• NO additional commentary, NO preamble, NO extra explanations\n\n"

        "EXAMPLES:\n\n"

        "Example 1 (No character):\n"
        "ToM_Rating: 0\n"
        "Reasoning: Black screen with title text - no character present to connect with.\n\n"

        "Example 2 (Moderate positive):\n"
        "ToM_Rating: +5\n"
        "Reasoning: Boy's facial expression clearly shows curiosity and surprise as he opens the box.\n\n"

        "Example 3 (Strong positive):\n"
        "ToM_Rating: +8\n"
        "Reasoning: Boy's eyes widen with shock, then soften into compassion - I can feel his emotional journey from surprise to empathy.\n\n"

        "Example 4 (Moderate negative):\n"
        "ToM_Rating: -5\n"
        "Reasoning: Character's back is turned, face not visible - cannot access emotional state.\n\n"

        "═══════════════════════════════════════════════════════════════════════\n"
        "                            BEGIN ANALYSIS                              \n"
        "═══════════════════════════════════════════════════════════════════════"
    )

def extract_frames_by_tr(video_path, tr=1.0):
    """
    Extract frames based on fMRI Repetition Time (TR)

    Args:
        video_path: Path to video file
        tr: Repetition Time in seconds (e.g., 0.8 or 1.0)

    Returns:
        List of (frame_index, timestamp_sec, PIL_Image) tuples
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video info:")
    print(f"  - Native FPS: {fps:.2f}")
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
    print(f"  - Extracted: {len(frames)} frames")
    return frames

def analyze_frame_tom(image, model, processor, device, current_sample_idx=0, description_df=None,
                     context_frames=0, tr=1.0, jump=0):
    """Analyze single frame for Theory of Mind with optional temporal context"""

    context_text = ""
    if context_frames > 0 and description_df is not None and current_sample_idx > 0:
        context_text = build_context_text(current_sample_idx, context_frames,
                                        description_df, tr, jump)

    tom_prompt = build_prompt_tom(context_text)

    print("    [1/3] Generating ToM analysis...", end='', flush=True)
    t_start = time.time()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": tom_prompt},
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

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    outputs = outputs[0][input_len:]

    raw_output = processor.decode(outputs, skip_special_tokens=True).strip()

    tom_rating, reasoning = postprocess_tom_output(raw_output)

    dt = time.time() - t_start
    print(f" Done ({dt:.2f}s)")

    return tom_rating, reasoning, raw_output

def postprocess_tom_output(text):
    """Extract ToM rating and reasoning from model output"""
    tom_rating = 0
    reasoning = "No valid output detected."

    rating_match = re.search(r'ToM_Rating:\s*([+-]?\d+)', text, re.IGNORECASE)
    if rating_match:
        try:
            tom_rating = int(rating_match.group(1))
            tom_rating = max(-10, min(10, tom_rating))
        except ValueError:
            tom_rating = 0

    reasoning_match = re.search(r'Reasoning:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        sentences = re.split(r'[.!?]+', reasoning)
        if len(sentences) >= 2 and sentences[1].strip():
            reasoning = sentences[0].strip() + '. ' + sentences[1].strip() + '.'
        elif len(sentences) >= 1 and sentences[0].strip():
            reasoning = sentences[0].strip() + '.'
        else:
            reasoning = "Unable to extract reasoning."

    return tom_rating, reasoning

def main():
    parser = argparse.ArgumentParser(description='Gemma 3 Theory of Mind Analysis with Context')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--csv', required=True, help='Output CSV file path')
    parser.add_argument('--srt', required=True, help='Output SRT file path')
    parser.add_argument('--tr', type=float, default=1.0, help='fMRI TR in seconds (default: 1.0)')
    parser.add_argument('--context-frames', type=int, default=0,
                       help='Number of previous frames for temporal context (default: 0)')
    parser.add_argument('--jump', type=int, default=0,
                       help='Jump parameter for context sampling: 0=consecutive, 1=skip 1, 2=skip 2, etc.')
    parser.add_argument('--description-csv',
                       help='Path to description CSV file for temporal context')
    parser.add_argument('--test-first-n', type=int, help='Test mode: process only first N frames')
    parser.add_argument('--model-id', default="google/gemma-3-12b-it")
    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()
    device = args.device

    if args.context_frames > 0 and not args.description_csv:
        parser.error("--context-frames requires --description-csv")

    print("=" * 80)
    print("GEMMA 3 THEORY OF MIND ANALYSIS WITH TEMPORAL CONTEXT")
    print("=" * 80)
    print(f"Video: {args.video}")
    print(f"CSV Output: {args.csv}")
    print(f"SRT Output: {args.srt}")
    print(f"TR: {args.tr}s")
    print(f"Context frames: {args.context_frames}")
    print(f"Jump parameter: {args.jump}")
    print(f"Description CSV: {args.description_csv}")
    print("=" * 80)

    # ============ Step 1: Load Model ============
    print("Step 1/5: Loading Gemma 3 model...")
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print("Model loaded\n")

    # ============ Step 2: Load Description CSV (if using context) ============
    description_df = None
    if args.context_frames > 0:
        print("=" * 80)
        print("Step 2/5: Loading description CSV for temporal context...")
        print("=" * 80)
        description_df = load_description_csv(args.description_csv)
        print()

    # ============ Step 3: Extract Frames ============
    print("=" * 80)
    print(f"Step {3 if args.context_frames == 0 else '3'}/5: Extracting video frames...")
    print("=" * 80)

    frames = extract_frames_by_tr(args.video, args.tr)

    if args.test_first_n:
        frames = frames[:args.test_first_n]
        print(f"\nTest mode: Processing only first {len(frames)} frames")

    print(f"\nReady to analyze: {len(frames)} frames\n")

    # ============ Step 4: Analyze Frames ============
    print("=" * 80)
    print(f"Step {4 if args.context_frames == 0 else '4'}/5: Analyzing Theory of Mind...")
    print("=" * 80)

    results = []
    total_time = 0

    for idx, (frame_idx, timestamp, image) in enumerate(frames):
        print(f"\n▶ Frame {idx+1}/{len(frames)} (TR timepoint: {timestamp:.1f}s)")

        t0 = time.time()
        tom_rating, reasoning, raw_output = analyze_frame_tom(
            image, model, processor, device,
            current_sample_idx=idx,
            description_df=description_df,
            context_frames=args.context_frames,
            tr=args.tr,
            jump=args.jump
        )
        dt = time.time() - t0
        total_time += dt

        rating_str = f"+{tom_rating}" if tom_rating > 0 else str(tom_rating)
        print(f"    [2/3] Time: {dt:.1f}s")
        print(f"    [3/3] ToM Rating: {rating_str}/10")
        print(f"          Reasoning: {reasoning[:80]}..." if len(reasoning) > 80 else f"          Reasoning: {reasoning}")

        if idx % 5 == 0 and torch.cuda.is_available():
            print(f"          VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")

        results.append({
            'index': idx,
            'timestamp': timestamp,
            'tom_rating': tom_rating,
            'tom_reasoning': reasoning,
            'raw': raw_output
        })

        if idx % 10 == 0:
            torch.cuda.empty_cache()

    avg = total_time / len(frames) if frames else 0
    print(f"\nAnalysis complete!")
    print(f"  Avg: {avg:.1f}s/frame")
    print(f"  Total: {total_time/60:.1f} min")

    if frames:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_video_frames / fps
        cap.release()
        expected_full = int(duration / args.tr) + 1
        print(f"  Est. for full video ({duration:.1f}s): {expected_full} frames x {avg:.1f}s = {avg*expected_full/60:.1f} min\n")

    # ============ Step 5: Save Results ============
    print("=" * 80)
    print(f"Step {5 if args.context_frames == 0 else '5'}/5: Saving results...")
    print("=" * 80)

    with open(args.srt, 'w', encoding='utf-8') as f:
        for i, r in enumerate(results, 1):
            t1 = r['timestamp']
            t2 = results[i]['timestamp'] if i < len(results) else t1 + args.tr
            rating_str = f"+{r['tom_rating']}" if r['tom_rating'] > 0 else str(r['tom_rating'])
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(t1)} --> {format_srt_time(t2)}\n")
            f.write(f"ToM: {rating_str}/10\n")
            f.write(f"{r['tom_reasoning']}\n\n")

    print(f"SRT: {args.srt}")

    with open(args.csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_index', 'timestamp_sec', 'tom_rating', 'tom_reasoning', 'raw_output'])
        for r in results:
            w.writerow([
                r['index'],
                f"{r['timestamp']:.3f}",
                r['tom_rating'],
                r['tom_reasoning'],
                r['raw']
            ])

    print(f"CSV: {args.csv}")

    ratings = [r['tom_rating'] for r in results]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    max_rating = max(ratings) if ratings else 0
    min_rating = min(ratings) if ratings else 0

    positive_frames = sum(1 for r in ratings if r > 0)
    negative_frames = sum(1 for r in ratings if r < 0)
    neutral_frames = sum(1 for r in ratings if r == 0)

    print(f"\nToM Statistics:")
    print(f"  Context frames used: {args.context_frames}")
    if args.context_frames > 0:
        print(f"  Jump parameter: {args.jump} (step size: {args.jump + 1})")
    print(f"  TR used: {args.tr}s")
    print(f"  Frames processed: {len(results)}")
    print(f"  Average Rating: {avg_rating:+.2f}/10")
    print(f"  Max Rating: {max_rating:+}/10")
    print(f"  Min Rating: {min_rating:+}/10")
    print(f"  Positive ToM (>0): {positive_frames}/{len(results)} ({100*positive_frames/len(results) if results else 0:.1f}%)")
    print(f"  Negative ToM (<0): {negative_frames}/{len(results)} ({100*negative_frames/len(results) if results else 0:.1f}%)")
    print(f"  Neutral ToM (=0): {neutral_frames}/{len(results)} ({100*neutral_frames/len(results) if results else 0:.1f}%)")

    print("\n" + "=" * 80)
    print("ToM Analysis Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
