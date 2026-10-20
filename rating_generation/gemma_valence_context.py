# -*- coding: utf-8 -*-
"""
gemma_valence_context.py
Emotional Valence Analysis with Temporal Context Support using Gemma 3

Adapted from gemma_tom_context.py (context/jump mechanism) and
video_analysis_valence_tr.py (prompt wording + postprocessing), so the
"add narrative context" ablation can be run for valence the same way it
already exists for ToM (qwen3vl_tom_context_jump.py).
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
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def load_description_csv(csv_path):
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
    Build temporal context text from previous sampled frames with optional jump
    sampling. Same mechanism as gemma_tom_context.py, reworded for valence.
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
    context_lines.append("narrative context to help you understand how the story has been developing")
    context_lines.append("emotionally leading up to this moment.")
    context_lines.append("")
    context_lines.append("Use this story progression to inform your emotional valence rating.")
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
    context_lines.append("rate your emotional valence for THIS current moment.")
    context_lines.append("")
    context_lines.append("═" * 77)
    context_lines.append("")

    return "\n".join(context_lines)

def build_prompt_valence(context_text=""):
    """
    Emotional Valence Prompt - EXACT COPY from video_analysis_valence_tr.py,
    with an optional temporal-context block prepended.
    """
    context_section = ""
    if context_text:
        context_section = context_text + "\n"

    return (
        "[Answer ONLY in English]\n\n"
        + context_section +
        "═══════════════════════════════════════════════════════════════════════\n"
        "                    EMOTIONAL VALENCE RATING TASK                      \n"
        "═══════════════════════════════════════════════════════════════════════\n\n"

        "DEFINITION:\n"
        "Emotional valence refers to how positive or negative YOUR feelings are \n"
        "at this specific moment while watching the video. This is about YOUR \n"
        "immediate emotional experience as a viewer.\n\n"

        "🎯 KEY QUESTION:\n"
        "\"How does this scene make ME feel emotionally right now?\"\n\n"

        "YOUR TASK AS A VIEWER:\n"
        "Rate your PERSONAL emotional response to what you see. Focus on:\n"
        "• How you PERSONALLY feel in this moment (not what you think)\n"
        "• Your IMMEDIATE emotional reaction (pleasant vs. unpleasant)\n"
        "• Your GUT feeling about the scene (positive, negative, or neutral)\n\n"

        "IMPORTANT DISTINCTIONS:\n"
        "❌ NOT character's emotions (what they feel)\n"
        "❌ NOT objective description (what's in the scene)\n"
        "✅ YOUR emotional response (how the scene makes YOU feel)\n\n"

        "───────────────────────────────────────────────────────────────────────\n"
        "                    RATING SCALE: -10 to +10                           \n"
        "───────────────────────────────────────────────────────────────────────\n\n"

        "This is a CONTINUOUS SCALE measuring your emotional state:\n\n"

        "NEGATIVE VALENCE (-10 to -1): UNPLEASANT FEELINGS\n"
        "\"This scene makes me feel BAD / UNPLEASANT / UNCOMFORTABLE\"\n\n"

        "-10 to -8: EXTREMELY NEGATIVE\n"
        "• Overwhelming distress, anguish, or revulsion\n"
        "• Highly disturbing, deeply upsetting content\n"
        "• Intense discomfort, profound sadness, severe anxiety\n"
        "• You feel strongly aversive - want to look away\n"
        "Example: Graphic violence, extreme suffering, traumatic content\n\n"

        "-7 to -5: STRONGLY NEGATIVE\n"
        "• Clear negative feelings - significantly unpleasant\n"
        "• Distressing, very sad, quite uncomfortable\n"
        "• Strong sense of unease, worry, or melancholy\n"
        "• Definitely NOT enjoying this moment\n"
        "Example: Character in obvious pain, conflict, rejection, loneliness\n\n"

        "-4 to -2: MODERATELY NEGATIVE\n"
        "• Noticeable unpleasant feelings but not overwhelming\n"
        "• Somewhat sad, mildly uncomfortable, slightly worried\n"
        "• Gentle negative mood - like a gray cloudy day\n"
        "• You feel a bit down or uneasy\n"
        "Example: Disappointment, mild tension, awkward social moment\n\n"

        "-1: SLIGHTLY NEGATIVE\n"
        "• Barely unpleasant - just a hint of negativity\n"
        "• Almost neutral but leaning slightly uncomfortable\n"
        "• Subtle sense something isn't quite right\n"
        "• Very mild displeasure\n"
        "Example: Minor inconvenience, slight concern, faint unease\n\n"

        "┌───────────────────────────────────────────────────────────────────────┐\n\n"

        "0: NEUTRAL BASELINE\n"
        "• No emotional response - emotionally flat\n"
        "• Neither pleasant nor unpleasant\n"
        "• Neutral observation - no feelings triggered\n"
        "• Like watching informational text or blank screen\n"
        "Example: Title cards, credits, empty scenes, mundane objects\n\n"

        "└───────────────────────────────────────────────────────────────────────┘\n\n"

        "POSITIVE VALENCE (+1 to +10): PLEASANT FEELINGS\n"
        "\"This scene makes me feel GOOD / PLEASANT / HAPPY\"\n\n"

        "+1: SLIGHTLY POSITIVE\n"
        "• Barely pleasant - just a hint of positivity\n"
        "• Almost neutral but leaning slightly comfortable\n"
        "• Subtle warmth or contentment\n"
        "• Very mild pleasure\n"
        "Example: Peaceful moment, gentle smile, soft music\n\n"

        "+2 to +4: MODERATELY POSITIVE\n"
        "• Noticeable pleasant feelings - feel somewhat good\n"
        "• Pleasant, nice, somewhat happy, mildly uplifted\n"
        "• Gentle positive mood - like a sunny spring day\n"
        "• You feel a bit better watching this\n"
        "Example: Friendly interaction, beauty in nature, kindness shown\n\n"

        "+5 to +7: STRONGLY POSITIVE\n"
        "• Clear positive feelings - significantly pleasant\n"
        "• Happy, uplifting, heartwarming, joyful\n"
        "• Strong sense of warmth, hope, or delight\n"
        "• Definitely enjoying this moment\n"
        "Example: Reunion, success, playfulness, tender connection\n\n"

        "+8 to +10: EXTREMELY POSITIVE\n"
        "• Overwhelming positive feelings - pure joy\n"
        "• Deeply moving, profoundly uplifting, ecstatic\n"
        "• Intense happiness, euphoria, transcendent beauty\n"
        "• You feel wonderfully alive and connected\n"
        "Example: Breakthrough moment, triumph, pure love, awe-inspiring beauty\n\n"

        "───────────────────────────────────────────────────────────────────────\n"
        "                         RATING PRINCIPLES                              \n"
        "───────────────────────────────────────────────────────────────────────\n\n"

        "1. CONTINUOUS RATING:\n"
        "   • Think of this as a SLIDER that can be anywhere from -10 to +10\n"
        "   • Use the FULL RANGE - don't cluster around 0\n"
        "   • Your feelings can be intense (-10, +10) or subtle (-1, +1)\n\n"

        "2. IMMEDIATE RESPONSE:\n"
        "   • Rate what you feel RIGHT NOW in this moment\n"
        "   • Don't overthink - go with your gut emotional reaction\n"
        "   • Your first emotional impulse is usually most accurate\n\n"

        "3. SUBJECTIVE EXPERIENCE:\n"
        "   • There are NO wrong answers - only YOUR feelings matter\n"
        "   • Two viewers can rate the same scene very differently\n"
        "   • Focus on: \"What am I FEELING?\" not \"What should I feel?\"\n\n"

        "4. VIEWER PERSPECTIVE:\n"
        "   • This is about how the scene affects YOU as a viewer\n"
        "   • A sad character might make you feel empathy (+2) or discomfort (-2)\n"
        "   • Consider the TOTAL IMPACT: visuals, context, atmosphere, story\n\n"

        "5. DYNAMIC CHANGES:\n"
        "   • Your ratings should change as the video progresses\n"
        "   • Early scenes might feel neutral, later scenes more emotional\n"
        "   • Emotional intensity can build or fade moment by moment\n\n"

        "───────────────────────────────────────────────────────────────────────\n"
        "                    WHAT INFLUENCES YOUR RATING?                       \n"
        "───────────────────────────────────────────────────────────────────────\n\n"

        "VISUAL ELEMENTS:\n"
        "• Lighting: Bright/warm → more positive; Dark/cold → more negative\n"
        "• Colors: Vibrant/warm colors vs. muted/cold tones\n"
        "• Composition: Spacious/balanced vs. claustrophobic/chaotic\n"
        "• Subject matter: What you're actually seeing\n\n"

        "EMOTIONAL CONTENT:\n"
        "• Character expressions: Smiles, tears, anger, fear\n"
        "• Interactions: Connection, conflict, support, rejection\n"
        "• Body language: Open/relaxed vs. closed/tense\n"
        "• Actions: What characters are doing\n\n"

        "CONTEXT & ATMOSPHERE:\n"
        "• Story progression: Where are we in the narrative?\n"
        "• Music/sound: Uplifting, tense, melancholic (if you can infer)\n"
        "• Pacing: Calm vs. intense\n"
        "• Overall mood: The \"feeling\" of the scene\n\n"

        "───────────────────────────────────────────────────────────────────────\n"
        "                              OUTPUT FORMAT                             \n"
        "───────────────────────────────────────────────────────────────────────\n\n"

        "Valence_rating: <single integer from -10 to +10>\n\n"

        "Valence_reasoning: <2-4 sentences explaining:\n"
        "   1. What do you feel emotionally when viewing this scene?\n"
        "   2. What specific elements (visual, emotional, contextual) contribute?\n"
        "   3. Why this specific intensity level?\n"
        "   4. How would most viewers likely feel?>\n\n"

        "───────────────────────────────────────────────────────────────────────\n"
        "                                EXAMPLES                                \n"
        "───────────────────────────────────────────────────────────────────────\n\n"

        "Example 1: STRONGLY POSITIVE (+7)\n"
        "Valence_rating: +7\n"
        "Valence_reasoning: I feel genuinely happy and uplifted watching this scene. \n"
        "The boy and dog playing together joyfully in bright natural sunlight creates \n"
        "a heartwarming atmosphere. The playful energy, genuine smiles, and warm colors \n"
        "all combine to make me feel good. It's +7 because while definitely joyful, \n"
        "it's not overwhelmingly ecstatic.\n\n"

        "Example 2: MODERATELY NEGATIVE (-4)\n"
        "Valence_rating: -4\n"
        "Valence_reasoning: I feel somewhat sad and uncomfortable watching this scene. \n"
        "The boy sits alone on a couch looking withdrawn and disappointed, with dim \n"
        "interior lighting creating a melancholic mood. His dejected posture and the \n"
        "lonely atmosphere make me feel empathetic sadness. It's -4 rather than more \n"
        "negative because it's melancholic rather than distressing.\n\n"

        "Example 3: NEUTRAL (0)\n"
        "Valence_rating: 0\n"
        "Valence_reasoning: I feel nothing emotionally when viewing this frame. \n"
        "It's just a simple title card with text on a plain background - purely \n"
        "informational with no emotional content. No visual or contextual elements \n"
        "trigger any pleasant or unpleasant feelings. Complete emotional neutrality.\n\n"

        "Example 4: SLIGHTLY POSITIVE (+2)\n"
        "Valence_rating: +2\n"
        "Valence_reasoning: I feel mildly pleasant watching this quiet moment. \n"
        "The scene shows a peaceful indoor setting with soft natural light coming \n"
        "through windows. Nothing dramatically positive happens, but the calm atmosphere \n"
        "and gentle colors create a subtle sense of comfort and tranquility. Just above \n"
        "neutral, like a gentle contentment.\n\n"

        "Example 5: EXTREMELY POSITIVE (+9)\n"
        "Valence_rating: +9\n"
        "Valence_reasoning: I feel overwhelming joy and warmth watching this breakthrough \n"
        "moment. The boy's face radiates pure happiness as he connects with the puppy, \n"
        "his eyes bright with wonder and compassion. The emotional authenticity and the \n"
        "triumph of acceptance over prejudice create profound positive feelings. It's +9 \n"
        "because this is a deeply moving, transformative moment that makes me feel \n"
        "profoundly uplifted.\n\n"

        "Example 6: MODERATELY NEGATIVE (-3)\n"
        "Valence_rating: -3\n"
        "Valence_reasoning: I feel mildly anxious and uncomfortable watching this tense \n"
        "moment. The boy hesitates before opening the box, his expression showing worry \n"
        "and uncertainty. The suspenseful atmosphere and his nervous body language create \n"
        "a sense of unease in me as a viewer. It's -3 because while uncomfortable, it's \n"
        "not deeply distressing - more like nervous anticipation.\n\n"

        "═══════════════════════════════════════════════════════════════════════\n"
        "Remember: Rate YOUR PERSONAL emotional response moment-by-moment. Focus on \n"
        "how the scene makes YOU feel (pleasant vs. unpleasant), not on objective \n"
        "description. Use the full -10 to +10 range to capture emotional intensity.\n"
        "═══════════════════════════════════════════════════════════════════════"
    )

def postprocess_valence(text):
    """Extract Valence rating and reasoning from model output - EXACT COPY from video_analysis_valence_tr.py"""
    rating = None
    reasoning = ""

    lines = text.split('\n')

    for i, line in enumerate(lines):
        line_clean = line.strip()
        line_lower = line_clean.lower()

        if 'valence_rating:' in line_lower:
            match = re.search(r'valence_rating:\s*([+-]?\d+)', line_clean, re.IGNORECASE)
            if match:
                try:
                    rating = int(match.group(1))
                    rating = max(-10, min(10, rating))
                except ValueError:
                    rating = None

        elif 'valence_reasoning:' in line_lower:
            reasoning_text = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
            for j in range(i + 1, min(i + 6, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    break
                if any(keyword in next_line.lower() for keyword in ['valence_', 'rating:', 'example', '═══']):
                    break
                reasoning_text += " " + next_line
            reasoning = reasoning_text.strip()
            break

    if rating is None:
        for line in lines:
            rating_patterns = [
                r'rating[:\s]*([+-]?\d+)',
                r'rate[:\s]+([+-]?\d+)',
                r'score[:\s]*([+-]?\d+)',
                r'([+-]?\d+)\s*/\s*10',
                r'([+-]?\d+)\s*out\s*of\s*10'
            ]
            for pattern in rating_patterns:
                match = re.search(pattern, line.lower())
                if match:
                    try:
                        rating = int(match.group(1))
                        rating = max(-10, min(10, rating))
                        break
                    except ValueError:
                        continue
            if rating is not None:
                break

        if rating is None:
            for line in lines:
                numbers = re.findall(r'([+-]?\d+)', line)
                for num_str in numbers:
                    try:
                        num = int(num_str)
                        if -10 <= num <= 10:
                            rating = num
                            break
                    except ValueError:
                        continue
                if rating is not None:
                    break

    if not reasoning:
        text_lines = [line.strip() for line in lines if line.strip() and
                     not line.strip().lower().startswith('valence_rating:') and
                     len(line.strip()) > 15]
        if text_lines:
            reasoning = " ".join(text_lines[:4])

    if rating is None:
        rating = 0
    if not reasoning:
        reasoning = "Unable to extract clear reasoning from model output."

    reasoning = re.sub(r'\s+', ' ', reasoning)
    reasoning = reasoning[:1000]

    return rating, reasoning

def extract_frames_by_tr(video_path, tr=1.0):
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

def analyze_frame_valence(image, model, processor, device, current_sample_idx=0, description_df=None,
                          context_frames=0, tr=1.0, jump=0):
    """Analyze single frame for Emotional Valence with optional temporal context"""

    context_text = ""
    if context_frames > 0 and description_df is not None and current_sample_idx > 0:
        context_text = build_context_text(current_sample_idx, context_frames,
                                        description_df, tr, jump)

    valence_prompt = build_prompt_valence(context_text)

    print("    [1/3] Generating Valence analysis...", end='', flush=True)
    t_start = time.time()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": valence_prompt},
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
    valence_rating, reasoning = postprocess_valence(raw_output)

    dt = time.time() - t_start
    print(f" Done ({dt:.2f}s)")

    return valence_rating, reasoning, raw_output

def main():
    parser = argparse.ArgumentParser(description='Gemma 3 Emotional Valence Analysis with Context')
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
    print("GEMMA 3 EMOTIONAL VALENCE ANALYSIS WITH TEMPORAL CONTEXT")
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
    print(f"Step {4 if args.context_frames == 0 else '4'}/5: Analyzing Emotional Valence...")
    print("=" * 80)

    results = []
    total_time = 0

    for idx, (frame_idx, timestamp, image) in enumerate(frames):
        print(f"\n▶ Frame {idx+1}/{len(frames)} (TR timepoint: {timestamp:.1f}s)")

        t0 = time.time()
        rating, reasoning, raw_output = analyze_frame_valence(
            image, model, processor, device,
            current_sample_idx=idx,
            description_df=description_df,
            context_frames=args.context_frames,
            tr=args.tr,
            jump=args.jump
        )
        dt = time.time() - t0
        total_time += dt

        rating_str = f"+{rating}" if rating > 0 else str(rating)
        print(f"    [2/3] Time: {dt:.1f}s")
        print(f"    [3/3] Valence Rating: {rating_str}/10")
        print(f"          Reasoning: {reasoning[:80]}..." if len(reasoning) > 80 else f"          Reasoning: {reasoning}")

        if idx % 5 == 0 and torch.cuda.is_available():
            print(f"          VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")

        results.append({
            'index': idx,
            'timestamp': timestamp,
            'valence_rating': rating,
            'valence_reasoning': reasoning,
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
            rating_str = f"+{r['valence_rating']}" if r['valence_rating'] > 0 else str(r['valence_rating'])
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(t1)} --> {format_srt_time(t2)}\n")
            f.write(f"Valence: {rating_str}/10\n")
            f.write(f"{r['valence_reasoning']}\n\n")

    print(f"SRT: {args.srt}")

    with open(args.csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_index', 'timestamp_sec', 'valence_rating', 'valence_reasoning', 'raw_output'])
        for r in results:
            w.writerow([
                r['index'],
                f"{r['timestamp']:.3f}",
                r['valence_rating'],
                r['valence_reasoning'],
                r['raw']
            ])

    print(f"CSV: {args.csv}")

    ratings = [r['valence_rating'] for r in results]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    max_rating = max(ratings) if ratings else 0
    min_rating = min(ratings) if ratings else 0

    positive_frames = sum(1 for r in ratings if r > 0)
    negative_frames = sum(1 for r in ratings if r < 0)
    neutral_frames = sum(1 for r in ratings if r == 0)

    print(f"\nValence Statistics:")
    print(f"  Context frames used: {args.context_frames}")
    if args.context_frames > 0:
        print(f"  Jump parameter: {args.jump} (step size: {args.jump + 1})")
    print(f"  TR used: {args.tr}s")
    print(f"  Frames processed: {len(results)}")
    print(f"  Average Rating: {avg_rating:+.2f}/10")
    print(f"  Max Rating: {max_rating:+}/10")
    print(f"  Min Rating: {min_rating:+}/10")
    print(f"  Positive (>0): {positive_frames}/{len(results)} ({100*positive_frames/len(results) if results else 0:.1f}%)")
    print(f"  Negative (<0): {negative_frames}/{len(results)} ({100*negative_frames/len(results) if results else 0:.1f}%)")
    print(f"  Neutral (=0): {neutral_frames}/{len(results)} ({100*neutral_frames/len(results) if results else 0:.1f}%)")

    print("\n" + "=" * 80)
    print("Valence Analysis Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
