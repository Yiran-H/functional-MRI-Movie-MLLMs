# -*- coding: utf-8 -*-
"""
video_analysis_tom.py (Ultra-detailed version)
Theory of Mind (ToM) Analysis
Rating: -10 to +10 with EACH level precisely defined
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
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def build_prompt_tom():
    """
    Theory of Mind Prompt - Focus on EMOTIONAL ENGAGEMENT and EMPATHY
    Based on: ability to understand and infer what others might be thinking/feeling
    """
    return (
        "[Answer ONLY in English]\n\n"
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
        
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        "0: NEUTRAL BASELINE\n"
        "• No emotional engagement or connection\n"
        "• Cannot perceive any mental states\n"
        "• Emotionally flat scene / transition moment\n"
        "Example: Black screen, title card, empty scene, or emotionally neutral moment\n\n"
        
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
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
        
        "ToM_rating: <single integer from -10 to +10>\n\n"
        
        "ToM_reasoning: <2-4 sentences explaining:\n"
        "   1. What do you perceive about the character's mental/emotional state?\n"
        "   2. How strong is your emotional connection/understanding?\n"
        "   3. What specific cues (or lack thereof) drive your rating?\n"
        "   4. Why this specific number (e.g., why +7 not +5 or +9)?>\n\n"
        
        "───────────────────────────────────────────────────────────────────────\n"
        "                                EXAMPLES                                \n"
        "───────────────────────────────────────────────────────────────────────\n\n"
        
        "Example 1: HIGH POSITIVE (+9)\n"
        "ToM_rating: +9\n"
        "ToM_reasoning: I feel a profound connection to the boy's emotional journey. \n"
        "His face reveals a complex mix of surprise, wonder, and dawning compassion as \n"
        "he discovers the three-legged puppy. I can sense his internal shift from shock \n"
        "to empathy - it's palpable. The vulnerability in his expression makes me feel \n"
        "what he's feeling. This is +9 (not +10) because while deeply moving, it's still \n"
        "a developing realization rather than a complete transformative breakthrough.\n\n"
        
        "Example 2: MODERATE POSITIVE (+5)\n"
        "ToM_rating: +5\n"
        "ToM_reasoning: I can clearly see the character is curious and slightly anxious \n"
        "as they peek into the box. Their furrowed brow and forward lean communicate \n"
        "hesitation mixed with interest. I understand what they're feeling - that \"should \n"
        "I look?\" moment. Solid emotional connection but not deeply complex. It's +5 \n"
        "because the emotion is clear and relatable, but straightforward without multiple layers.\n\n"
        
        "Example 3: NEUTRAL (0)\n"
        "ToM_rating: 0\n"
        "ToM_reasoning: This frame shows only a black screen as the video begins. There \n"
        "is no character present, so there are no mental states for me to perceive or \n"
        "connect with. No emotional engagement is happening at this moment. Pure baseline.\n\n"
        
        "Example 4: MODERATE NEGATIVE (-5)\n"
        "ToM_rating: -5\n"
        "ToM_reasoning: The character's back is completely to the camera, making it \n"
        "impossible to see their face or gauge their emotional state. I can see their \n"
        "posture is neutral, giving me no clear emotional cues. I feel disconnected - \n"
        "I want to understand what they're thinking but the viewing angle blocks me. \n"
        "It's -5 because there's clear physical obstruction preventing emotional access.\n\n"
        
        "Example 5: LOW POSITIVE (+2)\n"
        "ToM_rating: +2\n"
        "ToM_reasoning: I can barely detect some emotional presence - the character's \n"
        "expression is very subtle, perhaps slightly pensive? The lighting and angle make \n"
        "it hard to read clearly. I sense something is there emotionally, but it's vague \n"
        "and ambiguous. Minimal connection, just above neutral baseline.\n\n"
        
        "Example 6: STRONG POSITIVE (+8)\n"
        "ToM_rating: +8\n"
        "ToM_reasoning: The character's joy is radiating through their entire being - \n"
        "bright eyes, genuine smile, relaxed shoulders, playful energy. I feel their \n"
        "happiness viscerally. I can infer they're experiencing relief and delight, and \n"
        "I predict they'll engage more openly now. Strong empathic resonance - their joy \n"
        "is contagious. It's +8 because while the emotion is clear and affecting, it's a \n"
        "single strong emotion rather than a complex multi-layered state.\n\n"
        
        "═══════════════════════════════════════════════════════════════════════\n"
        "Remember: Rate YOUR subjective experience of emotional connection, not just \n"
        "whether a character is present. Focus on the felt sense of understanding and \n"
        "empathy, which can vary from moment to moment as the story unfolds.\n"
        "═══════════════════════════════════════════════════════════════════════"
    )

def postprocess_tom(text):
    """Extract ToM rating and reasoning - supports -10 to +10"""
    t = text.strip().replace("\r\n", "\n")
    rating = None
    reasoning = ""
    
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        line_lower = line_clean.lower()
        
        if 'tom_rating' in line_lower or ('rating' in line_lower and rating is None):
            numbers = re.findall(r'[+-]?\d+', line_clean)
            if numbers:
                num = int(numbers[0])
                if -10 <= num <= 10:
                    rating = num
        
        elif 'tom_reasoning' in line_lower or 'reasoning' in line_lower:
            if ':' in line_clean:
                reasoning = line_clean.split(':', 1)[1].strip()
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not next_line.lower().startswith(('tom_', 'rating')):
                    reasoning += " " + next_line
    
    if rating is None:
        rating_patterns = [
            r'(?:rating|score):\s*([+-]?\d+)',
            r'([+-]?\d+)\s*/\s*10',
            r'\b([+-]?\d+)\b'
        ]
        for pattern in rating_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                num = int(match.group(1))
                if -10 <= num <= 10:
                    rating = num
                    break
    
    if rating is None:
        rating = 0
    
    if not reasoning:
        sentences = re.split(r'[.!?]\s+', text)
        candidates = [s for s in sentences if len(s) > 20 and 'rating' not in s.lower()]
        if candidates:
            reasoning = candidates[0].strip() + "."
        else:
            reasoning = "Unable to extract reasoning from model output."
    
    reasoning = reasoning.strip()
    if not reasoning.endswith('.'):
        reasoning += '.'
    
    return rating, reasoning

def extract_frames_by_tr(video_path, tr=1.0):
    """
    Extract one frame every `tr` seconds using a target-time accumulator,
    instead of an integer frame-skip count. Native video fps here is a
    non-integer NTSC rate (~23.976), so `int(fps / fps_sample)`-style frame
    skipping truncates and drifts away from the intended interval over a
    long video. Accumulating the target timestamp and grabbing the first
    native frame at or past it keeps every sample within one native frame
    (~0.04s) of the intended `tr` grid, with no cumulative drift.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video info: FPS={fps:.2f}, Total frames={total_frames}, Duration={duration:.2f}s")
    print(f"Sampling every {tr}s")

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
    print(f"Sampled {len(frames)} frames")
    return frames

def analyze_frame_tom(image, model, processor, device):
    """Analyze frame for Theory of Mind"""

    print("    [1/4] Building ToM prompt...", end='', flush=True)
    t_start = time.time()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt_tom()},
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

    print("    [2/4] Generating...", end='', flush=True)
    t_start = time.time()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    outputs = outputs[0][input_len:]

    print(f" Done ({time.time()-t_start:.2f}s)")

    print("    [3/4] Decoding...", end='', flush=True)
    t_start = time.time()
    raw = processor.decode(outputs, skip_special_tokens=True).strip()
    rating, reasoning = postprocess_tom(raw)
    print(f" Done ({time.time()-t_start:.2f}s)")

    return rating, reasoning, raw

def main():
    parser = argparse.ArgumentParser(description="Gemma 3 Theory of Mind Analysis (Ultra-detailed -10 to +10)")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--srt", required=True, help="Path to output SRT subtitle file")
    parser.add_argument("--csv", required=True, help="Path to output CSV data file")
    parser.add_argument("--model-id", default="google/gemma-3-12b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tr", type=float, default=1.0, help="Sampling interval in seconds (one frame every N seconds)")
    parser.add_argument("--test-first-n", type=int, default=None, help="Test with first N frames only")
    args = parser.parse_args()
    
    device = args.device
    
    print("=" * 80)
    print("🧠 THEORY OF MIND (ToM) ANALYSIS - ULTRA-DETAILED VERSION")
    print("=" * 80)
    print("Task: Evaluate perceptibility of character's mental states")
    print("Rating Scale: -10 to +10 (21 distinct levels)")
    print("=" * 80)
    print()
    
    print("=" * 80)
    print("Step 1/4: Loading model...")
    print("=" * 80)
    
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("✓ Model loaded\n")
    
    print("=" * 80)
    print("Step 2/4: Extracting frames...")
    print("=" * 80)
    frames = extract_frames_by_tr(args.video, args.tr)
    
    if args.test_first_n:
        frames = frames[:args.test_first_n]
        print(f"⚠️  Test mode: {len(frames)} frames only")
    
    print(f"✓ Extracted: {len(frames)} frames\n")
    
    print("=" * 80)
    print("Step 3/4: Analyzing Theory of Mind...")
    print("=" * 80)
    
    results = []
    total_time = 0
    
    for idx, (frame_idx, timestamp, image) in enumerate(frames):
        print(f"\n▶ Frame {idx+1}/{len(frames)} (t={timestamp:.1f}s)")
        
        t0 = time.time()
        rating, reasoning, raw = analyze_frame_tom(
            image, model, processor, device
        )
        dt = time.time() - t0
        total_time += dt
        
        print(f"  ✓ Time: {dt:.1f}s")
        rating_str = f"+{rating}" if rating > 0 else str(rating)
        print(f"  🧠 ToM Rating: {rating_str}/10")
        print(f"  💭 Reasoning: {reasoning[:80]}..." if len(reasoning) > 80 else f"  💭 Reasoning: {reasoning}")
        
        if idx % 10 == 0 and torch.cuda.is_available():
            print(f"  💾 VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        results.append({
            'index': frame_idx,
            'timestamp': timestamp,
            'tom_rating': rating,
            'tom_reasoning': reasoning,
            'raw': raw
        })
    
    avg = total_time / len(frames) if frames else 0
    print(f"\n✓ Complete!")
    print(f"  Avg: {avg:.1f}s/frame")
    print(f"  Total: {total_time/60:.1f} min")
    if args.test_first_n is None:
        print(f"  Est. for 212 frames: {avg*212/60:.1f} min\n")
    
    print("=" * 80)
    print("Step 4/4: Saving results...")
    print("=" * 80)
    
    # Save SRT
    with open(args.srt, 'w', encoding='utf-8') as f:
        for i, r in enumerate(results, 1):
            t1 = r['timestamp']
            t2 = results[i]['timestamp'] if i < len(results) else t1 + 1.0
            rating_str = f"+{r['tom_rating']}" if r['tom_rating'] > 0 else str(r['tom_rating'])
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(t1)} --> {format_srt_time(t2)}\n")
            f.write(f"ToM: {rating_str}/10\n")
            f.write(f"{r['tom_reasoning']}\n\n")
    
    print(f"✓ SRT: {args.srt}")
    
    # Save CSV
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
    
    print(f"✓ CSV: {args.csv}")
    
    # Statistics
    ratings = [r['tom_rating'] for r in results]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    max_rating = max(ratings) if ratings else 0
    min_rating = min(ratings) if ratings else 0
    
    positive_frames = sum(1 for r in ratings if r > 0)
    negative_frames = sum(1 for r in ratings if r < 0)
    neutral_frames = sum(1 for r in ratings if r == 0)
    
    print(f"\n📊 ToM Statistics:")
    print(f"  Average Rating: {avg_rating:+.2f}/10")
    print(f"  Max Rating: {max_rating:+}/10")
    print(f"  Min Rating: {min_rating:+}/10")
    print(f"  Positive (>0): {positive_frames}/{len(ratings)} ({100*positive_frames/len(ratings):.1f}%)")
    print(f"  Negative (<0): {negative_frames}/{len(ratings)} ({100*negative_frames/len(ratings):.1f}%)")
    print(f"  Neutral (=0): {neutral_frames}/{len(ratings)} ({100*neutral_frames/len(ratings):.1f}%)")
    print(f"  High ToM (≥+7): {sum(1 for r in ratings if r >= 7)}/{len(ratings)}")
    print(f"  Highly obscured (≤-5): {sum(1 for r in ratings if r <= -5)}/{len(ratings)}")
    
    print("\n" + "=" * 80)
    print("✓✓✓ Theory of Mind Analysis Complete! ✓✓✓")
    print("=" * 80)

if __name__ == "__main__":
    main()
