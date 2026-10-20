# -*- coding: utf-8 -*-
"""
video_analysis_valence_tr.py
Emotional Valence Analysis with fMRI TR support
Evaluates viewer's moment-by-moment emotional response to each scene
Rating: -10 to +10 (continuous scale based on SAM methodology)
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

def build_prompt_valence():
    """
    Emotional Valence Prompt - Updated to -10 to +10 scale
    Based on SAM (Self-Assessment Manikin) methodology and continuous rating paradigm
    Focus on viewer's SUBJECTIVE EMOTIONAL EXPERIENCE
    """
    return (
        "[Answer ONLY in English]\n\n"
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
    """Extract Valence rating and reasoning from model output - updated for -10 to +10 scale"""
    rating = None
    reasoning = ""

    lines = text.split('\n')

    for i, line in enumerate(lines):
        line_clean = line.strip()
        line_lower = line_clean.lower()

        # Extract Valence rating
        if 'valence_rating:' in line_lower:
            match = re.search(r'valence_rating:\s*([+-]?\d+)', line_clean, re.IGNORECASE)
            if match:
                try:
                    rating = int(match.group(1))
                    # Clamp to valid range -10 to +10
                    rating = max(-10, min(10, rating))
                except ValueError:
                    rating = None

        # Extract Valence reasoning
        elif 'valence_reasoning:' in line_lower:
            # Get reasoning from this line and potentially next lines
            reasoning_text = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""

            # Collect additional lines that might be part of reasoning
            for j in range(i + 1, min(i + 6, len(lines))):  # Look ahead max 5 lines
                next_line = lines[j].strip()
                if not next_line:  # Empty line
                    break
                if any(keyword in next_line.lower() for keyword in ['valence_', 'rating:', 'example', '═══']):
                    break
                reasoning_text += " " + next_line

            reasoning = reasoning_text.strip()
            break

    # Fallback extraction if standard format not found
    if rating is None:
        # Try to find rating in different formats
        for line in lines:
            # Look for patterns like "Rating: +5" or "I rate this: 7" etc
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

        # Last resort: look for standalone numbers
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
        # Extract any substantial text that might be reasoning
        text_lines = [line.strip() for line in lines if line.strip() and
                     not line.strip().lower().startswith('valence_rating:') and
                     len(line.strip()) > 15]
        if text_lines:
            reasoning = " ".join(text_lines[:4])  # Take first few substantial lines

    # Final fallbacks
    if rating is None:
        rating = 0
    if not reasoning:
        reasoning = "Unable to extract clear reasoning from model output."

    # Clean up reasoning
    reasoning = re.sub(r'\s+', ' ', reasoning)  # Normalize whitespace
    reasoning = reasoning[:1000]  # Limit length for valence (longer than ToM)

    return rating, reasoning

def extract_frames_by_tr(video_path, tr=1.0):
    """
    Extract frames based on fMRI Repetition Time (TR)
    
    Args:
        video_path: Path to video file
        tr: Repetition Time in seconds (e.g., 0.8 or 1.0)
            One frame will be extracted every TR seconds
    
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

def analyze_frame_valence(image, model, processor, device):
    """Analyze frame for Emotional Valence"""

    print("    [1/4] Building Valence prompt...", end='', flush=True)
    t_start = time.time()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt_valence()},
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
    rating, reasoning = postprocess_valence(raw)
    print(f" Done ({time.time()-t_start:.2f}s)")

    return rating, reasoning, raw

def main():
    parser = argparse.ArgumentParser(
        description="Gemma 3 Emotional Valence Analysis with fMRI TR support (-10 to +10 scale)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # For TR = 0.8s
  python video_analysis_valence_tr.py --video video.mp4 --tr 0.8 --csv valence_tr08.csv --srt valence_tr08.srt
  
  # For TR = 1.0s
  python video_analysis_valence_tr.py --video video.mp4 --tr 1.0 --csv valence_tr10.csv --srt valence_tr10.srt
  
  # Test with first 5 frames
  python video_analysis_valence_tr.py --video video.mp4 --tr 0.8 --csv test.csv --srt test.srt --test-first-n 5
        """
    )
    
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--srt", required=True, help="Path to output SRT subtitle file")
    parser.add_argument("--csv", required=True, help="Path to output CSV data file")
    parser.add_argument("--tr", type=float, default=1.0, 
                        help="fMRI Repetition Time in seconds (default: 1.0)")
    parser.add_argument("--model-id", default="google/gemma-3-12b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-first-n", type=int, default=None, 
                        help="Test mode: only process first N frames")
    
    args = parser.parse_args()
    device = args.device
    
    print("=" * 80)
    print("💗 EMOTIONAL VALENCE ANALYSIS with fMRI TR Support")
    print("=" * 80)
    print(f"TR (Repetition Time): {args.tr}s")
    print("Task: Evaluate viewer's moment-by-moment emotional response")
    print("Rating Scale: -10 to +10 (continuous subjective experience)")
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
    print("Step 2/4: Extracting frames based on TR...")
    print("=" * 80)
    frames = extract_frames_by_tr(args.video, args.tr)
    
    if args.test_first_n:
        frames = frames[:args.test_first_n]
        print(f"\n⚠️  Test mode: Processing only first {len(frames)} frames")
    
    print(f"\n✓ Ready to process: {len(frames)} frames\n")
    
    print("=" * 80)
    print("Step 3/4: Analyzing Emotional Valence...")
    print("=" * 80)
    
    results = []
    total_time = 0
    
    for idx, (frame_idx, timestamp, image) in enumerate(frames):
        print(f"\n▶ Frame {idx+1}/{len(frames)} (TR timepoint: {timestamp:.1f}s)")
        
        t0 = time.time()
        rating, reasoning, raw = analyze_frame_valence(
            image, model, processor, device
        )
        dt = time.time() - t0
        total_time += dt
        
        print(f"  ✓ Time: {dt:.1f}s")
        rating_str = f"+{rating}" if rating > 0 else str(rating)
        print(f"  💗 Valence Rating: {rating_str}/10")
        print(f"  📝 Reasoning: {reasoning[:80]}..." if len(reasoning) > 80 else f"  📝 Reasoning: {reasoning}")
        
        if idx % 10 == 0 and torch.cuda.is_available():
            print(f"  💾 VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        results.append({
            'index': frame_idx,
            'timestamp': timestamp,
            'valence_rating': rating,
            'valence_reasoning': reasoning,
            'raw': raw
        })
    
    avg = total_time / len(frames) if frames else 0
    print(f"\n✓ Complete!")
    print(f"  Avg: {avg:.1f}s/frame")
    print(f"  Total: {total_time/60:.1f} min")
    
    # Calculate expected frames for full video
    if frames:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        expected_full = int(duration / args.tr) + 1
        print(f"  Est. for full video ({duration:.1f}s): {expected_full} frames × {avg:.1f}s = {avg*expected_full/60:.1f} min\n")
    
    print("=" * 80)
    print("Step 4/4: Saving results...")
    print("=" * 80)
    
    # Save SRT
    with open(args.srt, 'w', encoding='utf-8') as f:
        for i, r in enumerate(results, 1):
            t1 = r['timestamp']
            t2 = results[i]['timestamp'] if i < len(results) else t1 + args.tr
            rating_str = f"+{r['valence_rating']}" if r['valence_rating'] > 0 else str(r['valence_rating'])
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(t1)} --> {format_srt_time(t2)}\n")
            f.write(f"Valence: {rating_str}/10\n")
            f.write(f"{r['valence_reasoning']}\n\n")
    
    print(f"✓ SRT: {args.srt}")
    
    # Save CSV
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
    
    print(f"✓ CSV: {args.csv}")
    
    # Statistics
    ratings = [r['valence_rating'] for r in results]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    max_rating = max(ratings) if ratings else 0
    min_rating = min(ratings) if ratings else 0
    
    positive_frames = sum(1 for r in ratings if r > 0)
    negative_frames = sum(1 for r in ratings if r < 0)
    neutral_frames = sum(1 for r in ratings if r == 0)
    
    print(f"\n📊 Valence Statistics:")
    print(f"  TR used: {args.tr}s")
    print(f"  Frames processed: {len(results)}")
    print(f"  Average Valence: {avg_rating:+.2f}/10")
    print(f"  Max Valence: {max_rating:+}/10")
    print(f"  Min Valence: {min_rating:+}/10")
    print(f"  Positive (>0): {positive_frames}/{len(ratings)} ({100*positive_frames/len(ratings):.1f}%)")
    print(f"  Negative (<0): {negative_frames}/{len(ratings)} ({100*negative_frames/len(ratings):.1f}%)")
    print(f"  Neutral (=0): {neutral_frames}/{len(ratings)} ({100*neutral_frames/len(ratings):.1f}%)")
    print(f"  Strong positive (≥+6): {sum(1 for r in ratings if r >= 6)}/{len(ratings)}")
    print(f"  Strong negative (≤-6): {sum(1 for r in ratings if r <= -6)}/{len(ratings)}")
    
    print("\n" + "=" * 80)
    print("✓✓✓ Emotional Valence Analysis Complete! ✓✓✓")
    print("=" * 80)

if __name__ == "__main__":
    main()
