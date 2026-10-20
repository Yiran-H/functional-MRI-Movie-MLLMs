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
import copy
import time
import csv
import argparse
import pandas as pd
import torch
from PIL import Image
import cv2

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def load_description_csv(csv_path):
    """Load description CSV file into pandas DataFrame"""
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
    EXACT COPY from qwen3vl_tom_context_jump.py / gemma_tom_context.py
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
    context_lines.append("and current emotional atmosphere.")
    context_lines.append("")
    context_lines.append("Use this story progression to inform your Emotional Valence rating.")
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
    context_lines.append("rate the Emotional Valence for THIS current moment.")
    context_lines.append("")
    context_lines.append("═" * 77)
    context_lines.append("")

    return "\n".join(context_lines)

def build_prompt_valence(context_text=""):
    """
    Emotional Valence Prompt - Updated to -10 to +10 scale
    Based on SAM (Self-Assessment Manikin) methodology and continuous rating paradigm
    Focus on viewer's SUBJECTIVE EMOTIONAL EXPERIENCE

    Args:
        context_text: Optional temporal context from previous frames
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
        
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        "0: NEUTRAL BASELINE\n"
        "• No emotional response - emotionally flat\n"
        "• Neither pleasant nor unpleasant\n"
        "• Neutral observation - no feelings triggered\n"
        "• Like watching informational text or blank screen\n"
        "Example: Title cards, credits, empty scenes, mundane objects\n\n"
        
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
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
    t = text.strip().replace("\r\n", "\n")
    rating = None
    reasoning = ""
    
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        line_lower = line_clean.lower()
        
        # Extract Valence rating
        if 'valence_rating' in line_lower or ('rating' in line_lower and rating is None):
            # Try to find a number -10 to +10 (including + and - signs)
            numbers = re.findall(r'[+-]?\d+', line_clean)
            if numbers:
                num = int(numbers[0])
                if -10 <= num <= 10:
                    rating = num
        
        # Extract Valence reasoning
        elif 'valence_reasoning' in line_lower or 'reasoning' in line_lower:
            # Get the content after the colon
            if ':' in line_clean:
                reasoning = line_clean.split(':', 1)[1].strip()
            # Also check next lines for continuation
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not next_line.lower().startswith(('valence_', 'rating')):
                    reasoning += " " + next_line
    
    # Fallback: if rating not found, try to extract from anywhere in text
    if rating is None:
        # Look for patterns like "rating: -3" or "score: +2"
        rating_patterns = [
            r'(?:rating|score):\s*([+-]?\d+)',
            r'([+-]?\d+)\s*/\s*10',
            r'\b([+-]?\d+)\b'  # Any number from -10 to +10
        ]
        for pattern in rating_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                num = int(match.group(1))
                if -10 <= num <= 10:
                    rating = num
                    break
    
    # If still no rating, default to 0 (neutral)
    if rating is None:
        rating = 0
    
    # If no reasoning found, extract from text
    if not reasoning:
        # Take the longest sentence that's not about the rating
        sentences = re.split(r'[.!?]\s+', text)
        candidates = [s for s in sentences if len(s) > 20 and 'rating' not in s.lower()]
        if candidates:
            reasoning = candidates[0].strip() + "."
        else:
            reasoning = "Unable to extract reasoning from model output."
    
    # Clean up reasoning
    reasoning = reasoning.strip()
    if not reasoning.endswith('.'):
        reasoning += '.'
    
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

def analyze_frame_valence(image, tokenizer, model, image_processor, device, current_sample_idx=0,
                         description_df=None, context_frames=0, tr=1.0, jump=0):
    """Analyze frame for Emotional Valence with optional temporal context"""

    context_text = ""
    if context_frames > 0 and description_df is not None and current_sample_idx > 0:
        context_text = build_context_text(current_sample_idx, context_frames,
                                        description_df, tr, jump)

    print("    [1/5] Preprocessing...", end='', flush=True)
    t_start = time.time()
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [img.to(dtype=torch.float16, device=device) for img in image_tensor]
    image_sizes = [image.size]
    print(f" Done ({time.time()-t_start:.2f}s)")

    print("    [2/5] Building Valence prompt...", end='', flush=True)
    t_start = time.time()
    question = DEFAULT_IMAGE_TOKEN + "\n" + build_prompt_valence(context_text)
    conv = copy.deepcopy(conv_templates["llava_llada"])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    print(f" Done ({time.time()-t_start:.2f}s)")
    
    print(f"    [3/5] Ready...", end='', flush=True)
    print(f" Done (0.00s)")
    
    print("    [4/5] Generating (Fast-dLLM)...", end='', flush=True)
    t_start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            steps=128,
            gen_length=128,
            block_length=128,
            tokenizer=tokenizer,
            stopping_criteria=['<|eot_id|>'],
            prefix_refresh_interval=32,
            threshold=1,
        )
    
    print(f" Done ({time.time()-t_start:.2f}s) ⚡")
    
    print("    [5/5] Decoding...", end='', flush=True)
    t_start = time.time()
    raw = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    rating, reasoning = postprocess_valence(raw)
    print(f" Done ({time.time()-t_start:.2f}s)")
    
    return rating, reasoning, raw

def main():
    parser = argparse.ArgumentParser(
        description="LLaDA-V Emotional Valence Analysis with fMRI TR support (-10 to +10 scale)",
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
    parser.add_argument("--model-id", default="GSAI-ML/LLaDA-V")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-frames", type=int, default=0,
                       help="Number of previous frames for temporal context (default: 0)")
    parser.add_argument("--jump", type=int, default=0,
                       help="Jump parameter for context sampling: 0=consecutive, 1=skip 1, 2=skip 2, etc.")
    parser.add_argument("--description-csv",
                       help="Path to description CSV file for temporal context")
    parser.add_argument("--test-first-n", type=int, default=None,
                        help="Test mode: only process first N frames")

    args = parser.parse_args()

    if args.context_frames > 0 and not args.description_csv:
        parser.error("--context-frames requires --description-csv")

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
    
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_id, None, "llava_llada", 
        attn_implementation="sdpa", device_map=device
    )
    model.eval()
    
    print("⚡ Registering Fast-dLLM hook...")
    try:
        register_fast_dllm_hook(model)
        print("✓ Fast-dLLM enabled")
    except Exception as e:
        print(f"⚠️  Fast-dLLM failed: {e}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("✓ Model loaded\n")

    description_df = None
    if args.context_frames > 0:
        print("=" * 80)
        print("Loading description CSV for temporal context...")
        print("=" * 80)
        description_df = load_description_csv(args.description_csv)
        print()

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
            image, tokenizer, model, image_processor, device,
            current_sample_idx=idx,
            description_df=description_df,
            context_frames=args.context_frames,
            tr=args.tr,
            jump=args.jump,
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
