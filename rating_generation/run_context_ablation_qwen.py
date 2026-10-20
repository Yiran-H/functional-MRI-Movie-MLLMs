# -*- coding: utf-8 -*-
"""
run_context_ablation_qwen.py — Driver for the Qwen3-VL context ablation runs
(ToM + Valence, 3 movies x 4 variants). A plain Python script rather than a
SLURM .sh, since Qwen3-VL is run locally (see --cache-dir default of
qwen3vl_valence.py / qwen3vl_tom_context_jump.py, a Windows path).

Uses the same 2x2 (window x density) variant scheme as Gemma/LLaDA-V --
see context_variants.py. base (no context) already exists from the
original runs (qwen_{MV}_tom.csv / qwen_{MV}_valence.csv).

Usage:
    python run_context_ablation_qwen.py --movies-root . [--cache-dir C:\\Qwen3-VL\\models]

Edit MOVIES/FILES/TRS below if your video paths differ.
"""
import argparse
import os
import subprocess
import sys

from context_variants import context_params, VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))

MOVIES = ["TP", "DM", "PC"]
FILES = {"TP": "movie/tp.mp4", "DM": "movie/dm.mp4", "PC": "movie/pc.mov"}
TRS = {"TP": 0.8, "DM": 0.8, "PC": 1.0}
MEASURES = ["tom", "valence"]
SCRIPT = {"tom": "qwen3vl_tom_context_jump.py", "valence": "qwen3vl_valence.py"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies-root", default=".",
                     help="Directory containing movie/ and embedding/ (default: current dir)")
    ap.add_argument("--cache-dir", default="C:\\Qwen3-VL\\models")
    ap.add_argument("--only-movie", choices=MOVIES)
    ap.add_argument("--only-measure", choices=MEASURES)
    args = ap.parse_args()

    for mv in MOVIES:
        if args.only_movie and mv != args.only_movie:
            continue
        tr = TRS[mv]
        video = os.path.join(args.movies_root, FILES[mv])
        outdir = os.path.join(args.movies_root, "embedding", f"{mv}_qwen")
        os.makedirs(outdir, exist_ok=True)
        desc_csv = os.path.join(outdir, f"qwen_{mv}_description.csv")

        if not os.path.exists(desc_csv):
            print(f"ERROR: missing description CSV: {desc_csv}", file=sys.stderr)
            sys.exit(1)

        for measure in MEASURES:
            if args.only_measure and measure != args.only_measure:
                continue
            script = os.path.join(HERE, SCRIPT[measure])

            for label in VARIANTS:
                p = context_params(mv, label)
                srt = os.path.join(outdir, f"qwen_{mv}_{measure}_{label}.srt")
                csv_out = os.path.join(outdir, f"qwen_{mv}_{measure}_{label}.csv")

                print(f"===== [{mv} {measure}] {label} "
                      f"(context_frames={p['context_frames']}, jump={p['jump']}, "
                      f"~{p['window_sec']:.1f}s) =====")

                cmd = [
                    sys.executable, script,
                    "--video", video,
                    "--srt", srt,
                    "--csv", csv_out,
                    "--tr", str(tr),
                    "--context-frames", str(p["context_frames"]),
                    "--jump", str(p["jump"]),
                    "--description-csv", desc_csv,
                    "--cache-dir", args.cache_dir,
                ]
                subprocess.run(cmd, check=True)

    print("All Qwen3-VL context ablation variants done.")


if __name__ == "__main__":
    main()
