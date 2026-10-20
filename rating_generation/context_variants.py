# -*- coding: utf-8 -*-
"""
context_variants.py — Shared config for the context-injection ablation grid
used by LLaDA-V / Qwen3-VL / Gemma 3 generation scripts and by
rating_benchmark/context_ablation_analysis*.py.

WHY THIS EXISTS
----------------
The old scheme (context3, context5, ctx5_j1, ctx5_j2) defined variants by a
raw sample COUNT and jump. Because TP/DM are frame-sampled at tr=0.8s during
generation but PC is sampled at tr=1.0s, the same (context_frames, jump) pair
meant a different real-time lookback window per movie:
    context3   -> TP/DM 2.4s   vs PC 3.0s
    context5   -> TP/DM 4.0s   vs PC 5.0s
    ctx5_j1    -> TP/DM 8.0s   vs PC 10.0s
    ctx5_j2    -> TP/DM 12.0s  vs PC 15.0s
Not comparable across movies -- and an earlier revision of this file made it
worse by crossing window length with sampling density (short_dense vs
short_sparse), which produced two "different" variants covering the exact
same real-time window -- not useful when the only thing we actually want to
vary is HOW FAR BACK the context reaches.

NEW SCHEME: single axis (window length only), exact cross-movie alignment
----------------------------------------------------------------------------
Each movie uses a FIXED context_frames count (chosen once per movie, not per
variant) and only `jump` varies across the 4 variants, exactly like the old
ctx5_j1/ctx5_j2 mechanism (fixed sample count, stride reaches further back).
The fixed frame count is chosen per movie so that BASE_FRAMES * tr is equal
across movies:
    TP/DM: 5 frames * 0.8s tr = 4.0s
    PC:    4 frames * 1.0s tr = 4.0s
That 4.0s baseline then scales by an integer factor (jump+1) for every
variant, so the realized window is EXACTLY equal across TP/DM/PC at every
rung -- no rounding, no approximation:

    variant   jump   window (all movies)
    ctx_4s    0      4.0s
    ctx_8s    1      8.0s
    ctx_12s   2      12.0s
    ctx_16s   3      16.0s

Trade-off: PC carries one fewer context item (4 vs 5) than TP/DM at every
rung, in exchange for exact time-alignment instead of an approximation.
There is no exact cross-movie match shorter than 4.0s given TP/DM's 0.8s
grid and PC's 1.0s grid (their smallest common multiple).
"""

MOVIE_GEN_TR = {"TP": 0.8, "DM": 0.8, "PC": 1.0}
BASE_FRAMES = {"TP": 5, "DM": 5, "PC": 4}  # chosen so BASE_FRAMES * tr == 4.0s for every movie

VARIANTS = ["ctx_4s", "ctx_8s", "ctx_12s", "ctx_16s"]
_VARIANT_JUMP = {"ctx_4s": 0, "ctx_8s": 1, "ctx_12s": 2, "ctx_16s": 3}

VARIANT_COLOR = {
    "base": "#999999",
    "ctx_4s": "#4C72B0",
    "ctx_8s": "#55A868",
    "ctx_12s": "#C44E52",
    "ctx_16s": "#8172B2",
}


def context_params(movie, variant):
    """Return dict(context_frames=..., jump=..., window_sec=...) for a movie/variant."""
    tr = MOVIE_GEN_TR[movie]
    frames = BASE_FRAMES[movie]
    jump = _VARIANT_JUMP[variant]
    window_sec = frames * (jump + 1) * tr
    return dict(context_frames=frames, jump=jump, window_sec=round(window_sec, 2))


if __name__ == "__main__":
    # Quick CLI lookup used by shell launchers:
    #   python context_variants.py TP ctx_8s -> "5 1"
    import sys
    mv, variant = sys.argv[1], sys.argv[2]
    p = context_params(mv, variant)
    print(f"{p['context_frames']} {p['jump']}")
