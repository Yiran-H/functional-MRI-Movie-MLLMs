"""
paired_source_tests.py — Direct paired comparisons between rating sources
(Human, LLaDA-V, Qwen3-VL, Gemma 3) on held-out encoding performance.

Counting FDR-significant ROIs per source (analyze_full_solo_roi_significance.py)
tests each source against zero separately; it does not test whether one source
predicts better than another. Here each pair of sources is compared directly on
the same participants:

    d[s, r] = score_A[s, r] - score_B[s, r]

where score is the single-source (solo) model's held-out Pearson r, rebuilt
exactly from the saved arrays as in analyze_full_solo_roi_significance.py:
    solo_human = results_r2 - unique_r2s[:, model_idx]
    solo_model = results_r2 - unique_r2s[:, human_idx]
solo_human is identical across the three model files of a movie x measure
(it never sees model ratings), so it is taken from the first file.

Tests (two-sided sign-flip permutation on the paired differences, 5000
resamples, same seed as the pipeline):
    whole-brain : d averaged over ROIs per participant -> one test per pair,
                  BH-FDR across all pairs within each movie x measure
    ROI-level   : one test per ROI, BH-FDR across the 114 ROIs of each pair

Usage:
    python paired_source_tests.py [results_dir] [tag]
Output:
    <results_dir>/paired_source_tests_pearson_r.csv
"""
import glob
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "human_vs_model_results", "diagnostic_singledelay")
TAG = sys.argv[2] if len(sys.argv) > 2 else "singledelay"
N_PERM = 5000
SEED = 42
LABEL = {"human": "Human", "llada": "LLaDA-V", "qwen": "Qwen3-VL", "gemma": "Gemma 3"}

PAT = re.compile(r"hvm_(?P<movie>tp|dm|pc)_(?P<model>llada|qwen|gemma)_(?P<measure>tom|valence)"
                 r"_sigma[\d.]+_chunk\d+_" + TAG + r"_pearson_r_unique_shared\.npz$")


def signflip_p(d, rng):
    """Two-sided sign-flip permutation p for mean(d) != 0. d: [S] or [S, R]."""
    d = np.asarray(d, float)
    obs = np.abs(d.mean(0))
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, d.shape[0]))
    null = np.abs(np.tensordot(signs, d, axes=(1, 0)) / d.shape[0])
    return (1 + (null >= obs).sum(0)) / (N_PERM + 1)


solo = {}   # (movie, measure) -> {source: [S, R]}
for f in sorted(glob.glob(os.path.join(DIR, "*.npz"))):
    m = PAT.search(os.path.basename(f))
    if not m:
        continue
    d = np.load(f)
    order = list(d["feature_order"])
    hi = order.index(f"human_{m['measure']}")
    mi = order.index(f"model_{m['measure']}")
    full, uniq = d["results_r2"], d["unique_r2s"]
    srcs = solo.setdefault((m["movie"], m["measure"]), {})
    srcs.setdefault("human", full - uniq[:, mi, :])
    srcs[m["model"]] = full - uniq[:, hi, :]

rng = np.random.default_rng(SEED)
rows = []
for (movie, measure), srcs in sorted(solo.items()):
    names = [s for s in ["human", "llada", "qwen", "gemma"] if s in srcs]
    block = []
    for a, b in itertools.combinations(names, 2):
        d = srcs[a] - srcs[b]                      # [S, R]
        p_wb = signflip_p(d.mean(1), rng)
        p_roi = signflip_p(d, rng)
        rej = multipletests(p_roi, alpha=0.05, method="fdr_bh")[0]
        mean_d = d.mean(0)
        block.append(dict(
            movie=movie, measure=measure, source_a=LABEL[a], source_b=LABEL[b],
            n_subjects=d.shape[0],
            mean_r_a=float(srcs[a].mean()), mean_r_b=float(srcs[b].mean()),
            mean_diff=float(d.mean()), p_wholebrain=float(p_wb),
            n_roi_a_gt_b=int((rej & (mean_d > 0)).sum()),
            n_roi_b_gt_a=int((rej & (mean_d < 0)).sum())))
    p_fdr = multipletests([r["p_wholebrain"] for r in block], method="fdr_bh")[1]
    for r, q in zip(block, p_fdr):
        r["q_wholebrain_fdr"] = float(q)
    rows += block

out = pd.DataFrame(rows)
out_csv = os.path.join(DIR, "paired_source_tests_pearson_r.csv")
out.to_csv(out_csv, index=False)
pd.set_option("display.width", 220)
print(out.round(4).to_string(index=False))
print(f"\nSaved -> {out_csv}")
