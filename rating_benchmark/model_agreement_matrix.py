"""
model_agreement_matrix.py — Replaces the 6-bar panel A of the inter-model
composite with 6 correlation matrices (3 movies x 2 measures), each a
4x4 lower-triangular heatmap (including the diagonal) over {Human, LLaDA-V,
Qwen3-VL, Gemma 3} showing Pearson r between every pair: best-lag for
human-model pairs (human response delay), lag 0 for model-model pairs.

Reuses results/bestlag_summary.csv (same 6 pairwise r's used everywhere
else in this project).

Output (rating_benchmark/results/):
  fig_model_agreement_matrix.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "results")
best = pd.read_csv(os.path.join(RES, "bestlag_summary.csv"))

MOVIES = ["TP", "DM", "PC"]
MOVIE_TITLE = {"TP": "The Present", "DM": "Despicable Me", "PC": "Partly Cloudy"}
MEASURES = ["tom", "valence"]
# Lower-triangle layout INCLUDING the diagonal: both axes = Human + 3 models.
LABELS = ["Human", "LLaDA-V", "Qwen3-VL", "Gemma 3"]
N = len(LABELS)

# Classic ColorBrewer-style RdBu diverging scale (deep blue -> white -> deep
# red), matching the reference figure.
CMAP = plt.cm.RdBu_r
NORM = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

FONT = {"font.family": "Times New Roman", "font.size": 15}


def get_r(li, lj, mv, measure):
    if li == lj:
        return 1.0
    name1, name2 = f"{li}–{lj}", f"{lj}–{li}"
    sub = best[(best.movie == mv) & (best.measure == measure)]
    row = sub[sub.pair.isin([name1, name2])]
    return float(row["pearson_r"].iloc[0]) if len(row) else np.nan


def build_matrix(mv, measure):
    M = np.full((N, N), np.nan)
    for i, li in enumerate(LABELS):
        for j, lj in enumerate(LABELS):
            if j > i:
                continue  # upper half redundant -- lower triangle incl. diagonal only
            M[i, j] = get_r(li, lj, mv, measure)
    return M


with plt.rc_context(FONT):
    fig, axes = plt.subplots(len(MEASURES), len(MOVIES), figsize=(15, 10.5))

    im = None
    for ri, measure in enumerate(MEASURES):
        for ci, mv in enumerate(MOVIES):
            ax = axes[ri][ci]
            M = build_matrix(mv, measure)
            im = ax.imshow(M, cmap=CMAP, norm=NORM)
            for i in range(N):
                for j in range(N):
                    v = M[i, j]
                    if not np.isfinite(v):
                        continue
                    txt_color = "white" if abs(v) > 0.6 else "#1a1a1a"
                    is_diag = (i == j)
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=12.5, color=txt_color,
                            fontweight="bold" if is_diag else "normal")
            # white grid lines between cells, like the reference figure
            ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2.5)
            ax.tick_params(which="minor", length=0)
            ax.set_xticks(range(N)); ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=11.5)
            ax.set_yticks(range(N)); ax.set_yticklabels(LABELS, fontsize=11.5)
            ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5)
            ax.set_title(f"{MOVIE_TITLE[mv]} - {measure.upper()}", fontsize=14.5, fontweight="bold")
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.tick_params(which="major", length=0)

    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02, label="Pearson r")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Pearson r", fontsize=14, fontfamily="Times New Roman")

    out = os.path.join(RES, "fig_model_agreement_matrix.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out} (+.pdf)")
