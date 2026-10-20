"""
rating_distribution_calibration.py — Do the models use the full -10..+10
rating range like humans, or stay clustered near the middle?

Correlation-based metrics only capture SHAPE agreement; this looks at the
raw VALUE distribution instead. Raw per-frame ratings, NOT Gaussian-smoothed
and NOT z-scored -- smoothing would artificially compress the range and
defeat the purpose of a calibration check; human ratings are never smoothed
in this project either, so this keeps both sides on equal footing.

Produces BOTH the pooled (all-3-movies-together) figure used in the
inter-model composite AND the 3 per-movie figures (baseline/range usage can
differ by movie, so both views are kept).

Outputs (rating_benchmark/results/):
  fig_rating_distribution.png      pooled across TP/DM/PC
  fig_rating_distribution_TP.png   per-movie
  fig_rating_distribution_DM.png
  fig_rating_distribution_PC.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from rating_utils import sample2tr

RES = os.path.join(_HERE, "results")
MOVIES = ["TP", "DM", "PC"]
MOVIE_TITLE = {"TP": "The Present", "DM": "Despicable Me", "PC": "Partly Cloudy"}
MEASURES = ["tom", "valence"]
RATING_COL = {"tom": "tom_rating", "valence": "valence_rating"}

MODELS = ["llada", "qwen", "gemma"]
MODEL_LABEL = {"llada": "LLaDA-V", "qwen": "Qwen3-VL", "gemma": "Gemma 3"}
MODEL_DIR_SUFFIX = {"llada": "final", "qwen": "qwen", "gemma": "gemma"}
MODEL_FILE_PREFIX = {"llada": "", "qwen": "qwen_", "gemma": "gemma_"}
# Human = neutral dark grey (reference); models = validated categorical
# palette (dataviz-skill slots 1-3): blue / orange / aqua.
COLORS = {"human": "#1f1f1f", "llada": "#2a78d6", "qwen": "#eb6834", "gemma": "#1baf7a"}

HUMAN_NPY = {
    "tom": {"TP": "tp_tom_mean.npy", "DM": "dm_tom_mean.npy", "PC": "pc_tom_mean.npy"},
    "valence": {"TP": "tp_valence_mean.npy", "DM": "dm_valence_mean.npy", "PC": "pc_valence_mean.npy"},
}
DT = 0.25


def human_raw(mv, measure):
    lag_h = 0 if mv == "PC" else int(round(8.0 / DT))
    y = np.load(os.path.join(_ROOT, HUMAN_NPY[measure][mv])).astype(float)
    return y[lag_h:] if lag_h > 0 else y


def machine_raw(mv, model, measure):
    sub = MODEL_DIR_SUFFIX[model]
    pre = MODEL_FILE_PREFIX[model]
    csv_path = os.path.join(_ROOT, "embedding", f"{mv}_{sub}", f"{pre}{mv}_{measure}.csv")
    start = 10 if mv in ("TP", "DM") else 0
    return sample2tr(csv_path, start_tr=start, time_col="timestamp_sec",
                      rating_col=RATING_COL[measure], pc_downsample=False)


plt.rcParams.update({"font.family": "Times New Roman", "font.size": 15})

# ═══ pooled figure (all 3 movies together) -- used in the intermodel composite ═══
fig, axes = plt.subplots(1, len(MEASURES), figsize=(18, 8.2))
pooled_rows = []
for ax, measure in zip(axes, MEASURES):
    groups = ["human"] + MODELS
    data = []
    for g in groups:
        pooled = []
        for mv in MOVIES:
            pooled.extend(human_raw(mv, measure) if g == "human" else machine_raw(mv, g, measure))
        pooled = np.asarray(pooled, dtype=float)
        pooled = pooled[np.isfinite(pooled)]
        data.append(pooled)
        pooled_rows.append(dict(measure=measure, source=("Human" if g == "human" else MODEL_LABEL[g]),
                                n=len(pooled), mean=np.mean(pooled), std=np.std(pooled),
                                min=np.min(pooled), max=np.max(pooled)))

    labels = ["Human"] + [MODEL_LABEL[m] for m in MODELS]
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc, g in zip(parts["bodies"], groups):
        pc.set_facecolor(COLORS[g]); pc.set_alpha(0.35)
    bp = ax.boxplot(data, widths=0.15, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=1.5))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(COLORS[g]); patch.set_alpha(0.85)

    ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels(labels, fontsize=30)
    ax.tick_params(axis="y", labelsize=24)
    ax.axhline(0, color="#999999", lw=0.6, ls=":")
    ax.set_ylim(-11, 11)
    ax.set_title(measure.upper(), fontsize=30, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5)
    if measure == MEASURES[0]:
        ax.set_ylabel("Rating (-10 to +10)", fontsize=30)

plt.tight_layout()
out = os.path.join(RES, "fig_rating_distribution.png")
fig.savefig(out, dpi=600, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out} (+.pdf)")
pd.DataFrame(pooled_rows).to_csv(os.path.join(RES, "rating_distribution_summary_pooled.csv"), index=False)

# ═══ per-movie figures ══════════════════════════════════════════════════════
summary_rows = []
for mv in MOVIES:
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(12, 6))

    for ax, measure in zip(axes, MEASURES):
        groups = ["human"] + MODELS
        data = []
        for g in groups:
            pooled = human_raw(mv, measure) if g == "human" else machine_raw(mv, g, measure)
            pooled = np.asarray(pooled, dtype=float)
            pooled = pooled[np.isfinite(pooled)]
            data.append(pooled)
            summary_rows.append(dict(movie=mv, measure=measure, source=("Human" if g == "human" else MODEL_LABEL[g]),
                                     n=len(pooled), mean=np.mean(pooled), std=np.std(pooled),
                                     min=np.min(pooled), max=np.max(pooled)))

        labels = ["Human"] + [MODEL_LABEL[m] for m in MODELS]
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for pc, g in zip(parts["bodies"], groups):
            pc.set_facecolor(COLORS[g]); pc.set_alpha(0.35)
        bp = ax.boxplot(data, widths=0.15, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black", lw=1.5))
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(COLORS[g]); patch.set_alpha(0.85)

        ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels(labels, fontsize=18)
        ax.axhline(0, color="#999999", lw=0.6, ls=":")
        ax.set_ylim(-11, 11)
        ax.set_title(measure.upper(), fontsize=18, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5)
        if measure == MEASURES[0]:
            ax.set_ylabel("Rating (-10 to +10, raw)", fontsize=18)

    fig.suptitle(f"{MOVIE_TITLE[mv]}: rating distribution, human vs. models",
                fontsize=20, fontweight="bold", fontfamily="Times New Roman")
    plt.tight_layout()
    out = os.path.join(RES, f"fig_rating_distribution_{mv}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out} (+.pdf)")

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(RES, "rating_distribution_summary.csv"), index=False)
pd.set_option("display.width", 200)
print(summary.round(2).to_string(index=False))
