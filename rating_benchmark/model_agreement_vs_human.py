"""
model_agreement_vs_human.py — Are the 3 models more similar to EACH OTHER
than they are to human ratings?

Reuses results/bestlag_summary.csv (already produced by analyze_bestlag.py --
the 3 human-model pairs each at their own best-Pearson lag, the 3 model-model
pairs at lag 0 since models have no response delay). Base variant only.

Outputs (rating_benchmark/results/):
  fig_model_agreement_vs_human.png   6-bar-per-panel: 3 movies x 2 measures,
                                      human-model pairs vs model-model pairs
  fig_agreement_summary.png          headline: mean r across all cells,
                                      Human-Model vs Model-Model, with individual
                                      cells overlaid as points
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "results")
best = pd.read_csv(os.path.join(RES, "bestlag_summary.csv"))

MOVIES = ["TP", "DM", "PC"]
MEASURES = ["tom", "valence"]
HUMAN_PAIRS = ["Human–LLaDA-V", "Human–Qwen3-VL", "Human–Gemma 3"]
MODEL_PAIRS = ["LLaDA-V–Qwen3-VL", "LLaDA-V–Gemma 3", "Qwen3-VL–Gemma 3"]
ALL_PAIRS = HUMAN_PAIRS + MODEL_PAIRS
PAIR_SHORT = {"Human–LLaDA-V": "H-L", "Human–Qwen3-VL": "H-Q", "Human–Gemma 3": "H-G",
              "LLaDA-V–Qwen3-VL": "L-Q", "LLaDA-V–Gemma 3": "L-G", "Qwen3-VL–Gemma 3": "Q-G"}
# Validated categorical palette (dataviz-skill slots 1-2): blue / orange.
GROUP_COLOR = {"human": "#2a78d6", "model": "#eb6834"}
FONT = {"font.family": "Times New Roman", "font.size": 15}

with plt.rc_context(FONT):
    # ═══ FIG 1: per movie/measure, all 6 pairwise r's side by side ═════════
    fig, axes = plt.subplots(len(MEASURES), len(MOVIES), figsize=(16, 7.5), squeeze=False)

    for ri, measure in enumerate(MEASURES):
        for ci, mv in enumerate(MOVIES):
            ax = axes[ri][ci]
            sub = best[(best.movie == mv) & (best.measure == measure)].set_index("pair").reindex(ALL_PAIRS)
            vals = sub["pearson_r"].to_numpy()
            colors = [GROUP_COLOR["human"] if p in HUMAN_PAIRS else GROUP_COLOR["model"] for p in ALL_PAIRS]
            x = np.arange(len(ALL_PAIRS))
            bars = ax.bar(x, vals, color=colors)
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.text(xi, v + (0.015 if v >= 0 else -0.03), f"{v:.2f}", ha="center", fontsize=11)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(2.5, color="#999999", lw=0.8, ls=":")
            ax.set_xticks(x); ax.set_xticklabels([PAIR_SHORT[p] for p in ALL_PAIRS], fontsize=12)
            ax.set_title(f"{mv} - {measure.upper()}", fontsize=15)
            ax.grid(axis="y", ls="--", lw=0.4, alpha=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if ci == 0:
                ax.set_ylabel("Pearson r", fontsize=15)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=GROUP_COLOR["human"], label="Human - Model"),
                         Patch(color=GROUP_COLOR["model"], label="Model - Model")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=15,
               prop={"family": "Times New Roman", "size": 15})
    fig.suptitle("Human-model and model-model agreement, by movie and measure (base variant)",
                fontsize=18, y=1.1, fontfamily="Times New Roman", fontweight="bold")
    plt.tight_layout()
    out1 = os.path.join(RES, "fig_model_agreement_vs_human.png")
    fig.savefig(out1, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out1}")

    # ═══ FIG 2: headline summary -- mean r, Human-Model vs Model-Model ═════
    human_vals = best[best.pair.isin(HUMAN_PAIRS)]["pearson_r"].to_numpy()
    model_vals = best[best.pair.isin(MODEL_PAIRS)]["pearson_r"].to_numpy()

    XPOS = [0, 0.9]  # wide enough apart that the bigger xtick labels don't collide
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    for i, (label, vals, color) in zip(XPOS, [("Human-Model\n(3 pairs x 6 cells)", human_vals, GROUP_COLOR["human"]),
                                                ("Model-Model\n(3 pairs x 6 cells)", model_vals, GROUP_COLOR["model"])]):
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.09
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, alpha=0.55, s=60, zorder=2)
        m, sd = np.mean(vals), np.std(vals)
        ax.errorbar(i, m, yerr=sd, fmt="o", color="black", ms=12, capsize=7, lw=2.5, zorder=3)
        ax.text(i, m + sd + 0.03, f"mean={m:.3f}", ha="center", fontsize=24, fontweight="bold")

    ax.set_xticks(XPOS); ax.set_xticklabels(["Human-Model", "Model-Model"], fontsize=26)
    ax.set_xlim(XPOS[0] - 0.35, XPOS[1] + 0.35)
    ax.set_ylabel("Pearson r", fontsize=26)
    ax.tick_params(axis="both", labelsize=24)
    ax.axhline(0, color="#999999", lw=0.6, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5)
    plt.tight_layout()
    out2 = os.path.join(RES, "fig_agreement_summary.png")
    fig.savefig(out2, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out2}")

print(f"\nMean Human-Model r = {np.mean(human_vals):.3f} (sd={np.std(human_vals):.3f}, n={len(human_vals)})")
print(f"Mean Model-Model r = {np.mean(model_vals):.3f} (sd={np.std(model_vals):.3f}, n={len(model_vals)})")
