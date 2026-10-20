"""
tom_valence_consistency_scatter.py — Within a model, does being good at ToM
also mean being good at Valence, or are the two measures decoupled?

Reuses results/bestlag_summary.csv (human-model pairs, base variant, best-lag
Pearson r -- same methodology as everywhere else). One point per
(model, movie): x = ToM r vs human, y = Valence r vs human.

Output (rating_benchmark/results/):
  fig_tom_valence_consistency.png
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "results")
best = pd.read_csv(os.path.join(RES, "bestlag_summary.csv"))

MOVIES = ["TP", "DM", "PC"]
HUMAN_PAIRS = ["Human–LLaDA-V", "Human–Qwen3-VL", "Human–Gemma 3"]
MODEL_OF = {"Human–LLaDA-V": "LLaDA-V", "Human–Qwen3-VL": "Qwen3-VL", "Human–Gemma 3": "Gemma 3"}
# Validated categorical palette (dataviz-skill slots 1-3): blue / orange / aqua.
COLORS = {"LLaDA-V": "#2a78d6", "Qwen3-VL": "#eb6834", "Gemma 3": "#1baf7a"}
MARKERS = {"TP": "o", "DM": "s", "PC": "^"}

hb = best[best.pair.isin(HUMAN_PAIRS)].copy()
hb["model"] = hb.pair.map(MODEL_OF)

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 22})
fig, ax = plt.subplots(figsize=(13.5, 11.5))

lims = [-0.2, 1.0]     # shared x/y range and ticks
all_tom_pts, all_val_pts = [], []
for model in COLORS:
    tom_pts, val_pts = [], []
    for mv in MOVIES:
        tom_r = hb[(hb.model == model) & (hb.movie == mv) & (hb.measure == "tom")]["pearson_r"]
        val_r = hb[(hb.model == model) & (hb.movie == mv) & (hb.measure == "valence")]["pearson_r"]
        if tom_r.empty or val_r.empty:
            continue
        tom_pts.append(tom_r.iloc[0]); val_pts.append(val_r.iloc[0])
        ax.scatter(tom_r.iloc[0], val_r.iloc[0], color=COLORS[model], marker=MARKERS[mv],
                   s=220, edgecolor="white", linewidth=0.8, zorder=3)
    all_tom_pts.extend(tom_pts); all_val_pts.extend(val_pts)
    # per-model mean (bigger, semi-transparent halo)
    tm = hb[(hb.model == model) & (hb.measure == "tom")]["pearson_r"].mean()
    vm = hb[(hb.model == model) & (hb.measure == "valence")]["pearson_r"].mean()
    ax.scatter(tm, vm, color=COLORS[model], marker="*", s=400, edgecolor="black",
              linewidth=1, zorder=4, label=f"{model} (mean)")

# Single least-squares fit over all 9 (model x movie) points combined.
all_tom_pts, all_val_pts = np.array(all_tom_pts), np.array(all_val_pts)
overall_fit = stats.linregress(all_tom_pts, all_val_pts)
xx = np.array(lims)
ax.plot(xx, overall_fit.intercept + overall_fit.slope * xx, color="black", lw=2, ls="-", zorder=2)

ax.plot(lims, lims, color="#999999", lw=1, ls="--", zorder=1, label="ToM r = Valence r")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xticks(ax.get_yticks())
ax.set_aspect("equal")   # equal tick spacing on x and y
ax.set_anchor("NW")   # anchor the (shrunk-to-square) axes to the top-left, so
                       # the freed space consistently lands on the right (for
                       # the legends, placed in FIGURE coordinates below, not
                       # axes-relative -- bbox_to_anchor on ax.transAxes
                       # combined with aspect="equal" is unreliable with
                       # bbox_inches="tight", which can silently clip legend
                       # text at some figsize/aspect combos) and there's no
                       # leftover gap above the axes either.
fig.subplots_adjust(left=0.10, right=0.60, top=0.99, bottom=0.10)
# Drop the x-axis's -0.2 label (keep the y-axis one) -- at the corner, equal
# aspect + shared lims puts both axes' "-0.2" tick right next to each other.
ax.set_xticklabels(["" if i == 0 else f"{t:g}" for i, t in enumerate(ax.get_xticks())])
ax.set_xlabel("Pearson r (M_ToM vs H_ToM)", fontsize=32, fontfamily="Times New Roman")
ax.set_ylabel("Pearson r (M_Valence vs H_Valence)", fontsize=32, fontfamily="Times New Roman")
ax.axhline(0, color="#cccccc", lw=0.6); ax.axvline(0, color="#cccccc", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(ls="--", lw=0.4, alpha=0.5)
ax.tick_params(axis="both", labelsize=28)

from matplotlib.lines import Line2D
model_handles = [Line2D([0], [0], marker="*", color="w", markerfacecolor=COLORS[m],
                        markeredgecolor="black", markersize=24, label=m) for m in COLORS]
movie_handles = [Line2D([0], [0], marker=MARKERS[mv], color="w", markerfacecolor="gray",
                        markersize=18, label=mv) for mv in MOVIES]
TNR = {"family": "Times New Roman", "size": 30}
# Both legends placed in the blank margin to the right of the (now square,
# equal-aspect) plot area, in FIGURE coordinates (bbox_transform=fig.transFigure)
# so their position is fixed and independent of how aspect="equal" happens to
# shrink/anchor the axes box -- avoids bbox_inches="tight" silently clipping
# legend text that was positioned via ax.transAxes.
leg1 = fig.legend(handles=model_handles, title="Model (mean)", alignment="left",
                  loc="upper left", bbox_to_anchor=(0.64, 0.98), bbox_transform=fig.transFigure,
                  frameon=False, prop=TNR, title_fontproperties=TNR)
leg2 = fig.legend(handles=movie_handles + [Line2D([0], [0], color="#999999", lw=1, ls="--", label="y = x")],
          title="Movie", alignment="left", loc="upper left", bbox_to_anchor=(0.64, 0.68), bbox_transform=fig.transFigure,
          frameon=False, prop=TNR, title_fontproperties=TNR)

fit_eq = f"y = {overall_fit.slope:.2f}x {'+' if overall_fit.intercept >= 0 else '-'} {abs(overall_fit.intercept):.2f}"
trend_handles = [
    Line2D([0], [0], color="black", lw=2, ls="-",
           label=f"{fit_eq}  ($R^2$={overall_fit.rvalue**2:.2f})"),
]
TNR_FIT = {"family": "Times New Roman", "size": 32}
fit_legend = ax.legend(handles=trend_handles, loc="upper right",
          bbox_to_anchor=(1.08, 0.98), bbox_transform=ax.transAxes,
          ncol=1, frameon=False, prop=TNR_FIT)
ax.add_artist(fit_legend)

out = os.path.join(RES, "fig_tom_valence_consistency.png")
fig.savefig(out, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")

for model in COLORS:
    tm = hb[(hb.model == model) & (hb.measure == "tom")]["pearson_r"].mean()
    vm = hb[(hb.model == model) & (hb.measure == "valence")]["pearson_r"].mean()
    print(f"{model:10s} mean ToM r={tm:.3f}  mean Valence r={vm:.3f}  diff(valence-tom)={vm-tm:+.3f}")
