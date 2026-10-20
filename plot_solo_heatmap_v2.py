"""
plot_solo_heatmap_v2.py — Redesign of fig_solo_human_model_overlap.png
(raw mean solo Pearson r per ROI, single+LOO Pearson-r results):

  - x-axis = ROI, grouped Vis -> SomM -> DorsA -> SVA -> Limb -> Cont ->
    Default -> AAL (left to right), with a visual gap inserted between
    consecutive networks.
  - 2 rows x 3 cols of panels: top row = tom (TP, DM, PC), bottom row =
    valence (TP, DM, PC) -- reading order left-to-right, top-to-bottom
    puts all 3 tom panels first, all 3 valence panels after.
  - y-axis (4 rows per panel) = [Human, LLaDA-V, Qwen3-VL, Gemma 3], labels
    shown once on the RIGHT side only (not repeated on every panel).
  - Single shared colorbar at the bottom. No title.
  - Black dot = FDR-significant (sign-permutation, 5000 resamples).

Usage:
    python plot_solo_heatmap_v2.py
Output:
    human_vs_model_results/diagnostic_singledelay/fig_solo_human_model_overlap.png
"""
import glob
import os
import re
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
matplotlib.rcParams["font.size"] = 16

_HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(_HERE, "human_vs_model_results", "diagnostic_singledelay")
TAG = "singledelay"

spec = importlib.util.spec_from_file_location("encoding_pipeline_copy", os.path.join(_HERE, "encoding_pipeline copy.py"))
ep = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)

MOVIES = ["tp", "dm", "pc"]
MODELS = ["llada", "qwen", "gemma"]
MEASURES = ["tom", "valence"]
MODEL_LABEL = {"llada": "LLaDA-V", "qwen": "Qwen3-VL", "gemma": "Gemma 3"}
MOVIE_LABEL = {"tp": "The Present", "dm": "Despicable Me", "pc": "Partly Cloudy"}
ROW_LABELS = ["Human"] + [MODEL_LABEL[m] for m in MODELS]

FNAME_RE = re.compile(
    r"hvm_(?P<movie>tp|dm|pc)_(?P<model>llada|qwen|gemma)_(?P<measure>tom|valence)"
    r"_sigma[\d.]+_chunk\d+_" + TAG + r"_pearson_r_unique_shared\.npz$")

cache = {}
for f in sorted(glob.glob(os.path.join(DIR, f"hvm_*_{TAG}_pearson_r_unique_shared.npz"))):
    m = FNAME_RE.search(os.path.basename(f))
    if not m:
        continue
    d = np.load(f, allow_pickle=True)
    feature_order = list(d["feature_order"])
    measure = m["measure"]
    hi, mi = feature_order.index(f"human_{measure}"), feature_order.index(f"model_{measure}")
    results_r2 = d["results_r2"]; unique_r2s = d["unique_r2s"]
    solo_human = results_r2 - unique_r2s[:, mi, :]
    solo_model = results_r2 - unique_r2s[:, hi, :]
    stacked = np.stack([results_r2, solo_human, solo_model], axis=1)
    _, sig_mask = ep.run_statistics(stacked, ["full", "solo_human", "solo_model"], n_perm=5000, fdr_alpha=0.05)
    cache[(m["movie"], m["model"], measure)] = dict(
        solo_human=np.nanmean(solo_human, axis=0), solo_model=np.nanmean(solo_model, axis=0),
        sig_human=sig_mask[1], sig_model=sig_mask[2],
    )

# ── x-axis layout: ROI grouped by network, no gap columns -- a black
# divider line is drawn at each network boundary instead ──
NET_ORDER = ep.NET_ORDER
GAP = 0
roi_order, xpos, net_span = [], [], {}
x = 0
for net in NET_ORDER:
    idx = ep.ROI_BLOCKS_0[net]
    start = x
    for r in idx:
        roi_order.append(r); xpos.append(x); x += 1
    net_span[net] = (start, x - 1)
    x += GAP
N_X = x - GAP

# fixed color range (was data-driven; the actual data range made the bar
# asymmetric/hard to read, so pin it to a fixed, symmetric +-0.2)
all_vals = np.concatenate([v["solo_human"] for v in cache.values()] +
                          [v["solo_model"] for v in cache.values()])
data_min, data_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
VMIN, VMAX = -0.2, 0.2
print(f"Data range: [{data_min:.4f}, {data_max:.4f}]  ->  color range [{VMIN:.4f}, {VMAX:.4f}]")

fig, axes = plt.subplots(3, 2, figsize=(19, 7.5), sharex=True, sharey=True)
cmap = plt.cm.RdBu_r
norm = TwoSlopeNorm(vmin=VMIN, vcenter=0, vmax=VMAX)

for row, mv in enumerate(MOVIES):
    for col, measure in enumerate(MEASURES):
        ax = axes[row, col]
        mat = np.full((4, N_X), np.nan)
        sig = np.zeros((4, N_X), dtype=bool)
        human_key = next((mv, model, measure) for model in MODELS if (mv, model, measure) in cache)
        for xi, roi in zip(xpos, roi_order):
            mat[0, xi] = cache[human_key]["solo_human"][roi]
            sig[0, xi] = cache[human_key]["sig_human"][roi]
        for ri, model in enumerate(MODELS, start=1):
            key = (mv, model, measure)
            if key not in cache:
                continue
            for xi, roi in zip(xpos, roi_order):
                mat[ri, xi] = cache[key]["solo_model"][roi]
                sig[ri, xi] = cache[key]["sig_model"][roi]

        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="none",
                 extent=(-0.5, N_X - 0.5, 3.5, -0.5))
        ys, xs = np.where(sig)
        ax.scatter(xs, ys, s=5, c="black", marker="o", linewidths=0, zorder=3)

        for net, (s, e) in net_span.items():
            if s > 0:
                ax.axvline(s - 0.5, color="black", linewidth=1.0, zorder=2)
        ax.set_yticks([]); ax.set_xticks([])
        if row == 0:
            ax.set_title(measure.upper(), fontsize=24, fontweight="bold", pad=8)
        if col == 0:
            ax.set_ylabel(MOVIE_LABEL[mv], fontsize=18, fontweight="bold")
        if col == 1:
            ax.set_yticks(range(4))
            ax.set_yticklabels(ROW_LABELS, fontsize=18)
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", length=0)
        if row == 2:
            mids = [(net_span[net][0] + net_span[net][1]) / 2 for net in NET_ORDER]
            ax.set_xticks(mids); ax.set_xticklabels(NET_ORDER, fontsize=18, rotation=0)

fig.subplots_adjust(hspace=0.10, wspace=0.02, bottom=0.18, top=0.95, right=0.90)
cax = fig.add_axes([0.32, 0.055, 0.24, 0.035])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("Pearson r (mean across subjects)", fontsize=18)
cbar.set_ticks([VMIN, 0.0, VMAX])
cbar.ax.set_xticklabels([f"{VMIN:.2f}", "0.00", f"{VMAX:.2f}"])
cbar.ax.tick_params(labelsize=18)

fig.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
                           markeredgewidth=0, markersize=9,
                           label='FDR p<0.05')],
          loc="lower right", bbox_to_anchor=(0.89, 0.055), fontsize=18, frameon=False,
          handletextpad=0.4)

out_png = os.path.join(DIR, "fig_solo_human_model_overlap.png")
fig.savefig(out_png, dpi=600, bbox_inches="tight")
print(f"Saved -> {out_png}")
