"""
plot_solo_overlap_count_brains.py — Brain-space companion to
plot_solo_heatmap_v2.py's 6-panel (movie x measure) layout.

Instead of the 8-category human/model-combination palette used in
plot_solo_overlap_brains.py, here each ROI is colored by a single COUNT:
how many of the 4 series (Human, LLaDA-V, Qwen3-VL, Gemma 3) are
FDR-significant (sign-permutation, 5000 resamples) at that ROI, for that
(movie, measure) cell -- i.e. exactly the number of black dots that ROI's
column would carry in the heatmap. 0 = white (background), 1-4 = an
increasingly dark yellow ramp. Cortical Schaefer-100 ROIs only (the 14 AAL
subcortical ROIs, 101-114, are excluded, matching the request to not count
AAL here).

Layout: 3 rows (movies: TP, DM, PC) x 2 cols (measures: tom, valence),
same reading order as the heatmap. No panel titles. Single shared discrete
colorbar (0/1/2/3/4) at the bottom.

Usage:
    python plot_solo_overlap_count_brains.py
Output:
    human_vs_model_results/diagnostic_singledelay/fig_solo_overlap_count_brains.png
"""
import glob
import os
import re
import importlib.util
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from nilearn import plotting
from nilearn.image import resample_to_img

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
matplotlib.rcParams["font.size"] = 16

_HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(_HERE, "human_vs_model_results", "diagnostic_singledelay")
TAG = "singledelay"

MOVIES = ["tp", "dm", "pc"]
MODELS = ["llada", "qwen", "gemma"]
MEASURES = ["tom", "valence"]
MOVIE_LABEL = {"tp": "The Present", "dm": "Despicable Me", "pc": "Partly Cloudy"}

# 0 = white (background/no significance), 1-4 = increasingly dark yellow.
COUNT_COLORS = ["#FFFFFF", "#FFF9C4", "#FFEB3B", "#FBC02D", "#F57F17"]
HUMAN_MARKER_SIZE = 12   # black dot at the centroid of each human-significant ROI

# --- 1. Combined 114-ROI volumetric atlas (build once, cache to disk;
#     reused here for its cortical Schaefer-100 block only -- ROI 101-114
#     (AAL subcortical) are excluded from this figure per request). ---
COMBINED_ATLAS_PATH = os.path.join(_HERE, "atlas", "combined_roi114_atlas.nii.gz")
SCHAEFER_PATH = os.path.join(_HERE, "atlas", "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii")
AAL_PATH = os.path.join(_HERE, "atlas", "aal.nii")
AAL_INDICES = [37, 38, 39, 40, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78]

if not os.path.exists(COMBINED_ATLAS_PATH):
    print("Building combined 114-ROI atlas volume (one-time) ...")
    schaefer_img = nib.load(SCHAEFER_PATH)
    aal_resampled = resample_to_img(nib.load(AAL_PATH), schaefer_img,
                                    interpolation="nearest", force_resample=True, copy_header=True)
    aal_data = aal_resampled.get_fdata()
    combined = schaefer_img.get_fdata().copy()
    for i, code in enumerate(AAL_INDICES):
        combined[(aal_data == code) & (combined == 0)] = 101 + i
    nib.save(nib.Nifti1Image(combined, schaefer_img.affine, schaefer_img.header), COMBINED_ATLAS_PATH)

atlas_img = nib.load(COMBINED_ATLAS_PATH)
atlas_data = atlas_img.get_fdata()
CORTICAL_MASK = atlas_data <= 100  # exclude the 14 AAL subcortical ROIs (101-114)

# Schaefer-100 centroids (MNI RAS), used to mark human-significant ROIs with a
# black dot so human-model overlap can be told apart from model-only overlap.
CENTROID_PATH = os.path.join(_HERE, "atlas", "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv")
_cent = pd.read_csv(CENTROID_PATH).set_index("ROI Label")
ROI_CENTROID = {int(k): (r.R, r.A, r.S) for k, r in _cent.iterrows()}

# --- 2. Re-derive FDR-significance masks for human + each model, per
#     (movie, measure) -- same stats as analyze_solo_roi_overlap.py /
#     plot_solo_heatmap_v2.py, reusing their cached npz outputs. ---
spec = importlib.util.spec_from_file_location("encoding_pipeline_copy", os.path.join(_HERE, "encoding_pipeline copy.py"))
ep = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)

FNAME_RE = re.compile(
    r"hvm_(?P<movie>tp|dm|pc)_(?P<model>llada|qwen|gemma)_(?P<measure>tom|valence)"
    r"_sigma[\d.]+_chunk\d+_" + TAG + r"_pearson_r_unique_shared\.npz$")

human_sig_set = {}   # (movie, measure) -> set of 1-indexed ROIs
model_sig_set = {}   # (movie, model, measure) -> set of 1-indexed ROIs
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
    key_hm = (m["movie"], measure)
    if key_hm not in human_sig_set:
        human_sig_set[key_hm] = set((np.where(sig_mask[1])[0] + 1).tolist())
    model_sig_set[(m["movie"], m["model"], measure)] = set((np.where(sig_mask[2])[0] + 1).tolist())

# --- 3. Plot: 3 (movie) x 2 (measure) grid, matching plot_solo_heatmap_v2.py ---
CMAP = ListedColormap(COUNT_COLORS[1:])   # count 1-4 only; count 0 stays background (thresholded out)
# vmin=1, vmax=1+N maps count k (1..N) exactly to colors[k-1] (same trick as
# plot_solo_overlap_brains.py's category index mapping).

fig, axes = plt.subplots(3, 2, figsize=(13, 7.6))
fig.subplots_adjust(hspace=0.05, wspace=0.02, top=0.93, bottom=0.09, left=0.05, right=0.98)

# Capture each ax's grid position BEFORE plot_glass_brain touches it -- nilearn
# reshapes/repositions the passed-in ax to fit its multi-view display, so
# labels anchored via ax.transAxes AFTER that call drift out of alignment
# with the original row/col grid. Row/column labels below are placed with
# fig.text using these original (pre-nilearn) grid coordinates instead.
orig_pos = {(row, col): axes[row, col].get_position() for row in range(3) for col in range(2)}

for row, mv in enumerate(MOVIES):
    for col, measure in enumerate(MEASURES):
        ax = axes[row, col]
        human_rois = human_sig_set.get((mv, measure), set())
        counts = {}
        for roi in range(1, 101):  # cortical Schaefer ROIs only, exclude AAL
            c = int(roi in human_rois)
            for model in MODELS:
                c += int(roi in model_sig_set.get((mv, model, measure), set()))
            if c > 0:
                counts[roi] = c

        vol = np.zeros_like(atlas_data)
        for roi, c in counts.items():
            vol[(atlas_data == roi) & CORTICAL_MASK] = c
        stat_img = nib.Nifti1Image(vol, atlas_img.affine)

        display = plotting.plot_glass_brain(stat_img, display_mode="lzr", axes=ax,
                                            title=None, black_bg=False,
                                            cmap=CMAP, vmin=1, vmax=1 + len(COUNT_COLORS) - 1, threshold=0.5,
                                            plot_abs=False, colorbar=False, annotate=False,
                                            resampling_interpolation="nearest")
        human_cortical = sorted(r for r in human_rois if r <= 100)
        if human_cortical:
            display.add_markers([ROI_CENTROID[r] for r in human_cortical],
                                marker_color="black", marker_size=HUMAN_MARKER_SIZE)

for col, measure in enumerate(MEASURES):
    x_center = (orig_pos[(0, col)].x0 + orig_pos[(0, col)].x1) / 2
    fig.text(x_center, 0.935, measure.upper(), fontsize=22, fontweight="bold", ha="center", va="bottom")

for row, mv in enumerate(MOVIES):
    y_center = (orig_pos[(row, 0)].y0 + orig_pos[(row, 0)].y1) / 2
    fig.text(0.035, y_center, MOVIE_LABEL[mv], fontsize=22, fontweight="bold",
             rotation=90, ha="center", va="center")

sm = plt.cm.ScalarMappable(cmap=ListedColormap(COUNT_COLORS), norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5))
cax = fig.add_axes([0.32, 0.03, 0.36, 0.03])
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 1, 2, 3, 4])
cbar.ax.set_xticklabels(["0", "1", "2", "3", "4"], fontsize=18)
cbar.set_label("Number of FDR-significant rating sources (p<0.05)", fontsize=18)

fig.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color="black", markersize=8,
                               label="Human significant")],
           loc="lower left", bbox_to_anchor=(0.71, 0.025), frameon=False, fontsize=18)

fig_path = os.path.join(DIR, "fig_solo_overlap_count_brains_humanmarked.png")
fig.savefig(fig_path, dpi=600, bbox_inches="tight")
print(f"Saved -> {fig_path}")
