"""
analyze_solo_roi_overlap.py — Using the single+LOO Pearson-r results
(human_vs_model_results/diagnostic_singledelay/), for each of the 18
(movie, model, measure) combos:
  1. List the FDR-significant ROIs for solo_model (index, network, mean r).
  2. For each of the 6 (movie, measure) pairs, compare the FDR-significant
     ROI set of solo_human against each of the 3 models' solo_model
     significant ROI sets -- how many ROIs overlap ("shared"), how many are
     model-only, how many are human-only.

solo_human is identical across the 3 models for a given (movie, measure)
(the leave-one-out "human-only" refit never touches the model feature), so
it is computed once per (movie, measure) and reused for the 3 comparisons.

Usage:
    python analyze_solo_roi_overlap.py
Outputs (human_vs_model_results/diagnostic_singledelay/):
    solo_model_significant_rois.csv       one row per (movie, model, measure, roi)
    solo_human_model_overlap.csv          one row per (movie, measure, model) with overlap counts
    fig_solo_human_model_overlap.png      heatmap of significance per ROI x [Human, LLaDA, Qwen, Gemma], one panel per movie/measure
"""
import glob
import os
import re
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("encoding_pipeline_copy",
                                               os.path.join(_HERE, "encoding_pipeline copy.py"))
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)

DIR = os.path.join(_HERE, "human_vs_model_results", "diagnostic_singledelay")
TAG = "singledelay"
MOVIES = ["tp", "dm", "pc"]
MODELS = ["llada", "qwen", "gemma"]
MEASURES = ["tom", "valence"]
MODEL_LABEL = {"llada": "LLaDA-V", "qwen": "Qwen3-VL", "gemma": "Gemma 3"}

FNAME_RE = re.compile(
    r"hvm_(?P<movie>tp|dm|pc)_(?P<model>llada|qwen|gemma)_(?P<measure>tom|valence)"
    r"_sigma(?P<sigma>[\d.]+)_chunk(?P<chunk>\d+)_" + TAG + r"_pearson_r_unique_shared\.npz$"
)

roi_to_net = {}
for net, idx in ep.ROI_BLOCKS_0.items():
    for r in idx:
        roi_to_net[r] = net

# --- load everything, compute solo_human / solo_model + FDR sig masks ---
cache = {}   # (movie, model, measure) -> dict(full, solo_human, solo_model, sig_full, sig_human, sig_model)
for f in sorted(glob.glob(os.path.join(DIR, f"hvm_*_{TAG}_pearson_r_unique_shared.npz"))):
    m = FNAME_RE.search(os.path.basename(f))
    if not m:
        continue
    d = np.load(f, allow_pickle=True)
    feature_order = list(d["feature_order"])
    measure = m["measure"]
    hi, mi = feature_order.index(f"human_{measure}"), feature_order.index(f"model_{measure}")

    results_r2 = d["results_r2"]
    unique_r2s = d["unique_r2s"]
    solo_human = results_r2 - unique_r2s[:, mi, :]
    solo_model = results_r2 - unique_r2s[:, hi, :]

    stacked = np.stack([results_r2, solo_human, solo_model], axis=1)
    _, sig_mask = ep.run_statistics(stacked, ["full", "solo_human", "solo_model"], n_perm=5000, fdr_alpha=0.05)

    cache[(m["movie"], m["model"], measure)] = dict(
        full=results_r2, solo_human=solo_human, solo_model=solo_model,
        sig_full=sig_mask[0], sig_human=sig_mask[1], sig_model=sig_mask[2],
    )

# ═══ (1) significant ROIs for solo_model, per combo ═══════════════════════
rows_sig = []
for mv in MOVIES:
    for model in MODELS:
        for measure in MEASURES:
            key = (mv, model, measure)
            if key not in cache:
                print(f"[skip] missing {key}")
                continue
            c = cache[key]
            mean_r = np.nanmean(c["solo_model"], axis=0)
            sig_idx = np.where(c["sig_model"])[0]
            for ri in sig_idx:
                rows_sig.append(dict(movie=mv, model=model, measure=measure,
                                     roi=int(ri) + 1, network=roi_to_net[ri],
                                     mean_r=float(mean_r[ri])))

df_sig = pd.DataFrame(rows_sig)
out1 = os.path.join(DIR, "solo_model_significant_rois.csv")
df_sig.to_csv(out1, index=False)
print(f"Saved -> {out1}  ({len(df_sig)} sig ROI rows)")

pd.set_option("display.width", 200)
print("\n=== solo_model significant ROIs per combo (index=network=mean_r) ===")
for mv in MOVIES:
    for model in MODELS:
        for measure in MEASURES:
            sub = df_sig[(df_sig.movie == mv) & (df_sig.model == model) & (df_sig.measure == measure)]
            sub = sub.sort_values("roi")
            if len(sub) == 0:
                print(f"{mv} {model:6s} {measure:8s}: (none)")
                continue
            items = ", ".join(f"ROI{r.roi}({r.network})={r.mean_r:.3f}" for r in sub.itertuples())
            print(f"{mv} {model:6s} {measure:8s} [{len(sub)}]: {items}")

# ═══ (2) human vs model overlap, per (movie, measure) ══════════════════════
rows_overlap = []
for mv in MOVIES:
    for measure in MEASURES:
        # solo_human is identical across models for this (movie, measure) --
        # pull it from whichever model happens to be present.
        human_key = next((mv, model, measure) for model in MODELS if (mv, model, measure) in cache)
        human_sig = set(np.where(cache[human_key]["sig_human"])[0].tolist())
        for model in MODELS:
            key = (mv, model, measure)
            if key not in cache:
                continue
            model_sig = set(np.where(cache[key]["sig_model"])[0].tolist())
            shared = human_sig & model_sig
            rows_overlap.append(dict(
                movie=mv, measure=measure, model=model,
                n_human_sig=len(human_sig), n_model_sig=len(model_sig),
                n_shared=len(shared), n_human_only=len(human_sig - model_sig),
                n_model_only=len(model_sig - human_sig),
                shared_rois=",".join(str(r + 1) for r in sorted(shared)) if shared else "-",
            ))

df_overlap = pd.DataFrame(rows_overlap)
out2 = os.path.join(DIR, "solo_human_model_overlap.csv")
df_overlap.to_csv(out2, index=False)
print(f"\nSaved -> {out2}")
print("\n=== Human vs Model solo significant-ROI overlap ===")
print(df_overlap.to_string(index=False))

# ═══ (3) heatmap: ROI (grouped by network) x [Human, LLaDA, Qwen, Gemma], one panel per movie/measure ═══
NET_ORDER = ep.NET_ORDER
roi_order = [r for net in NET_ORDER for r in ep.ROI_BLOCKS_0[net]]  # 0-based, grouped by network
net_boundaries = np.cumsum([len(ep.ROI_BLOCKS_0[net]) for net in NET_ORDER])

fig, axes = plt.subplots(1, 6, figsize=(16, 12), sharey=True)
panel_idx = 0
cmap = plt.cm.RdBu_r
norm = TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)

for mv in MOVIES:
    for measure in MEASURES:
        ax = axes[panel_idx]; panel_idx += 1
        human_key = next((mv, model, measure) for model in MODELS if (mv, model, measure) in cache)
        cols_val = [np.nanmean(cache[human_key]["solo_human"], axis=0)]
        cols_sig = [cache[human_key]["sig_human"]]
        col_labels = ["Human"]
        for model in MODELS:
            key = (mv, model, measure)
            if key not in cache:
                cols_val.append(np.full(114, np.nan)); cols_sig.append(np.zeros(114, dtype=bool))
            else:
                cols_val.append(np.nanmean(cache[key]["solo_model"], axis=0))
                cols_sig.append(cache[key]["sig_model"])
            col_labels.append(MODEL_LABEL[model])

        mat = np.stack(cols_val, axis=1)[roi_order]          # [114, 4]
        sig = np.stack(cols_sig, axis=1)[roi_order]           # [114, 4]

        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="none")
        # mark significant cells with a black dot
        ys, xs = np.where(sig)
        ax.scatter(xs, ys, s=6, c="black", marker="o", linewidths=0)

        for b in net_boundaries[:-1]:
            ax.axhline(b - 0.5, color="white", linewidth=1.2)
        ax.set_xticks(range(4)); ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
        ax.set_title(f"{mv.upper()}\n{measure}", fontsize=9)
        if panel_idx == 1:
            net_mid = [0] + net_boundaries.tolist()
            ticks = [(net_mid[i] + net_mid[i+1]) / 2 - 0.5 for i in range(len(NET_ORDER))]
            ax.set_yticks(ticks); ax.set_yticklabels(NET_ORDER, fontsize=8)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
cbar.set_label("solo Pearson r (mean across subjects)")
fig.suptitle("Human-solo vs Model-solo prediction per ROI (dot = FDR-significant, single+LOO design)", fontsize=11)
fig_path = os.path.join(DIR, "fig_solo_human_model_overlap.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {fig_path}")
