"""
analyze_bestlag.py — Analyse the multi-metric rating benchmark at the best lag.

Operating point: for each human-model cell (movie x measure x pair) we pick the
lag that MAXIMISES Pearson r, then read every other metric at that same
alignment. Model-model pairs have no response delay to compensate and are
always taken at lag 0.

Reads  results/metrics_all_lags.csv
Writes results/
    bestlag_summary.csv            all metrics at the per-cell best (Pearson) lag
    fig_heatmap_metrics.png        metric heatmap, human pairs (agreement-oriented,
                                    naturally-bounded consistency metrics, per-row z-score)
    fig_heatmap_error_metrics.png  metric heatmap, unbounded absolute-error metrics
    fig_pearson_by_movie.png       human–model Pearson r per movie (per-model best lag, labeled)
    fig_pearson_by_movie_sharedlag.png  same, but one shared lag per movie/measure (all models)
    fig_pearson_by_movie_sharedlag_permodel.png  same, but one shared lag per movie/model
                                    (across ToM+Valence; different models can still differ)
    fig_model_comparison.png       LLaDA-V vs Qwen3-VL vs Gemma 3 across metrics (mean±SD)
    fig_relative_to_ceiling.png    human–model as % of mean inter-model ceiling
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "results")
df = pd.read_csv(os.path.join(RES, "metrics_all_lags.csv"))

MEASURES = ["tom", "valence"]
MOVIES = ["TP", "DM", "PC"]
MODELS = ["LLaDA-V", "Qwen3-VL", "Gemma 3"]
HUMAN_PAIRS = ["Human–LLaDA-V", "Human–Qwen3-VL", "Human–Gemma 3"]
MODEL_OF = {"Human–LLaDA-V": "LLaDA-V", "Human–Qwen3-VL": "Qwen3-VL", "Human–Gemma 3": "Gemma 3"}
# Validated categorical palette (dataviz-skill slots 1-3): blue / orange / aqua.
COLORS = {"LLaDA-V": "#2a78d6", "Qwen3-VL": "#eb6834", "Gemma 3": "#1baf7a"}

# metrics: (name, higher_is_better)
# Both natively -1..1 -> one heatmap, colour = true raw value, fixed -1..1 scale.
BOUNDED_METRICS = [("pearson_r", True), ("spearman_rho", True)]
# Unbounded absolute-error/scale metrics -> plotted separately, NOT color-comparable to the above.
ERROR_METRICS = [("rmse_z", False), ("mae_z", False), ("ba_loa_width", False)]
NICE = {"pearson_r": "Pearson r", "spearman_rho": "Spearman ρ",
        "mutual_info": "Mutual info", "sign_agree": "Sign agree",
        "kappa_tertile": "κ (tertile)", "rmse_z": "RMSE (z)",
        "mae_z": "MAE (z)", "ba_loa_width": "BA LoA width"}

# ── best-Pearson lag per cell ───────────────────────────────────────────────
# The lag search compensates for HUMAN response delay, so it applies to
# human-model pairs only. Model-model pairs rate the same frames with no
# response delay and are always taken at lag 0.
is_hm = df.pair.isin(HUMAN_PAIRS)
best_hm = df[is_hm].loc[lambda d: d.groupby(["movie", "measure", "pair"])["pearson_r"].idxmax()]
best_mm = df[~is_hm & (df.lag_sec == 0)]
best = pd.concat([best_hm, best_mm]).sort_index().reset_index(drop=True)
best.to_csv(os.path.join(RES, "bestlag_summary.csv"), index=False)

hb = best[best.pair.isin(HUMAN_PAIRS)].copy()
hb["model"] = hb.pair.map(MODEL_OF)
hb["cell"] = hb.movie + "\n" + hb.measure.str.upper()

# ═══ FIG 1: heatmap of metrics (agreement-oriented) ══════════════════════════
cols = []
labels = []
for mv in MOVIES:
    for me in MEASURES:
        for pr in HUMAN_PAIRS:
            row = hb[(hb.movie == mv) & (hb.measure == me) & (hb.pair == pr)]
            if row.empty:
                continue
            cols.append(row.iloc[0])
            labels.append(f"{mv}·{me[:3]}\n{MODEL_OF[pr]}")
M = pd.DataFrame(cols)

def make_fixed_range_heatmap(metrics, out_name, title, vmin=-1, vmax=1, cmap="RdBu_r"):
    """Colour = the raw metric value directly against a fixed -1..1 scale (no rescaling)."""
    mat = np.array([M[met].to_numpy(float) for met, _ in metrics])
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(M)), 1.2 * len(metrics) + 2))
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(M))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([NICE[m] for m, _ in metrics], fontsize=10)
    for i in range(len(metrics)):
        for j in range(len(M)):
            v = mat[i, j]
            frac = (v - vmin) / (vmax - vmin)
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(frac - 0.5) > 0.32 else "black", fontsize=7)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label(f"true value ({vmin:g} to {vmax:g})", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RES, out_name), dpi=200, bbox_inches="tight")
    plt.close(fig)

def make_heatmap(metrics, out_name, title, cbar_label):
    mat = np.zeros((len(metrics), len(M)))
    raw = np.zeros_like(mat)
    for i, (met, hib) in enumerate(metrics):
        v = M[met].to_numpy(float)
        raw[i] = v
        lo, hi = np.nanmin(v), np.nanmax(v)
        n = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
        mat[i] = n if hib else 1 - n           # flip so bright = more agreement

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(M)), 1.2 * len(metrics) + 2))
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(M))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([NICE[m] for m, _ in metrics], fontsize=10)
    for i in range(len(metrics)):
        for j in range(len(M)):
            ax.text(j, i, f"{raw[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.55 else "black", fontsize=7)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label(cbar_label, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RES, out_name), dpi=200, bbox_inches="tight")
    plt.close(fig)

make_fixed_range_heatmap(BOUNDED_METRICS, "fig_heatmap_metrics.png",
             "Rating-comparison metrics at best (Pearson) lag\n"
             "Pearson r / Spearman ρ on their true -1..1 scale (no rescaling)")
make_heatmap(ERROR_METRICS, "fig_heatmap_error_metrics.png",
             "Rating-comparison metrics at best (Pearson) lag\n"
             "unbounded absolute-error/scale metrics (NOT comparable to the metrics above);\n"
             "colour = per-metric min–max, low-is-better flipped so bright = smaller error",
             "← larger error   smaller error →")

n_models = len(MODELS)

# ═══ FIG 2: Pearson r by movie (per measure) — each bar at ITS OWN best lag ═
# Each model independently picks the lag that maximises its own Pearson r
# (see module docstring) -- so bars in the same movie are NOT necessarily at
# the same lag. Label each bar with the lag it actually used.
# ToM in the left panel, Valence in the right panel (MEASURES is already
# ordered ["tom", "valence"], so this is just zip(axes, MEASURES) below).
FONT = {"font.family": "Times New Roman", "font.size": 15}
with plt.rc_context(FONT):
    w2 = 0.8 / n_models
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(13, 5), sharey=True)
    for ax, me in zip(axes, MEASURES):
        sub = hb[hb.measure == me]
        x = np.arange(len(MOVIES))
        for k, model in enumerate(MODELS):
            s = sub[sub.model == model].set_index("movie").reindex(MOVIES)
            offset = (k - (n_models - 1) / 2) * w2
            bars = ax.bar(x + offset, s.pearson_r, w2, color=COLORS[model], label=model)
            for b, v, lag in zip(bars, s.pearson_r, s.lag_sec):
                if pd.notna(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.05),
                            f"{lag:g}s", ha="center", fontsize=11, color="#333")
        ax.set_title(me.upper(), fontsize=17); ax.set_xticks(x); ax.set_xticklabels(MOVIES)
        ax.axhline(0, color="k", lw=0.5); ax.grid(axis="y", ls="--", alpha=0.5)
        ax.set_ylabel("Pearson r", fontsize=15) if me == MEASURES[0] else None
    axes[-1].legend(fontsize=13, loc="upper right", prop={"family": "Times New Roman", "size": 13})
    fig.suptitle("Human-model Pearson r by movie (label = each bar's own best lag)",
                fontsize=18, fontfamily="Times New Roman")
    plt.tight_layout()
    fig.savefig(os.path.join(RES, "fig_pearson_by_movie.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)

# ═══ FIG 2b: Pearson r by movie, ONE SHARED lag per movie/measure ══════════
# Instead of letting each model pick its own best lag, pick the single lag
# (from the same grid) that maximises the MEAN Pearson r across all 3 models
# for that movie/measure, then read every model's r at that one shared lag.
# This is a fair head-to-head: everyone is compared at the same alignment.
shared_rows = []
for mv in MOVIES:
    for me in MEASURES:
        sub_all = df[(df.movie == mv) & (df.measure == me) & (df.pair.isin(HUMAN_PAIRS))]
        mean_by_lag = sub_all.groupby("lag_sec")["pearson_r"].mean()
        shared_lag = mean_by_lag.idxmax()
        for pr in HUMAN_PAIRS:
            row = sub_all[(sub_all.pair == pr) & (sub_all.lag_sec == shared_lag)]
            r = float(row.pearson_r.iloc[0]) if not row.empty else np.nan
            shared_rows.append(dict(movie=mv, measure=me, model=MODEL_OF[pr],
                                    shared_lag_sec=shared_lag, pearson_r=r))
shared_df = pd.DataFrame(shared_rows)
shared_df.to_csv(os.path.join(RES, "pearson_by_movie_sharedlag.csv"), index=False)

fig, axes = plt.subplots(1, len(MEASURES), figsize=(12, 4.5), sharey=True)
for ax, me in zip(axes, MEASURES):
    sub = shared_df[shared_df.measure == me]
    x = np.arange(len(MOVIES))
    piv = sub.pivot(index="movie", columns="model", values="pearson_r").reindex(MOVIES)
    lag_per_movie = sub.drop_duplicates("movie").set_index("movie")["shared_lag_sec"].reindex(MOVIES)
    for k, model in enumerate(MODELS):
        s = sub[sub.model == model].set_index("movie").reindex(MOVIES)
        offset = (k - (n_models - 1) / 2) * w2
        ax.bar(x + offset, s.pearson_r, w2, color=COLORS[model], label=model)
    for xi, mv in enumerate(MOVIES):
        ymax = piv.loc[mv].max()
        ax.text(xi, ymax + 0.03, f"lag={lag_per_movie[mv]:g}s", ha="center",
                fontsize=8, style="italic", color="#444")
    ax.set_title(me.upper()); ax.set_xticks(x); ax.set_xticklabels(MOVIES)
    ax.axhline(0, color="k", lw=0.5); ax.grid(axis="y", ls="--", alpha=0.5)
    ax.set_ylabel("Pearson r") if me == MEASURES[0] else None
axes[-1].legend(fontsize=8, loc="upper right")
fig.suptitle("Human–model Pearson r by movie (ONE shared lag per movie/measure)", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(RES, "fig_pearson_by_movie_sharedlag.png"), dpi=200,
            bbox_inches="tight")
plt.close(fig)

# ═══ FIG 2c: Pearson r by movie, lag shared across ToM+Valence PER MODEL ═══
# Within a movie, each model's ToM bar and Valence bar use the SAME lag (so
# that model's two measures are directly comparable to each other) -- but
# different MODELS can still use different lags. Lag = the one that maximises
# the MEAN Pearson r across {tom, valence} for that (movie, model).
shared2_rows = []
for mv in MOVIES:
    for pr in HUMAN_PAIRS:
        sub_all = df[(df.movie == mv) & (df.pair == pr) & (df.measure.isin(MEASURES))]
        mean_by_lag = sub_all.groupby("lag_sec")["pearson_r"].mean()
        shared_lag = mean_by_lag.idxmax()
        for me in MEASURES:
            row = sub_all[(sub_all.measure == me) & (sub_all.lag_sec == shared_lag)]
            r = float(row.pearson_r.iloc[0]) if not row.empty else np.nan
            shared2_rows.append(dict(movie=mv, measure=me, model=MODEL_OF[pr],
                                     shared_lag_sec=shared_lag, pearson_r=r))
shared2_df = pd.DataFrame(shared2_rows)
shared2_df.to_csv(os.path.join(RES, "pearson_by_movie_sharedlag_permodel.csv"), index=False)

fig, axes = plt.subplots(1, len(MEASURES), figsize=(12, 4.5), sharey=True)
for ax, me in zip(axes, MEASURES):
    sub = shared2_df[shared2_df.measure == me]
    x = np.arange(len(MOVIES))
    for k, model in enumerate(MODELS):
        s = sub[sub.model == model].set_index("movie").reindex(MOVIES)
        lag_s = (shared2_df[shared2_df.model == model].drop_duplicates("movie")
                 .set_index("movie")["shared_lag_sec"].reindex(MOVIES))
        offset = (k - (n_models - 1) / 2) * w2
        bars = ax.bar(x + offset, s.pearson_r, w2, color=COLORS[model], label=model)
        for b, v, lag in zip(bars, s.pearson_r, lag_s):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.05),
                        f"{lag:g}s", ha="center", fontsize=6.5, color="#333")
    ax.set_title(me.upper()); ax.set_xticks(x); ax.set_xticklabels(MOVIES)
    ax.axhline(0, color="k", lw=0.5); ax.grid(axis="y", ls="--", alpha=0.5)
    ax.set_ylabel("Pearson r") if me == MEASURES[0] else None
axes[-1].legend(fontsize=8, loc="upper right")
fig.suptitle("Human–model Pearson r by movie (lag shared across ToM+Valence, per model)", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(RES, "fig_pearson_by_movie_sharedlag_permodel.png"), dpi=200,
            bbox_inches="tight")
plt.close(fig)

# ═══ FIG 3: model comparison across metrics (mean ± SD over movies×measures) ═
key = ["pearson_r", "spearman_rho", "sign_agree", "kappa_tertile"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(key)); bw3 = 0.8 / n_models
for k, model in enumerate(MODELS):
    s = hb[hb.model == model]
    means = [s[m].mean() for m in key]
    sds = [s[m].std() for m in key]
    offset = (k - (n_models - 1) / 2) * bw3
    ax.bar(x + offset, means, bw3, yerr=sds, capsize=3,
           color=COLORS[model], label=model,
           error_kw=dict(elinewidth=1, ecolor="#444"))
ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in key], fontsize=10)
ax.axhline(0, color="k", lw=0.5); ax.grid(axis="y", ls="--", alpha=0.5)
ax.set_ylabel("mean over movies × measures"); ax.legend()
ax.set_title(" vs ".join(MODELS) + " — human alignment across metrics")
plt.tight_layout()
fig.savefig(os.path.join(RES, "fig_model_comparison.png"), dpi=200,
            bbox_inches="tight")
plt.close(fig)

# ═══ FIG 4: human–model as % of inter-model ceiling ═════════════════════════
# With >2 machine models there's no single inter-model pair to use as THE ceiling,
# so each model's ceiling = mean pairwise Pearson r (at lag 0) with the OTHER
# machine models -- i.e. "how well does this model agree with its peers".
import itertools as _it
mm_pair_df = {}
for a, b in _it.combinations(MODELS, 2):
    pair_key = f"{a}–{b}"
    mm_pair_df[(a, b)] = (df[(df.pair == pair_key) & (df.lag_sec == 0)]
                          .set_index(["movie", "measure"]))

def ceiling_for(model, mv, me):
    vals = []
    for (a, b), sub_df in mm_pair_df.items():
        if model not in (a, b):
            continue
        if (mv, me) in sub_df.index:
            vals.append(sub_df.loc[(mv, me), "pearson_r"])
    return float(np.mean(vals)) if vals else np.nan

fig, axes = plt.subplots(1, len(MEASURES), figsize=(12, 4.5), sharey=True)
for ax, me in zip(axes, MEASURES):
    x = np.arange(len(MOVIES)); bw4 = 0.8 / n_models
    for k, model in enumerate(MODELS):
        vals = []
        for mv in MOVIES:
            hm = hb[(hb.movie == mv) & (hb.measure == me) & (hb.model == model)]
            ceil = ceiling_for(model, mv, me)
            vals.append(100 * hm.pearson_r.iloc[0] / ceil
                        if (not hm.empty and ceil and ceil > 0) else np.nan)
        offset = (k - (n_models - 1) / 2) * bw4
        ax.bar(x + offset, vals, bw4, color=COLORS[model], label=model)
    ax.axhline(100, color="k", ls="--", lw=0.8)
    ax.set_title(me.upper()); ax.set_xticks(x); ax.set_xticklabels(MOVIES)
    ax.grid(axis="y", ls="--", alpha=0.5)
    if me == MEASURES[0]:
        ax.set_ylabel("human–model Pearson\n(% of mean inter-model ceiling)")
axes[-1].legend(fontsize=9)
fig.suptitle("How close machine–human gets to the machine–machine ceiling\n"
             "(ceiling = mean pairwise Pearson r with the other machine models)", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(RES, "fig_relative_to_ceiling.png"), dpi=200,
            bbox_inches="tight")
plt.close(fig)

# ── console summary ─────────────────────────────────────────────────────────
pd.set_option("display.width", 200)
print("=== best-(Pearson)-lag operating point, human pairs ===")
print(hb[["movie", "measure", "model", "lag_sec", "pearson_r",
          "spearman_rho", "sign_agree", "kappa_tertile", "rmse_z",
          "ba_loa_width"]].round(3).to_string(index=False))
print("\nSaved 7 figures + bestlag_summary.csv ->", RES)
