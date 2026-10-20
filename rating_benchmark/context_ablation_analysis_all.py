"""
context_ablation_analysis_all.py — Context-injection ablation across ALL
THREE models (Gemma 3, Qwen3-VL, LLaDA-V). Now that all three have base +
the same 4 window-length variants (ctx_4s/ctx_8s/ctx_12s/ctx_16s, see
rating_generation/context_variants.py) for BOTH ToM and Valence on all 3
movies, this replaces the earlier split (context_ablation_analysis.py for
Gemma only, context_ablation_analysis_llada_qwen.py for LLaDA/Qwen ToM only)
with one combined comparison.

LAG METHODOLOGY (same convention used everywhere else in this project --
rating_metrics.py / analyze_bestlag.py / context_ablation_analysis*.py):
for each (movie, measure, model, variant) cell INDEPENDENTLY, sweep the
TR-aligned lag grid below and pick the lag that MAXIMISES Pearson r, then
report the metric AT that lag. This is not circular -- the lag is chosen
purely to find each cell's best possible alignment, never using the number
you then report as the "finding" -- and every cell (every model, every
variant, base included) gets the IDENTICAL lag-search budget, so comparing
variants or models against each other on this basis is fair. We deliberately
do NOT force a single shared lag across variants here: if injecting context
changes a model's effective response latency, forcing one lag on all
variants would conflate "does context change timing" with "does context
improve alignment", which is not the question this ablation asks.
  TP / DM : TR = 0.8 s (fMRI alignment grid), lags 0 .. 6.4 s (0..8 TR)
  PC      : TR = 2.0 s (fMRI alignment grid), lags 0 .. 6.0 s (0..3 TR)
Preprocessing: human = 20%-low-corr-removed mean, linear interp to TR grid, no
smoothing. Machine = per-frame CSV -> Gaussian smooth (sigma=2s) -> TR grid.

Outputs (rating_benchmark/results/):
  context_ablation_bestlag_all.csv   best-lag Pearson r per movie/measure/model/variant
  fig_context_ablation_all.png       2 (measure) x 3 (movie) grid; each panel:
                                      variant on x-axis, grouped bars per model
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "rating_generation"))
from rating_utils import interp_to_fmri_grid, sample2tr, smooth_series
from context_variants import VARIANTS as CONTEXT_VARIANTS

DT = 0.25
TR = {"TP": 0.8, "DM": 0.8, "PC": 2.0}
FMRI_T = {"TP": 239, "DM": 739, "PC": 167}
PC_DROP_TR = 5
GAUSS_SIGMA_SEC = 2.0
MOVIES = ["TP", "DM", "PC"]
MOVIE_TITLE = {"TP": "The Present", "DM": "Despicable Me", "PC": "Partly Cloudy"}
MEASURES = ["tom", "valence"]
RATING_COL = {"tom": "tom_rating", "valence": "valence_rating"}

MODELS = ["llada", "qwen", "gemma"]
MODEL_LABEL = {"llada": "LLaDA-V", "qwen": "Qwen3-VL", "gemma": "Gemma 3"}
MODEL_DIR_SUFFIX = {"llada": "final", "qwen": "qwen", "gemma": "gemma"}
MODEL_FILE_PREFIX = {"llada": "", "qwen": "qwen_", "gemma": "gemma_"}
# Validated categorical palette (dataviz-skill slots 1-3): blue / orange / aqua.
MODEL_COLOR = {"llada": "#2a78d6", "qwen": "#eb6834", "gemma": "#1baf7a"}

VARIANTS = ["base"] + CONTEXT_VARIANTS
VARIANT_SUFFIX = {"base": ""}
VARIANT_SUFFIX.update({v: f"_{v}" for v in CONTEXT_VARIANTS})

LAG_SEC = {"TP": np.arange(0.0, 6.4 + 1e-6, 0.8),
           "DM": np.arange(0.0, 6.4 + 1e-6, 0.8),
           "PC": np.arange(0.0, 6.0 + 1e-6, 2.0)}

HUMAN = {mv: {"tom": f"{mv.lower()}_tom_mean.npy",
             "valence": f"{mv.lower()}_valence_mean.npy"} for mv in MOVIES}


def _grid_len(mv):
    return FMRI_T[mv] - (PC_DROP_TR if mv == "PC" else 0)


def human_series(mv, measure):
    T = _grid_len(mv)
    lag_h = 0 if mv == "PC" else int(round(8.0 / DT))
    y = np.load(os.path.join(_ROOT, HUMAN[mv][measure])).astype(float)
    yc = y[lag_h:] if lag_h > 0 else y
    return interp_to_fmri_grid(np.arange(len(yc)) * DT, yc, TR[mv], T).astype(float)


def machine_csv(mv, model, measure, variant):
    sub = MODEL_DIR_SUFFIX[model]
    pre = MODEL_FILE_PREFIX[model]
    name = f"{pre}{mv}_{measure}{VARIANT_SUFFIX[variant]}.csv"
    return os.path.join(_ROOT, "embedding", f"{mv}_{sub}", name)


def machine_series(mv, model, measure, variant):
    T = _grid_len(mv)
    start = 10 if mv in ("TP", "DM") else 0
    y = sample2tr(machine_csv(mv, model, measure, variant), start_tr=start,
                  time_col="timestamp_sec", rating_col=RATING_COL[measure],
                  pc_downsample=(mv == "PC"))
    y = smooth_series(y, TR[mv], method="gaussian", sigma_sec=GAUSS_SIGMA_SEC)
    return np.asarray(y[:T], dtype=float)


def apply_lag(human, machine, lag_tr):
    if lag_tr == 0:
        return human, machine
    return human[lag_tr:], machine[:-lag_tr]


def pearson_at_lag(human, machine, lag_tr):
    a, b = apply_lag(human, machine, lag_tr)
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], float), np.asarray(b[:n], float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 8:
        return np.nan
    return float(pearsonr(a[mask], b[mask])[0])


rows = []
for mv in MOVIES:
    lags_tr = [int(round(s / TR[mv])) for s in LAG_SEC[mv]]
    for measure in MEASURES:
        H = human_series(mv, measure)
        for model in MODELS:
            for variant in VARIANTS:
                path = machine_csv(mv, model, measure, variant)
                if not os.path.exists(path):
                    continue
                M = machine_series(mv, model, measure, variant)
                best_r, best_lag = -np.inf, 0.0
                for lt, ls in zip(lags_tr, LAG_SEC[mv]):
                    r = pearson_at_lag(H, M, lt)
                    if np.isfinite(r) and r > best_r:
                        best_r, best_lag = r, ls
                rows.append(dict(movie=mv, movie_title=MOVIE_TITLE[mv], measure=measure,
                                 model=model, variant=variant,
                                 best_lag_sec=round(float(best_lag), 2),
                                 pearson_r=round(float(best_r), 4)))

df = pd.DataFrame(rows)
out_dir = os.path.join(_HERE, "results")
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, "context_ablation_bestlag_all.csv"), index=False)

# ── console summary: delta vs base, per model ───────────────────────────────
pd.set_option("display.width", 200, "display.max_rows", 400)
print("=== Context ablation (all 3 models): best-lag Pearson r vs Human ===\n")
for mv in MOVIES:
    for measure in MEASURES:
        for model in MODELS:
            sub = df[(df.movie == mv) & (df.measure == measure) & (df.model == model)].set_index("variant")
            if "base" not in sub.index:
                continue
            base_r = sub.loc["base", "pearson_r"]
            print(f"{MOVIE_TITLE[mv]} ({mv}) — {measure.upper()} — {MODEL_LABEL[model]}  [base r={base_r:+.3f}]")
            for variant in CONTEXT_VARIANTS:
                if variant not in sub.index:
                    print(f"    {variant:10s} (missing)")
                    continue
                r = sub.loc[variant, "pearson_r"]
                lag = sub.loc[variant, "best_lag_sec"]
                delta = r - base_r
                arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
                print(f"    {variant:10s} r={r:+.3f}  (lag={lag:4.1f}s)  Δ={delta:+.3f} {arrow}")
        print()

print(f"Saved -> {out_dir}/context_ablation_bestlag_all.csv")

# ── figure: 2 (measure) x 3 (movie), one line per model vs lookback (sec) ──
# x-axis is the actual lookback window in seconds (base=0), not the variant
# label -- context_frames/jump are just the mechanism (see context_variants.py
# docstring); what matters for this plot is how far back the injected
# context reaches. A line per model shows the trend (does longer context
# help or hurt) far more clearly than grouped bars.
VARIANT_XPOS = {"base": 0, "ctx_4s": 4, "ctx_8s": 8, "ctx_12s": 12, "ctx_16s": 16}
XTICK_LABELS = ["base", "4s", "8s", "12s", "16s"]

FONT = {"font.family": "Times New Roman", "font.size": 20}
# Unify all 6 panels onto the same y-axis scale so bar/line heights are
# directly comparable across movies and measures, not just within a panel.
y_lo = float(np.nanmin(df["pearson_r"]))
y_hi = float(np.nanmax(df["pearson_r"]))
y_pad = 0.05 * (y_hi - y_lo)
Y_LIM = (y_lo - y_pad, y_hi + y_pad)

with plt.rc_context(FONT):
    fig, axes = plt.subplots(len(MEASURES), len(MOVIES), figsize=(17, 9.5), squeeze=False)
    xpos = [VARIANT_XPOS[v] for v in VARIANTS]

    line_handles, line_labels = None, None
    for ri, measure in enumerate(MEASURES):
        for ci, mv in enumerate(MOVIES):
            ax = axes[ri][ci]
            for model in MODELS:
                sub = df[(df.movie == mv) & (df.measure == measure) & (df.model == model)].set_index("variant").reindex(VARIANTS)
                vals = sub["pearson_r"].to_numpy()
                mask = np.isfinite(vals)
                ax.plot(np.array(xpos)[mask], vals[mask], marker="o", ms=9, lw=2.8,
                        color=MODEL_COLOR[model], label=MODEL_LABEL[model])
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylim(Y_LIM)
            ax.set_xticks(xpos); ax.set_xticklabels(XTICK_LABELS, fontsize=20)
            ax.set_xlabel("Context lookback", fontsize=20)
            ax.set_title(f"{MOVIE_TITLE[mv]} - {measure.upper()}", fontsize=21)
            ax.grid(ls="--", lw=0.4, alpha=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=20)
            if ci == 0:
                ax.set_ylabel("Pearson r", fontsize=20)
            if ri == 0 and ci == 0:
                line_handles, line_labels = ax.get_legend_handles_labels()

    # Single legend, one row, centered below all 6 panels (not per-panel).
    fig.legend(line_handles, line_labels, loc="lower center", ncol=len(MODELS),
              frameon=False, fontsize=22, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    out_png = os.path.join(out_dir, "fig_context_ablation_all.png")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
print(f"Saved -> {out_png}")
