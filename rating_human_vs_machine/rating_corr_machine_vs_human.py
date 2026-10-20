"""
rating_corr_machine_vs_human.py
Machine-rating vs human-rating correlation, per movie / measure / model /
ToM-variant. This is a RATING-vs-RATING comparison (no fMRI): both series are
put on the same fMRI-TR grid and Pearson-correlated.

Preprocessing (matches the methods text) -------------------------------------
  Human rating  : the 20%-low-corr-removed mean time series (npy, DT=0.25 s).
                  "resampled to the fMRI TR grid via linear interpolation
                   without additional smoothing."  -> interp_to_fmri_grid
  Machine rating: per-frame CSV ratings (embedding/).
                  "temporal smoothing with a Gaussian kernel prior to alignment
                   with the fMRI TR grid."  -> smooth_series(gaussian, sigma=2s)
                  The CSVs are sampled exactly at the TR (0.8 s for TP/DM,
                  1.0 s for PC -> [::2]), so alignment = drop start_tr + truncate.
  Both human and machine drop the same lead-in (8 s for TP/DM), so the two
  series stay mutually aligned; correlation is computed at zero relative lag.

Models -----------------------------------------------------------------------
  LLaDA-V   : embedding/<MV>_final/<MV>_*.csv
  Qwen3-VL  : embedding/<MV>_qwen/qwen_<MV>_*.csv
  Gemma 3   : embedding/<MV>_gemma/gemma_<MV>_*.csv

ToM variants -----------------------------------------------------------------
  base            : one description per frame
  contextN        : previous N frames of description added to the prompt
  ctx5_jK         : context-5 but sampling every K-th frame (jump, not contiguous)
Valence has a single version per movie.

Outputs ----------------------------------------------------------------------
  rating_corr_all.csv                 (tidy long table, every combination)
  llada_vs_qwen_correlation_repro.pdf/.png   (PDF-style bar figure)
"""
import os, sys, glob, re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # project root: data + rating_utils live here
sys.path.insert(0, _ROOT)
from rating_utils import interp_to_fmri_grid, sample2tr, smooth_series

DT     = 0.25
TR     = {"TP": 0.8, "DM": 0.8, "PC": 2.0}
FMRI_T = {"TP": 239, "DM": 739, "PC": 167}     # raw TR count of fmri_<mv>.npy
PC_DROP_TR = 5                                  # PC drops first 10 s (5 TR)
GAUSS_SIGMA_SEC = 2.0
MOVIES = ["TP", "DM", "PC"]
MOVIE_TITLE = {"TP": "The Present", "DM": "Despicable Me", "PC": "Partly Cloudy"}
MODELS = ["llada", "qwen", "gemma"]
MODEL_LABEL = {"llada": "LLaDA-V", "qwen": "Qwen3-VL", "gemma": "Gemma 3"}
MODEL_DIR_SUFFIX = {"llada": "final", "qwen": "qwen", "gemma": "gemma"}
MODEL_FILE_PREFIX = {"llada": "", "qwen": "qwen_", "gemma": "gemma_"}

HUMAN = {
    "TP": {"tom": "tp_tom_mean.npy", "valence": "tp_valence_mean.npy"},
    "DM": {"tom": "dm_tom_mean.npy", "valence": "dm_valence_mean.npy"},
    "PC": {"tom": "pc_tom_mean.npy", "valence": "pc_valence_mean.npy"},
}


# ── preprocessing ───────────────────────────────────────────────────────────
def _grid_len(mv):
    return FMRI_T[mv] - (PC_DROP_TR if mv == "PC" else 0)

def human_series(mv, kind):
    """Human mean rating -> fMRI TR grid, linear interp, no smoothing."""
    T = _grid_len(mv)
    lag_h = 0 if mv == "PC" else int(round(8.0 / DT))
    y = np.load(os.path.join(_ROOT, HUMAN[mv][kind])).astype(float)
    yc = y[lag_h:] if lag_h > 0 else y
    return interp_to_fmri_grid(np.arange(len(yc)) * DT, yc, TR[mv], T).astype(float)

def machine_series(mv, kind, csv_path):
    """Machine per-frame rating -> Gaussian smooth -> align to TR grid."""
    T = _grid_len(mv)
    col = "tom_rating" if kind == "tom" else "valence_rating"
    start = 10 if mv in ("TP", "DM") else 0
    y = sample2tr(csv_path, start_tr=start, time_col="timestamp_sec",
                  rating_col=col, pc_downsample=(mv == "PC"))
    y = smooth_series(y, TR[mv], method="gaussian", sigma_sec=GAUSS_SIGMA_SEC)
    return np.asarray(y[:T], dtype=float)

def rating_corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 6:
        return np.nan, int(m.sum())
    return float(pearsonr(a[m], b[m])[0]), int(m.sum())


# ── discover every machine CSV per (model, movie, measure, variant) ─────────
def model_dir(mv, model):
    return os.path.join(_ROOT, "embedding", f"{mv}_{MODEL_DIR_SUFFIX[model]}")

def variant_label(fname, mv, model):
    stem = re.sub(r"\.csv$", "", os.path.basename(fname))
    prefix = f"{MODEL_FILE_PREFIX[model]}{mv}_tom"
    v = stem[len(prefix):] if stem.startswith(prefix) else stem
    return v.lstrip("_") or "base"

def discover(mv, model, kind):
    d = model_dir(mv, model)
    pre = MODEL_FILE_PREFIX[model]
    if kind == "valence":
        p = os.path.join(d, f"{pre}{mv}_valence.csv")
        return [("base", p)] if os.path.exists(p) else []
    files = sorted(glob.glob(os.path.join(d, f"{pre}{mv}_tom*.csv")))
    return [(variant_label(f, mv, model), f) for f in files]


# ── compute full table ──────────────────────────────────────────────────────
rows = []
for mv in MOVIES:
    for kind in ["tom", "valence"]:
        h = human_series(mv, kind)
        for model in MODELS:
            for variant, csv in discover(mv, model, kind):
                m = machine_series(mv, kind, csv)
                r, n = rating_corr(h, m)
                rows.append(dict(movie=mv, movie_title=MOVIE_TITLE[mv],
                                 measure=kind, model=model,
                                 model_label=MODEL_LABEL[model],
                                 variant=variant, n=n, pearson_r=r,
                                 csv=os.path.relpath(csv, _HERE)))

df = pd.DataFrame(rows)
df_out = df[["movie", "measure", "model_label", "variant", "n", "pearson_r", "csv"]]
df_out.to_csv(os.path.join(_HERE, "rating_corr_all.csv"), index=False)

# pretty print
pd.set_option("display.width", 160, "display.max_rows", 200)
print("\n=== Machine-vs-Human rating correlation (all variants) ===")
for mv in MOVIES:
    for kind in ["tom", "valence"]:
        sub = df[(df.movie == mv) & (df.measure == kind)]
        print(f"\n{MOVIE_TITLE[mv]}  |  {kind.upper()}")
        for _, r in sub.iterrows():
            print(f"    {r.model_label:9s} {r.variant:10s} r={r.pearson_r:+.3f}  (n={r.n})")
print("\nSaved table -> rating_corr_all.csv")


# ── PDF-style bar figure (base ToM + valence) ───────────────────────────────
FIG_TOM_VARIANT = {  # which ToM variant to show per (movie, model); 'base' default
    # e.g. ("DM","llada"): "context3",
}
def pick(mv, kind, model):
    sub = df[(df.movie == mv) & (df.measure == kind) & (df.model == model)]
    if kind == "tom":
        want = FIG_TOM_VARIANT.get((mv, model), "base")
        hit = sub[sub.variant == want]
        if hit.empty:
            hit = sub[sub.variant == "base"]
        return float(hit.pearson_r.iloc[0]) if not hit.empty else np.nan
    return float(sub.pearson_r.iloc[0]) if not sub.empty else np.nan

COLORS = {"llada": "#4C72B0", "qwen": "#C44E52", "gemma": "#55A868"}
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
measures = ["tom", "valence"]
n_models = len(MODELS)
bw = 0.78 / n_models
xpos = np.arange(len(measures))
for ci, mv in enumerate(MOVIES):
    ax = axes[ci]
    for mi, model in enumerate(MODELS):
        vals = [pick(mv, kind, model) for kind in measures]
        offset = (mi - (n_models - 1) / 2) * bw
        bars = ax.bar(xpos + offset, vals, bw,
                      color=COLORS[model], label=MODEL_LABEL[model],
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(MOVIE_TITLE[mv], fontsize=15, fontweight="bold")
    ax.set_xticks(xpos); ax.set_xticklabels(["ToM", "Valence"], fontsize=13)
    ax.set_ylim(0, 0.8)
    ax.yaxis.grid(True, ls="--", lw=0.5, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Correlation with Human Ratings", fontsize=14)
axes[-1].legend(handles=[mpatches.Patch(color=COLORS[m], label=MODEL_LABEL[m])
                         for m in MODELS], loc="upper right", frameon=False, fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(_HERE, "llada_vs_qwen_correlation_repro.png"),
            dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(_HERE, "llada_vs_qwen_correlation_repro.pdf"),
            bbox_inches="tight")
print("Saved figure -> llada_vs_qwen_correlation_repro.png / .pdf")
