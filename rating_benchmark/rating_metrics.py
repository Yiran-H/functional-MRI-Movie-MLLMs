"""
rating_metrics.py — Multi-metric human-vs-machine rating comparison.

A RATING-vs-RATING benchmark (no fMRI): for each movie, measure (ToM / valence)
and model (LLaDA-V, Qwen3-VL) we put the human-mean and machine rating on the
same fMRI-TR grid and compare them with a battery of metrics, swept over a range
of temporal lags. A machine-machine pair (LLaDA-V vs Qwen3-VL) is included as an
inter-model consistency reference.

Preprocessing reuses the exact pipeline of
    ../rating_human_vs_machine/rating_corr_machine_vs_human.py
(human: 20%-low-corr-removed mean, linear interp to TR, no smoothing; machine:
per-frame CSV -> Gaussian sigma=2 s -> align to TR grid).

Lag convention -----------------------------------------------------------------
  Positive lag = MACHINE LEADS HUMAN (the human's manual rating trails the
  content the machine already scored). For lag L (in TR) we compare
      a = human[L:]      with   b = machine[:T-L]
  TP / DM : TR = 0.8 s, lags 0 .. 6.4 s  (0 .. 8 TR)
  PC      : TR = 2.0 s, lags 0 .. 6.0 s  (0 .. 3 TR)

Metrics ------------------------------------------------------------------------
  Scale-invariant (primary; the two rating instruments differ in scale):
    pearson_r, spearman_rho, mutual_info, sign_agree, kappa_tertile, dtw_z
  Common-scale / agreement:
    rmse_z, mae_z, ba_loa_width   on z-scored series -> residual shape misfit
  dtw_z is computed at lag 0 only (DTW already absorbs time shifts).

Outputs (rating_benchmark/results/) -------------------------------------------
  metrics_all_lags.csv     every movie x measure x pair x lag
  metrics_bestlag.csv      per pair, the lag maximizing Pearson (with all
                           metrics at that lag)
  pearson_vs_lag.png       Pearson r vs lag, per movie/measure/model
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import cohen_kappa_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from rating_utils import interp_to_fmri_grid, sample2tr, smooth_series

# ── config (matches rating_corr_machine_vs_human.py) ────────────────────────
DT      = 0.25
TR      = {"TP": 0.8, "DM": 0.8, "PC": 2.0}
FMRI_T  = {"TP": 239, "DM": 739, "PC": 167}
PC_DROP_TR = 5
GAUSS_SIGMA_SEC = 2.0
MOVIES  = ["TP", "DM", "PC"]
MOVIE_TITLE = {"TP": "The Present", "DM": "Despicable Me", "PC": "Partly Cloudy"}
MODELS  = ["llada", "qwen", "gemma"]
MODEL_LABEL = {"llada": "LLaDA-V", "qwen": "Qwen3-VL", "gemma": "Gemma 3"}
MODEL_DIR_SUFFIX = {"llada": "final", "qwen": "qwen", "gemma": "gemma"}
MODEL_FILE_PREFIX = {"llada": "", "qwen": "qwen_", "gemma": "gemma_"}
MEASURES = ["tom", "valence"]

# lag grids (seconds); TR-multiples chosen so shifts land on the grid
LAG_SEC = {"TP": np.arange(0.0, 6.4 + 1e-6, 0.8),
           "DM": np.arange(0.0, 6.4 + 1e-6, 0.8),
           "PC": np.arange(0.0, 6.0 + 1e-6, 2.0)}

HUMAN = {mv: {"tom": f"{mv.lower()}_tom_mean.npy",
             "valence": f"{mv.lower()}_valence_mean.npy"} for mv in MOVIES}


# ── series loaders ──────────────────────────────────────────────────────────
def _grid_len(mv):
    return FMRI_T[mv] - (PC_DROP_TR if mv == "PC" else 0)

def human_series(mv, kind):
    T = _grid_len(mv)
    lag_h = 0 if mv == "PC" else int(round(8.0 / DT))
    y = np.load(os.path.join(_ROOT, HUMAN[mv][kind])).astype(float)
    yc = y[lag_h:] if lag_h > 0 else y
    return interp_to_fmri_grid(np.arange(len(yc)) * DT, yc, TR[mv], T).astype(float)

def machine_csv(mv, kind, model):
    sub = MODEL_DIR_SUFFIX[model]
    pre = MODEL_FILE_PREFIX[model]
    name = f"{pre}{mv}_{'valence' if kind == 'valence' else 'tom'}.csv"
    return os.path.join(_ROOT, "embedding", f"{mv}_{sub}", name)

def machine_series(mv, kind, model):
    T = _grid_len(mv)
    col = "tom_rating" if kind == "tom" else "valence_rating"
    start = 10 if mv in ("TP", "DM") else 0
    y = sample2tr(machine_csv(mv, kind, model), start_tr=start,
                  time_col="timestamp_sec", rating_col=col,
                  pc_downsample=(mv == "PC"))
    y = smooth_series(y, TR[mv], method="gaussian", sigma_sec=GAUSS_SIGMA_SEC)
    return np.asarray(y[:T], dtype=float)


# ── metric helpers ──────────────────────────────────────────────────────────
def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()

def dtw_distance(x, y, band=None):
    """DTW on z-scored series, normalized by path length (n+m)."""
    x, y = _z(x), _z(y)
    n, m = len(x), len(y)
    if band is None:
        band = max(n, m)
    INF = np.inf
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        xi = x[i - 1]
        for j in range(j0, j1 + 1):
            cost = abs(xi - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))

def mutual_info(x, y):
    try:
        return float(mutual_info_regression(x.reshape(-1, 1), y,
                                            random_state=0)[0])
    except Exception:
        return np.nan

def sign_agree(x, y):
    sx = np.sign(x - np.median(x)); sy = np.sign(y - np.median(y))
    m = (sx != 0) & (sy != 0)
    return float(np.mean(sx[m] == sy[m])) if m.sum() else np.nan

def kappa_tertile(x, y):
    def terc(v):
        q = np.quantile(v, [1 / 3, 2 / 3])
        return np.digitize(v, q)
    try:
        return float(cohen_kappa_score(terc(x), terc(y)))
    except Exception:
        return np.nan

def all_metrics(a, b, with_dtw=False):
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], float), np.asarray(b[:n], float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 8:
        return {}
    az, bz = _z(a), _z(b)
    diff = az - bz
    out = dict(
        n=len(a),
        pearson_r=float(pearsonr(a, b)[0]),
        spearman_rho=float(spearmanr(a, b)[0]),
        rmse_z=float(np.sqrt(np.mean(diff ** 2))),
        mae_z=float(np.mean(np.abs(diff))),
        ba_loa_width=float(2 * 1.96 * diff.std()),   # Bland-Altman LoA width (z)
        mutual_info=mutual_info(a, b),
        sign_agree=sign_agree(a, b),
        kappa_tertile=kappa_tertile(a, b),
        dtw_z=dtw_distance(a, b) if with_dtw else np.nan,
    )
    return out


# ── lag application (positive = machine leads human) ────────────────────────
def apply_lag(human, machine, lag_tr):
    if lag_tr == 0:
        return human, machine
    return human[lag_tr:], machine[:-lag_tr]


# ── main sweep ──────────────────────────────────────────────────────────────
def series_for(mv, kind, who):
    return human_series(mv, kind) if who == "human" else machine_series(mv, kind, who)

import itertools
PAIRS = list(itertools.combinations(["human"] + MODELS, 2))

def pair_label(p):
    lab = {"human": "Human", **MODEL_LABEL}
    return f"{lab[p[0]]}–{lab[p[1]]}"

rows = []
for mv in MOVIES:
    lags_tr = [int(round(s / TR[mv])) for s in LAG_SEC[mv]]
    for kind in MEASURES:
        cache = {}
        for a_who, b_who in PAIRS:
            A = cache.setdefault(a_who, series_for(mv, kind, a_who))
            B = cache.setdefault(b_who, series_for(mv, kind, b_who))
            for lt, ls in zip(lags_tr, LAG_SEC[mv]):
                a, b = apply_lag(A, B, lt)
                met = all_metrics(a, b, with_dtw=(lt == 0))
                if not met:
                    continue
                rows.append(dict(movie=mv, movie_title=MOVIE_TITLE[mv],
                                 measure=kind, pair=pair_label((a_who, b_who)),
                                 a=a_who, b=b_who,
                                 lag_tr=lt, lag_sec=round(float(ls), 2), **met))

df = pd.DataFrame(rows)
out_dir = os.path.join(_HERE, "results")
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, "metrics_all_lags.csv"), index=False)

# ── best-lag table (by Pearson) ──────────────────────────────────────────────
best_rows = []
for (mv, kind, pair), g in df.groupby(["movie", "measure", "pair"]):
    best = g.loc[g["pearson_r"].idxmax()]
    best_rows.append(dict(movie=mv, measure=kind, pair=pair,
                          best_lag_sec=best["lag_sec"],
                          pearson_r=best["pearson_r"], spearman_rho=best["spearman_rho"],
                          rmse_z=best["rmse_z"], mae_z=best["mae_z"],
                          ba_loa_width=best["ba_loa_width"],
                          mutual_info=best["mutual_info"], sign_agree=best["sign_agree"],
                          kappa_tertile=best["kappa_tertile"]))
best = pd.DataFrame(best_rows)
best.to_csv(os.path.join(out_dir, "metrics_bestlag.csv"), index=False)

# ── console summary ─────────────────────────────────────────────────────────
pd.set_option("display.width", 200, "display.max_rows", 300)
print("\n=== best lag by Pearson (human pairs) ===")
show = best[best.pair.str.startswith("Human")]
print(show[["movie", "measure", "pair", "best_lag_sec", "pearson_r",
            "spearman_rho", "sign_agree", "kappa_tertile"]].to_string(index=False))

print(f"\nSaved -> {out_dir}/metrics_all_lags.csv")
print(f"Saved -> {out_dir}/metrics_bestlag.csv")

# ── figure: Pearson vs lag (human pairs) ─────────────────────────────────────
fig, axes = plt.subplots(len(MEASURES), len(MOVIES), figsize=(15, 8), squeeze=False)
plt.rcParams.update({"font.size": 11})
colors = {"llada": "#4C72B0", "qwen": "#C44E52", "gemma": "#55A868"}
for ri, kind in enumerate(MEASURES):
    for ci, mv in enumerate(MOVIES):
        ax = axes[ri][ci]
        for model in MODELS:
            g = df[(df.movie == mv) & (df.measure == kind) &
                   (df.a == "human") & (df.b == model)].sort_values("lag_sec")
            if g.empty:
                continue
            ax.plot(g.lag_sec, g.pearson_r, "-o", color=colors[model],
                    label=MODEL_LABEL[model], ms=4)
        ax.set_title(f"{MOVIE_TITLE[mv]} — {kind.upper()}", fontsize=12)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(True, ls="--", lw=0.4, alpha=0.6)
        if ci == 0:
            ax.set_ylabel("Pearson r")
        if ri == len(MEASURES) - 1:
            ax.set_xlabel("machine-leads lag (s)")
        if ri == 0 and ci == len(MOVIES) - 1:
            ax.legend(fontsize=8, loc="best")
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "pearson_vs_lag.png"), dpi=200,
            bbox_inches="tight")
print(f"Saved -> {out_dir}/pearson_vs_lag.png")
