"""
plot_valence_timeseries_all_models.py
Raw (native-scale, NOT z-scored) valence time series for all three movies,
each with Human + LLaDA-V + Qwen3-VL + Gemma 3 overlaid on one plot.

Preprocessing matches rating_corr_machine_vs_human.py:
  Human  : 20%-low-corr-removed mean, linear interp to fMRI-TR grid, no smoothing.
  Machine: per-frame CSV rating -> Gaussian smooth (sigma=2s) -> align to TR grid.
Both stay on their native -10..+10 rating scale (no z-scoring), so this shows
the actual magnitude/amplitude differences between models, not just shape.

Usage:
    python plot_valence_timeseries_all_models.py
Outputs (this directory), one figure per movie, each with human + all 3 models:
    valence_timeseries_TP_all_models.png/.pdf
    valence_timeseries_DM_all_models.png/.pdf
    valence_timeseries_PC_all_models.png/.pdf
"""
import os, sys
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from rating_utils import interp_to_fmri_grid, sample2tr, smooth_series

DT = 0.25
TR = {"TP": 0.8, "DM": 0.8, "PC": 2.0}
FMRI_T = {"TP": 239, "DM": 739, "PC": 167}
PC_DROP_TR = 5
GAUSS_SIGMA_SEC = 2.0
MOVIES = ["TP", "DM", "PC"]
MOVIE_TITLE = {"TP": "The Present", "DM": "Despicable Me", "PC": "Partly Cloudy"}
MODELS = ["llada", "qwen", "gemma"]
MODEL_LABEL = {"llada": "LLaDA-V", "qwen": "Qwen3-VL", "gemma": "Gemma 3"}
MODEL_DIR_SUFFIX = {"llada": "final", "qwen": "qwen", "gemma": "gemma"}
MODEL_FILE_PREFIX = {"llada": "", "qwen": "qwen_", "gemma": "gemma_"}
COLORS = {"llada": "#4C72B0", "qwen": "#C44E52", "gemma": "#55A868"}

HUMAN_NPY = {
    "TP": "tp_valence_mean.npy",
    "DM": "dm_valence_mean.npy",
    "PC": "pc_valence_mean.npy",
}

def _grid_len(mv):
    return FMRI_T[mv] - (PC_DROP_TR if mv == "PC" else 0)

def human_series(mv):
    T = _grid_len(mv)
    lag_h = 0 if mv == "PC" else int(round(8.0 / DT))
    y = np.load(os.path.join(_ROOT, HUMAN_NPY[mv])).astype(float)
    yc = y[lag_h:] if lag_h > 0 else y
    return interp_to_fmri_grid(np.arange(len(yc)) * DT, yc, TR[mv], T).astype(float)

def machine_csv(mv, model):
    d = os.path.join(_ROOT, "embedding", f"{mv}_{MODEL_DIR_SUFFIX[model]}")
    return os.path.join(d, f"{MODEL_FILE_PREFIX[model]}{mv}_valence.csv")

def machine_series(mv, model):
    T = _grid_len(mv)
    start = 10 if mv in ("TP", "DM") else 0
    y = sample2tr(machine_csv(mv, model), start_tr=start, time_col="timestamp_sec",
                  rating_col="valence_rating", pc_downsample=(mv == "PC"))
    y = smooth_series(y, TR[mv], method="gaussian", sigma_sec=GAUSS_SIGMA_SEC)
    return np.asarray(y[:T], dtype=float)

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})

for mv in MOVIES:
    T = _grid_len(mv)
    t = np.arange(T) * TR[mv]
    human = human_series(mv)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t, human, color="#1f1f1f", lw=1.6, label="Human rating")

    for model in MODELS:
        machine = machine_series(mv, model)
        n = min(len(human), len(machine))
        mask = np.isfinite(human[:n]) & np.isfinite(machine[:n])
        r = pearsonr(human[:n][mask], machine[:n][mask])[0] if mask.sum() >= 6 else np.nan
        ax.plot(t[:len(machine)], machine, color=COLORS[model], lw=1.3, ls="--",
                label=f"{MODEL_LABEL[model]} (r={r:.3f})")

    ax.axhline(0, color="#999999", lw=0.6, ls=":")
    ax.set_xlabel("Time (seconds)", fontsize=13)
    ax.set_ylabel("Valence rating (-10 to +10)", fontsize=13)
    ax.set_title(f"{MOVIE_TITLE[mv]} ({mv}) — Valence: Human vs all models (raw scale)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, t[-1])
    ax.legend(frameon=False, fontsize=11, loc="upper right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = os.path.join(_HERE, f"valence_timeseries_{mv}_all_models.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out} (+.pdf)")
