"""
build_features_human_vs_model.py — Build a 4-block feature dict
("human_tom", "human_valence", "model_tom", "model_valence") per (movie,
model), for the human-vs-model variance-partitioning question. Each measure
(ToM / Valence) is kept as its own 1-D block so unique/shared variance can be
computed SEPARATELY per measure (human_tom vs model_tom, then human_valence
vs model_valence) -- combining both measures into one "human"/"model" block
would conflate two different constructs into a single unique-variance number.

Base ratings only (no context variants). Uses the SAME per-movie alignment
convention as build_features.py / build_features_tpdm.py (start_tr, TR_FMRI,
DT, pc_downsample, fmri_start) and the same human/machine preprocessing
convention as the rest of this project (rating_utils.py):
  human : 20%-low-corr-removed mean, linear interp to fMRI TR grid, no smoothing.
  model : per-frame CSV rating -> Gaussian smooth (sigma=2s) -> TR grid.
Features are built at ZERO lag (the 8s human/model lag is already baked into
start_tr / lag_h below, matching build_features.py's convention) -- the
FIR Delayer in the encoding pipeline handles the hemodynamic lag.

Usage:
    python build_features_human_vs_model.py --movie tp --model gemma
Output:
    fea_{movie}_{model}_hvm.npy   pickled dict:
        {"human_tom": [T,1], "human_valence": [T,1],
         "model_tom": [T,1], "model_valence": [T,1]}
The runner then selects feature_order=["human_<measure>", "model_<measure>"]
per run (load_data() only loads the requested keys).
"""
import argparse
import os
import numpy as np
import pandas as pd

DT = 0.25
TR = {"tp": 0.8, "dm": 0.8, "pc": 2.0}
FMRI_T = {"tp": 239, "dm": 739, "pc": 167}
PC_DROP_TR = 5
GAUSS_SIGMA_SEC = 2.0
MEASURES = ["tom", "valence"]

HUMAN_NPY = {
    "tp": {"tom": "tp_tom_mean.npy", "valence": "tp_valence_mean.npy"},
    "dm": {"tom": "dm_tom_mean.npy", "valence": "dm_valence_mean.npy"},
    "pc": {"tom": "pc_tom_mean.npy", "valence": "pc_valence_mean.npy"},
}

MODEL_DIR_SUFFIX = {"llada": "final", "qwen": "qwen", "gemma": "gemma"}
MODEL_FILE_PREFIX = {"llada": "", "qwen": "qwen_", "gemma": "gemma_"}


def interp_to_fmri_grid(t_rating, y_rating, tr_fmri, T_fmri):
    t_fmri = np.arange(T_fmri) * tr_fmri
    return np.interp(t_fmri, t_rating, y_rating)


def sample2tr(csv_path, start_tr, time_col, rating_col, pc_downsample):
    df = pd.read_csv(csv_path)
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    ratings = df[rating_col].dropna().to_numpy()[start_tr:]
    if pc_downsample:
        ratings = ratings[::2]
    return ratings.astype(float)


def smooth_gaussian(y, tr, sigma_sec=GAUSS_SIGMA_SEC):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.asarray(y, dtype=float), sigma=max(0.5, sigma_sec / tr), mode="nearest")


def grid_len(movie):
    return FMRI_T[movie] - (PC_DROP_TR if movie == "pc" else 0)


def human_series(movie, measure):
    T = grid_len(movie)
    lag_h = 0 if movie == "pc" else int(round(8.0 / DT))
    y = np.load(HUMAN_NPY[movie][measure]).astype(float)
    yc = y[lag_h:] if lag_h > 0 else y
    return interp_to_fmri_grid(np.arange(len(yc)) * DT, yc, TR[movie], T)


def model_series(movie, model, measure, sigma_sec=GAUSS_SIGMA_SEC):
    T = grid_len(movie)
    start = 10 if movie in ("tp", "dm") else 0
    sub = MODEL_DIR_SUFFIX[model]
    pre = MODEL_FILE_PREFIX[model]
    MV = movie.upper()
    col = "tom_rating" if measure == "tom" else "valence_rating"
    csv_path = f"./embedding/{MV}_{sub}/{pre}{MV}_{measure}.csv"
    y = sample2tr(csv_path, start_tr=start, time_col="timestamp_sec",
                  rating_col=col, pc_downsample=(movie == "pc"))
    y = smooth_gaussian(y, TR[movie], sigma_sec=sigma_sec)
    return np.asarray(y[:T], dtype=float)


def build(movie, model, out_path=None, sigma_sec=GAUSS_SIGMA_SEC):
    blocks = {}
    lens = []
    for measure in MEASURES:
        h = human_series(movie, measure)
        m = model_series(movie, model, measure, sigma_sec=sigma_sec)
        blocks[f"human_{measure}"] = h
        blocks[f"model_{measure}"] = m
        lens += [len(h), len(m)]
    T = min(lens)
    feats = {k: v[:T].reshape(-1, 1) for k, v in blocks.items()}

    if out_path is None:
        out_path = f"fea_{movie}_{model}_hvm.npy"
    np.save(out_path, feats, allow_pickle=True)
    shapes = {k: v.shape for k, v in feats.items()}
    print(f"Saved -> {out_path}  {shapes}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", choices=["tp", "dm", "pc"], required=True)
    ap.add_argument("--model", choices=["llada", "qwen", "gemma"], required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build(args.movie, args.model, args.out)
