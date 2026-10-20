"""Shared rating helpers: load a rating CSV, put it on the fMRI TR grid, smooth it."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def interp_to_fmri_grid(t_rating, y_rating, tr_fmri, T_fmri):
    t_fmri = np.arange(T_fmri) * tr_fmri
    return np.interp(t_fmri, t_rating, y_rating)


def smooth_series(y, TR, method="gaussian", window_sec=6.0, sigma_sec=2.0):
    y = np.asarray(y, dtype=float)
    if method == "ma":
        W = max(1, int(round(window_sec / TR)))
        return np.convolve(y, np.ones(W) / W, mode="same")
    elif method == "gaussian":
        return gaussian_filter1d(y, sigma=max(0.5, sigma_sec / TR), mode="nearest")
    return y


def sample2tr(csv_path, start_tr=10, time_col="timestamp_sec",
              rating_col="tom_rating", pc_downsample=False):
    df = pd.read_csv(csv_path)
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    ratings = df[rating_col].dropna().to_numpy()[start_tr:]
    if pc_downsample:
        ratings = ratings[::2]
    return ratings.astype(float)
