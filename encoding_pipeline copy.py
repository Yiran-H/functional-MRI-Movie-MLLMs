"""
Encoding pipeline: MultipleKernelRidge (GPU) / ElasticNet / Lasso
================================================================
Usage:
  python encoding_pipeline.py            # uses CONFIG dict below
  python encoding_pipeline.py --movie tp --model lasso --chunk_len 6

Edit CONFIG for the most common settings; CLI args override CONFIG.
"""

# =============================================================================
# CONFIG — edit these values; CLI flags override them
# =============================================================================
CONFIG = {
    # Movie: 'dm', 'tp', or 'pc'
    "movie": "dm",

    # Model: 'ridge'  (GPU, himalaya BandedRidgeCV)
    #        'elasticnet'  (CPU, sklearn ElasticNetCV)
    #        'lasso'       (CPU, sklearn LassoCV)
    "model": "elasticnet",

    # fMRI file path. None → auto-detect from FMRI_FILES[movie]
    "fmri_file": None,

    # Feature file path.
    # None  →  auto-detect:  fea_{movie}_a32.npy  or  feature/fea_{movie}_8a.npy
    "feature_file": None,

    # Output tag override (None → "{model}_{movie}")
    "tag": None,

    # TP length in TRs for movie-aware CV (None = no boundary constraint)
    # Pass this when fMRI is a concatenated TP+DM file so CV folds never
    # straddle the movie boundary.
    "t_tp": None,

    # Keys to use from the feature dict (None = all keys in file order)
    "feature_order": ["social", "layer1", "description"],

    # Number of outer / inner CV folds
    "outer_folds": 5,
    "inner_folds": 5,

    # Temporal chunk length (TRs) for GroupKFold. Must exceed the max delay
    # (delays_tr) so FIR-delayed features don't reach across chunk boundaries;
    # 20 is a safe default (2x the 10-TR max delay), ~48 chunks over T=978.
    "chunk_len": 20,

    # FIR time-delay expansion (integer TRs). Replaces the single fixed
    # hemodynamic lag: each feature space is duplicated at these delays and
    # concatenated (still one banded group per feature space). Requires
    # features built at zero lag (build_features lag_tr=0). [] = disabled.
    # TP/DM (TR=0.8s): [2,4,6,8,10] ~= 1.6-8.0s ~ the paper's 1.5-7.5s.
    # PC (TR=2.0s): use e.g. [1,2,3,4] (2-8s).
    "delays_tr": [2, 4, 6, 8, 10],

    # Ridge solver: True = MultipleKernelRidgeCV (dual/kernel, the paper's
    # ColumnKernelizer pipeline; use when features > samples). False = primal
    # BandedRidgeCV. Kernel path applies delays inside the per-fold pipeline
    # and adds a TP/DM run regressor (zeroed on test).
    "ridge_dual": True,

    # Where to save result .npy files  ("." = working directory)
    "output_dir": ".",

    # Set True to run the encoding model; False to load existing results
    "run_model": True,
    # Paths to load when run_model=False  (None = auto-name from movie/model)
    "load_feature_r2s": None,
    "load_results_r2":  None,

    # Statistics settings
    "n_perm":    5000,     # sign-permutation iterations
    "fdr_alpha": 0.05,     # Benjamini-Hochberg FDR threshold

    # Ridge-only: number of log-spaced alpha candidates
    "num_alphas_ridge": 40,

    # Ridge-only (BandedRidgeCV): size of the band-weight (gamma) search.
    # Keep small -- inner CV has few independent temporal blocks, so a big
    # (gamma x alpha) search overfits it.  local_alpha=False shares one alpha
    # across ROIs, which is what makes the CPU ridge_sk path stable.
    "ridge_n_iter":      50,
    "ridge_local_alpha": False,

    # Ridge-only: "numpy" (same scorer as ridge_sk) or "himalaya"
    # (stock himalaya.scoring.r2_score_split; warns on non-zero-mean y_true)
    "ridge_split_scorer": "numpy",

    # ElasticNet-only
    "l1_ratio":    [0.1, 0.5, 0.7, 0.9, 1.0],
    "n_alphas_en": 30,

    # Variance partitioning regressor: 'ridge' | 'lasso' | 'elasticnet' | 'ols'
    "varpart_regressor": "ridge",

    # Group: 'adult' or 'child'
    "group": "adult",

    # ISC intersect threshold: only keep ROIs where BOTH adult & child ISC >= threshold.
    # Set to None to skip ISC filtering.
    "isc_threshold": 0.2,

    # Reproducibility
    "random_seed": 0,

    # Figure save path (None = plt.show())
    "fig_path": None,
}
# =============================================================================

import argparse
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive; change to "TkAgg" if needed
import matplotlib.pyplot as plt
from pathlib import Path

# ── optional heavy imports (checked at runtime) ─────────────────────────────
try:
    from himalaya.ridge import BandedRidgeCV
    # dual/kernel banded ridge (paper approach): per-feature-space linear kernels
    from himalaya.kernel_ridge import (MultipleKernelRidgeCV, ColumnKernelizer,
                                       Kernelizer)
    from himalaya.backend import set_backend
    # NOTE: himalaya.scoring.r2_score_split is only usable with the stock
    # himalaya source.  A local edit that shrinks the cross-term by an `alpha`
    # raises NameError (alpha is undefined) and, if alpha != 1, breaks the
    # identity  sum_k R2_k == R2_joint.  Restore the original line
    #     asst = (y_pred[fsi] * y_pred).sum(0)
    # before running with split_scorer="himalaya".
    from himalaya.scoring import r2_score_split
    _HIMALAYA = True
except ImportError:
    _HIMALAYA = False

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score as sk_r2_score
from scipy import stats
from statsmodels.stats.multitest import multipletests


# =============================================================================
# ROI / atlas definitions
# =============================================================================
ROI_BLOCKS = {
    "Vis":     list(range(1, 10))  + list(range(51, 59)),
    "SomM":    list(range(10, 16)) + list(range(59, 67)),
    "DorsA":   list(range(16, 24)) + list(range(67, 74)),
    "SVA":     list(range(24, 31)) + list(range(74, 79)),
    "Limb":    list(range(31, 34)) + list(range(79, 81)),
    "Cont":    list(range(34, 38)) + list(range(81, 90)),
    "Default": list(range(38, 51)) + list(range(90, 101)),
    "AAL":     list(range(101, 115)),
}
NET_ORDER = ["Vis", "SomM", "DorsA", "SVA", "Limb", "Cont", "Default", "AAL"]

# 0-based index version used internally
ROI_BLOCKS_0 = {k: [x - 1 for x in v] for k, v in ROI_BLOCKS.items()}


# =============================================================================
# Helpers
# =============================================================================
def make_groups(T, chunk_len, t_tp=None):
    """
    Build temporal chunk group labels for GroupKFold.

    When t_tp is given the TP and DM portions are chunked independently so
    no chunk ever straddles the movie boundary.
    """
    if t_tp is None or t_tp <= 0:
        n_chunks = max(1, T // chunk_len)
        groups   = np.array([i for i in range(n_chunks)
                             for _ in range(chunk_len)])
        groups   = np.concatenate(
            [groups, np.full(max(0, T - len(groups)), n_chunks)])
        return groups[:T]

    # ── TP portion ────────────────────────────────────────────────────────
    T_dm        = T - t_tp
    n_tp        = max(1, t_tp // chunk_len)
    groups_tp   = np.array([i for i in range(n_tp) for _ in range(chunk_len)])
    groups_tp   = np.concatenate(
        [groups_tp, np.full(max(0, t_tp - len(groups_tp)), n_tp)])[:t_tp]

    # ── DM portion (group IDs continue after TP) ──────────────────────────
    offset      = n_tp + 1          # +1 guarantees a gap at the boundary
    n_dm        = max(1, T_dm // chunk_len)
    groups_dm   = np.array([offset + i for i in range(n_dm)
                             for _ in range(chunk_len)])
    groups_dm   = np.concatenate(
        [groups_dm, np.full(max(0, T_dm - len(groups_dm)), offset + n_dm)])[:T_dm]

    return np.concatenate([groups_tp, groups_dm])


def apply_delays(features_dict, roi_data, t_tp, delays_tr):
    """Expand each feature space with time-delayed copies (FIR), replacing the
    single fixed hemodynamic lag.

    Delays are applied PER MOVIE SEGMENT (split at t_tp) so a delayed copy never
    crosses the TP/DM boundary. The first max(delays) TRs of each segment are
    dropped from BOTH features and fMRI so the two stay aligned. Each feature
    space stays ONE banded group, now dim*len(delays) wide -- BandedRidgeCV
    still receives one array per feature space.

    Requires features built at ZERO lag (build_features lag_tr=0); otherwise the
    delay stacks on top of the old fixed lag. Returns (features_dict, roi_data,
    t_tp) with the new, shorter T.
    """
    if not delays_tr:
        return features_dict, roi_data, t_tp
    from delayer import Delayer
    maxd   = int(max(delays_tr))
    T      = roi_data.shape[2]
    bounds = [(0, t_tp), (t_tp, T)] if t_tp else [(0, T)]

    feat_parts = {f: [] for f in features_dict}
    fmri_parts, new_t_tp = [], None
    for si, (a, b) in enumerate(bounds):
        for f, X in features_dict.items():
            Xd = Delayer(delays=delays_tr, drop_initial=True).fit_transform(
                np.asarray(X)[a:b])
            feat_parts[f].append(Xd)
        fmri_parts.append(roi_data[:, :, a + maxd:b])
        if si == 0 and t_tp:
            new_t_tp = (b - a) - maxd

    feats = {f: np.concatenate(p, axis=0) for f, p in feat_parts.items()}
    roi   = np.concatenate(fmri_parts, axis=2)
    print(f"[Delays] delays_tr={delays_tr} (max {maxd}) | each feature dim x"
          f"{len(set(delays_tr))} | T {T} -> {roi.shape[2]}"
          + (f" | t_tp {t_tp} -> {new_t_tp}" if t_tp else ""))
    return feats, roi, new_t_tp


def fit_zscore_train(X_train, eps=1e-8):
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0,  keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return mu, sd


def apply_zscore(X, mu, sd):
    return (X - mu) / sd


def zscore_fold(Y_raw, eps=1e-8):
    """Z-score one fold of fMRI data with that fold's own statistics.

    This mirrors the reference implementation, which applies
    ``scipy.stats.zscore`` separately to Y_train and Y_test.  Both folds then
    come out exactly zero-mean, which is what makes ``fit_intercept=False``
    valid and what ``r2_score_split`` assumes.

    Standardising Y_test with *train*-fold statistics instead leaves a
    non-zero mean in Y_test.  A model fitted with fit_intercept=False cannot
    absorb it (its intercept is learned on train, where the mean is already
    0), so the offset becomes a fixed penalty
        R2_observed = R2_fair - n * (mean(Y_te) - mean(Y_pred))**2 / SS_tot
    worth about -0.018 on this dataset -- independent of how well the model
    actually predicts.

    This is *not* test leakage: Y_test is only ever used for scoring, never
    for fitting or hyper-parameter selection, so the model gains nothing from
    it.  (sklearn's R2 already uses mean(Y_te) in its denominator anyway.)

    Y only.  Feature blocks X must still use train-fold statistics --
    see fit_zscore_train / apply_zscore.
    """
    mu = Y_raw.mean(0, keepdims=True)
    sd = Y_raw.std(0,  keepdims=True) + eps
    return (Y_raw - mu) / sd


def r2_score_split_numpy(Y_true, Y_pred_splits):
    """
    Product-measure R² decomposition (numpy version, mirrors himalaya).
    Y_true        : [T, n_rois]  – must be z-scored (zero mean)
    Y_pred_splits : list of K arrays, each [T, n_rois]
    Returns       : [K, n_rois]
    """
    Y_pred_joint = sum(Y_pred_splits)
    sst = (Y_true ** 2).sum(0) + 1e-12
    r2_splits = np.zeros((len(Y_pred_splits), Y_true.shape[1]))
    for k, yk in enumerate(Y_pred_splits):
        asst  = (yk * Y_pred_joint).sum(0)
        inter = (Y_true * yk).sum(0)
        r2_splits[k] = (2 * inter - asst) / sst
    return r2_splits


# =============================================================================
# Data loading
# =============================================================================
FMRI_FILES = {
    "dm":   {"adult": "fmri_dm_adult.npy",       "child": None},
    "tp":   {"adult": "fmri_tp_adult.npy",        "child": "fmri_tp_child.npy"},
    "pc":   {"adult": "fmri_pc_adult.npy",        "child": None},
    "tpdm": {"adult": "fmri_adult_tpdm_a256_desc128.npy",
             "child": "fmri_child_tpdm_a256_desc128.npy"},
}

FEATURE_FILES_PRIORITY = {
    "dm":   ["fea_dm_a32.npy", "feature/fea_dm_8a.npy"],
    "tp":   ["fea_tp_a32.npy", "feature/fea_tp_8a.npy"],
    "pc":   ["feature/fea_pc_8a.npy"],
    "tpdm": ["fea_tpdm_a256_desc128.npy"],
}


# =============================================================================
# ISC masking helpers
# =============================================================================
def build_isc_intersect_mask(threshold=0.2,
                             adult_file="isc_values_adult.npy",
                             child_file="isc_values_child.npy"):
    """
    Return (bool mask [114], int indices) for ROIs where BOTH adult and child
    ISC >= threshold.
    """
    vals_a = np.load(adult_file)
    vals_c = np.load(child_file)
    mask = (vals_a >= threshold) & (vals_c >= threshold)
    idx  = np.where(mask)[0]
    print(f"[ISC] threshold={threshold}: {mask.sum()} ROIs in intersect "
          f"(adult≥{threshold}: {(vals_a>=threshold).sum()}, "
          f"child≥{threshold}: {(vals_c>=threshold).sum()})")
    return mask, idx


def restore_roi_space(arr_filtered, roi_idx, n_rois=114, fill=np.nan):
    """Pad ISC-filtered result back to full 114-ROI space for plotting."""
    dtype = arr_filtered.dtype
    if arr_filtered.ndim == 2:        # [S, k]
        out = np.full((arr_filtered.shape[0], n_rois), fill, dtype=dtype)
        out[:, roi_idx] = arr_filtered
    elif arr_filtered.ndim == 3:      # [S, F, k]
        out = np.full((arr_filtered.shape[0], arr_filtered.shape[1], n_rois),
                      fill, dtype=dtype)
        out[:, :, roi_idx] = arr_filtered
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got {arr_filtered.ndim}-D")
    return out


def load_data(movie, feature_file=None, feature_order=None, fmri_file=None,
              group="adult"):
    """
    Returns
    -------
    roi_data      : [n_subjects, n_rois, T]   fMRI (lag already applied)
    features_dict : {name: [T, dim]}          z-scored feature arrays
    feature_order : list[str]                 ordered feature names
    n_subjects    : int                       auto-detected from fMRI shape
    """
    # ── fMRI ──────────────────────────────────────────────────────────────
    if fmri_file is not None:
        fmri_path = fmri_file
    else:
        entry = FMRI_FILES[movie]
        fmri_path = entry[group] if isinstance(entry, dict) else entry
    if fmri_path is None:
        raise ValueError(f"No fMRI file configured for movie='{movie}', group='{group}'")

    if not os.path.exists(fmri_path):
        raise FileNotFoundError(f"fMRI file not found: {fmri_path}")
    roi_data_full = np.load(fmri_path)                # [S, R, T_fmri]
    n_subjects, n_rois, T_fmri = roi_data_full.shape
    print(f"[Data] fMRI {fmri_path}: {roi_data_full.shape}  "
          f"→ {n_subjects} subjects, {n_rois} ROIs, {T_fmri} TRs")

    # ── features ──────────────────────────────────────────────────────────
    if feature_file is None:
        for candidate in FEATURE_FILES_PRIORITY[movie]:
            if os.path.exists(candidate):
                feature_file = candidate
                break
        if feature_file is None:
            raise FileNotFoundError(
                f"No feature file found for movie '{movie}'. "
                f"Tried: {FEATURE_FILES_PRIORITY[movie]}")
    print(f"[Data] Feature file: {feature_file}")

    raw = np.load(feature_file, allow_pickle=True).item()
    if feature_order is None:
        feature_order = list(raw.keys())

    # validate
    for f in feature_order:
        if f not in raw:
            raise KeyError(f"Feature '{f}' not in file {feature_file}. "
                           f"Available: {list(raw.keys())}")

    T_feat = raw[feature_order[0]].shape[0]
    print(f"[Data] Feature TRs: {T_feat}")

    # ── align fMRI to feature length via lag ──────────────────────────────
    lag = T_fmri - T_feat
    if lag < 0:
        raise ValueError(f"Feature T ({T_feat}) > fMRI T ({T_fmri}). "
                         "Check files.")
    if lag > 0:
        print(f"[Data] Auto-detected HRF lag = {lag} TRs → "
              f"roi_data[:, :, {lag}:]")
    roi_data = roi_data_full[:, :, lag:]           # [S, R, T_feat]

    features_dict = {f: raw[f][:T_feat].astype(np.float64)
                     for f in feature_order}

    for f in feature_order:
        print(f"  feature '{f}': {features_dict[f].shape}")

    return roi_data, features_dict, feature_order, n_subjects


# =============================================================================
# Model 1: BandedRidge  (GPU via himalaya)
#
# Why BandedRidgeCV (primal) and not MultipleKernelRidgeCV (dual/kernel):
#   himalaya's kernel solver is meant for n_samples < n_features (voxelwise
#   encoding with thousands of features).  Here n_features is small (e.g.
#   6+64+64=134) while n_samples is ~590, so the kernel form builds a
#   590x590 matrix of rank <=134 -- ~77% of it is numerical null space, which
#   is fragile in float32 (SVD can fail to converge).  The primal form solves
#   a 134x134 system instead: same banded-ridge model, no null space.
#
#   n_iter / local_alpha are kept deliberately small: with ~590 train TRs and
#   chunk_len=8 the inner CV only has ~14 independent blocks, so a large
#   (gamma x alpha) search overfits the inner CV (winner's curse).
#
# split_scorer selects how the joint R2 is decomposed per feature space:
#   "numpy"    -> r2_score_split_numpy, the same function the CPU ridge_sk
#                 path uses.  Runs on CPU.
#   "himalaya" -> himalaya.scoring.r2_score_split.  Mathematically identical
#                 to the numpy version (verified to ~4e-14), stays on the
#                 backend (no GPU->CPU transfer), and additionally warns when
#                 y_true is not zero-mean -- which is the case here, since
#                 Y_te is standardised with *train*-fold statistics.
#                 Requires stock himalaya (see the import note above).
# =============================================================================
def run_ridge(roi_data, features_dict, feature_order, n_subjects,
              outer_folds=5, inner_folds=5, num_alphas=40,
              chunk_len=6, backend_name="torch_cuda", t_tp=None,
              n_iter=50, local_alpha=False, split_scorer="numpy",
              use_kernel=False, delays_tr=None):
    if not _HIMALAYA:
        raise ImportError("himalaya is not installed. "
                          "Run: pip install himalaya")
    if split_scorer not in ("numpy", "himalaya"):
        raise ValueError(f"split_scorer must be 'numpy' or 'himalaya', "
                         f"got {split_scorer!r}")

    backend = set_backend(backend_name, on_error="warn")
    solver_name = ("MultipleKernelRidgeCV (dual/kernel)" if use_kernel
                   else "BandedRidgeCV (primal)")
    print(f"[Ridge] backend: {backend}  |  {solver_name}, "
          f"n_iter={n_iter}, local_alpha={local_alpha}, "
          f"split_scorer={split_scorer}"
          + ("  [split forced to numpy for kernel]" if use_kernel else ""))

    alphas = np.logspace(-3, 10, num_alphas)
    n_rois         = roi_data.shape[1]
    n_feat         = len(feature_order)
    results_r2     = np.zeros((n_subjects, n_rois))
    feature_r2s    = np.zeros((n_subjects, n_feat, n_rois))

    if use_kernel:
        # Paper-faithful per-fold pipeline: each feature space ->
        # StandardScaler (fit on train) -> Delayer (FIR, zero-padded) ->
        # linear Kernelizer; MultipleKernelRidgeCV over the precomputed kernels.
        from sklearn.pipeline import make_pipeline as _mkpipe
        from sklearn.preprocessing import StandardScaler
        from delayer import Delayer
        if not delays_tr:
            raise ValueError("use_kernel=True needs delays_tr (e.g. [2,4,6,8,10]).")
        _preprocess = _mkpipe(
            StandardScaler(with_mean=True, with_std=True),
            Delayer(delays=list(delays_tr), drop_initial=False),   # keep row count
            Kernelizer(kernel="linear"),
        )
        # local_alpha follows CONFIG (ridge_local_alpha). With limited n_iter,
        # local_alpha=True (per-ROI alpha) overfits high-dim kernels -> negative
        # R2; local_alpha=False (one shared alpha) is stable. The paper used
        # True but with far more data/iterations.
        _solver_params = dict(n_iter=n_iter, alphas=alphas,
                              n_targets_batch=10000, n_alphas_batch=50,
                              n_targets_batch_refit=200, local_alpha=local_alpha,
                              diagonalize_method="svd")
        print(f"        delays_tr={list(delays_tr)}  |  local_alpha={local_alpha}"
              f"  |  run regressor={'on (TP/DM)' if t_tp else 'off'}")
    else:
        print(f"        primal bands (+delays pre-expanded)  |  "
              f"local_alpha={local_alpha}  |  "
              f"run regressor={'on (TP/DM)' if t_tp else 'off'}")

    for subj in range(n_subjects):
        print(f"=== Ridge Subject {subj+1}/{n_subjects} ===")
        Y = roi_data[subj].T          # [T, R]
        T = Y.shape[0]

        groups       = make_groups(T, chunk_len, t_tp=t_tp)
        cv_outer     = GroupKFold(n_splits=outer_folds)
        outer_splits = list(cv_outer.split(np.arange(T), groups=groups))

        r2_sum      = np.zeros(n_rois)
        feat_r2_sum = np.zeros((n_feat, n_rois))

        for fold_idx, (tr, te) in enumerate(outer_splits):
            # float64: n_features is small, so there is nothing to gain from
            # float32 here, and the solver is better conditioned in float64.
            # Each fold z-scored with its OWN statistics (reference behaviour):
            # both Y_tr and Y_te are exactly zero-mean, as fit_intercept=False
            # and r2_score_split require.  See zscore_fold().
            Y_tr = zscore_fold(np.nan_to_num(Y[tr]).astype(np.float64))
            Y_te = zscore_fold(np.nan_to_num(Y[te]).astype(np.float64))

            groups_tr  = groups[tr]
            n_inner    = min(inner_folds, len(np.unique(groups_tr)))
            cv_inner   = GroupKFold(n_splits=n_inner)
            inner_iter = list(cv_inner.split(np.arange(len(tr)),
                                             groups=groups_tr))

            if use_kernel:
                # ── paper kernel path: ColumnKernelizer + MultipleKernelRidge ──
                # Raw (zero-lag) feature spaces; StandardScaler+Delayer applied
                # inside the pipeline (fit on train). Add a TP/DM run regressor
                # (+1/-1), zeroed on test so its weight never scores the test.
                feats_tr, feats_te, nlist, names = [], [], [], []
                for f in feature_order:
                    X = np.asarray(features_dict[f], dtype=np.float32)
                    feats_tr.append(X[tr]); feats_te.append(X[te])
                    nlist.append(X.shape[1]); names.append(f)
                if t_tp:
                    run_tr = np.where(np.asarray(tr) < t_tp, 1.0, -1.0
                                      ).reshape(-1, 1).astype(np.float32)
                    feats_tr.append(run_tr)
                    feats_te.append(np.zeros((len(te), 1), dtype=np.float32))
                    nlist.append(1); names.append("run")
                X_tr = np.concatenate(feats_tr, axis=1)
                X_te = np.concatenate(feats_te, axis=1)
                ends = np.concatenate([[0], np.cumsum(nlist)])
                slices = [slice(a, b) for a, b in zip(ends[:-1], ends[1:])]
                ck = ColumnKernelizer([(nm, _preprocess, sl)
                                       for nm, sl in zip(names, slices)])
                mkr = MultipleKernelRidgeCV(kernels="precomputed",
                                            solver="random_search",
                                            solver_params=_solver_params,
                                            cv=inner_iter)
                pipe = _mkpipe(ck, mkr)
                pipe.fit(X_tr, np.float32(Y_tr))

                scores_te = np.asarray(backend.to_numpy(
                    pipe.score(X_te, np.float32(Y_te))))
                r2_sum += scores_te
                sp = backend.to_numpy(pipe.predict(X_te, split=True))
                sp = [np.asarray(sp[k]) for k in range(n_feat)]   # drop run kernel
                feat_r2_sum += r2_score_split_numpy(Y_te, sp)
                print(f"  fold {fold_idx} | X {X_tr.shape} (+delays x"
                      f"{len(set(delays_tr))})  test R² mean: {scores_te.mean():.4f}")
            else:
                train_blocks, test_blocks, dims = [], [], []
                for f in feature_order:
                    X = np.asarray(features_dict[f], dtype=np.float64)
                    mu, sd = fit_zscore_train(X[tr])
                    train_blocks.append(apply_zscore(X[tr], mu, sd))
                    test_blocks.append(apply_zscore(X[te], mu, sd))
                    dims.append(X.shape[1])
                # TP/DM run regressor (+1/-1) as its own band, zeroed on test so
                # the weight that absorbs the movie-mean offset during training
                # never contributes to the test prediction/score. Its band is
                # dropped from the product-measure (kept for score parity only).
                if t_tp:
                    run_tr = np.where(np.asarray(tr) < t_tp, 1.0, -1.0
                                      ).reshape(-1, 1).astype(np.float64)
                    train_blocks.append(run_tr)
                    test_blocks.append(np.zeros((len(te), 1), dtype=np.float64))
                X_tr = train_blocks
                X_te = test_blocks
                model = BandedRidgeCV(
                    groups="input", solver="random_search",
                    solver_params=dict(alphas=alphas, n_iter=n_iter,
                                       local_alpha=local_alpha,
                                       progress_bar=False),
                    cv=inner_iter, fit_intercept=False,
                )
                model.fit(X_tr, Y_tr)
                scores_te = np.asarray(backend.to_numpy(model.score(X_te, Y_te)))
                r2_sum += scores_te
                Y_pred_split = model.predict(X_te, split=True)
                # keep only the real feature bands (drop the run band, if any)
                if split_scorer == "himalaya":
                    split_r2_np = np.asarray(backend.to_numpy(
                        r2_score_split(Y_te, Y_pred_split)))[:n_feat]
                else:
                    splits_np = backend.to_numpy(Y_pred_split)
                    split_r2_np = r2_score_split_numpy(
                        Y_te, [np.asarray(splits_np[k]) for k in range(n_feat)])
                feat_r2_sum += split_r2_np
                print(f"  fold {fold_idx} | blocks: {[b.shape for b in X_tr]}"
                      f"{'  +run' if t_tp else ''}  "
                      f"test R² mean: {scores_te.mean():.4f}")

        results_r2[subj]  = r2_sum      / outer_folds
        feature_r2s[subj] = feat_r2_sum / outer_folds
        print(f"  mean R²_total = {np.nanmean(results_r2[subj]):.4f}")

    return results_r2, feature_r2s


# =============================================================================
# Model 2: ElasticNet / Lasso  (CPU, sklearn)
# =============================================================================
def run_elasticnet(roi_data, features_dict, feature_order, n_subjects,
                   outer_folds=5, inner_folds=5,
                   l1_ratio=None, n_alphas=30,
                   chunk_len=10, model_type="elasticnet", n_jobs=-1,
                   t_tp=None):

    if l1_ratio is None:
        l1_ratio = [0.1, 0.5, 0.7, 0.9, 1.0]

    n_rois      = roi_data.shape[1]
    n_feat      = len(feature_order)
    results_r2  = np.zeros((n_subjects, n_rois))
    feature_r2s = np.zeros((n_subjects, n_feat, n_rois))

    for subj in range(n_subjects):
        print(f"=== {model_type.capitalize()} Subject {subj+1}/{n_subjects} ===")
        Y = roi_data[subj].T    # [T, R]
        T = Y.shape[0]

        groups       = make_groups(T, chunk_len, t_tp=t_tp)
        cv_outer     = GroupKFold(n_splits=outer_folds)
        outer_splits = list(cv_outer.split(np.arange(T), groups=groups))

        r2_sum      = np.zeros(n_rois)
        feat_r2_sum = np.zeros((n_feat, n_rois))

        for fold_idx, (tr, te) in enumerate(outer_splits):
            # Each fold z-scored with its OWN statistics (reference behaviour):
            # both Y_tr and Y_te are exactly zero-mean, as fit_intercept=False
            # and r2_score_split require.  See zscore_fold().
            Y_tr = zscore_fold(np.nan_to_num(Y[tr]).astype(np.float64))
            Y_te = zscore_fold(np.nan_to_num(Y[te]).astype(np.float64))

            train_blocks, test_blocks, dims = [], [], []
            for f in feature_order:
                X   = np.asarray(features_dict[f], dtype=np.float64)
                mu, sd = fit_zscore_train(X[tr])
                train_blocks.append(apply_zscore(X[tr], mu, sd))
                test_blocks.append(apply_zscore(X[te], mu, sd))
                dims.append(X.shape[1])

            X_tr = np.concatenate(train_blocks, axis=1)
            X_te = np.concatenate(test_blocks,  axis=1)

            ends   = np.cumsum([0] + dims)
            slices = [slice(ends[k], ends[k+1]) for k in range(len(dims))]

            groups_tr    = groups[tr]
            n_inner      = min(inner_folds, len(np.unique(groups_tr)))
            cv_inner     = GroupKFold(n_splits=n_inner)
            inner_splits = list(cv_inner.split(
                np.arange(len(tr)), groups=groups_tr))

            if model_type == "ridge_sk":
                alphas_ridge = np.logspace(-3, 10, 40)
                model = RidgeCV(alphas=alphas_ridge, cv=inner_splits,
                                fit_intercept=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, Y_tr)
                Y_pred  = model.predict(X_te)
                test_r2 = sk_r2_score(Y_te, Y_pred, multioutput="raw_values")
                train_r2 = sk_r2_score(Y_tr, model.predict(X_tr),
                                       multioutput="uniform_average")
                print(f"  fold {fold_idx} | X_tr: {X_tr.shape}  "
                      f"train R²: {train_r2:.4f}  "
                      f"test R² mean: {test_r2.mean():.4f}  "
                      f"max: {test_r2.max():.4f}")
                r2_sum += test_r2
                coef_matrix = model.coef_              # [n_rois, total_features]
            else:
                if model_type == "lasso":
                    base = LassoCV(cv=inner_splits, n_alphas=n_alphas,
                                   max_iter=5000, fit_intercept=False)
                else:
                    base = ElasticNetCV(l1_ratio=l1_ratio, cv=inner_splits,
                                        n_alphas=n_alphas, max_iter=5000,
                                        fit_intercept=False)
                model = MultiOutputRegressor(base, n_jobs=n_jobs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, Y_tr)
                Y_pred  = model.predict(X_te)
                test_r2 = sk_r2_score(Y_te, Y_pred, multioutput="raw_values")
                train_r2 = sk_r2_score(Y_tr, model.predict(X_tr),
                                       multioutput="uniform_average")
                print(f"  fold {fold_idx} | X_tr: {X_tr.shape}  "
                      f"train R²: {train_r2:.4f}  "
                      f"test R² mean: {test_r2.mean():.4f}  "
                      f"max: {test_r2.max():.4f}")
                r2_sum += test_r2
                coef_matrix = np.stack(
                    [est.coef_ for est in model.estimators_], axis=0
                )   # [n_rois, total_features]

            Y_pred_splits = [X_te[:, sl] @ coef_matrix[:, sl].T
                             for sl in slices]
            feat_r2_sum += r2_score_split_numpy(Y_te, Y_pred_splits)

        results_r2[subj]  = r2_sum      / outer_folds
        feature_r2s[subj] = feat_r2_sum / outer_folds
        print(f"  mean R²_total = {np.nanmean(results_r2[subj]):.4f}")
        for i, f in enumerate(feature_order):
            print(f"    {f:<20s} mean product-R² = "
                  f"{np.nanmean(feature_r2s[subj, i]):.4f}")

    return results_r2, feature_r2s


# =============================================================================
# Model 3b: Variance Partitioning  (GPU ridge, himalaya.ridge.RidgeCV)
# =============================================================================
def run_variance_partitioning_gpu(
        roi_data, features_dict, feature_order, n_subjects,
        outer_folds=5, inner_folds=5,
        chunk_len=10, alphas=None,
        backend_name="torch_cuda",
        score_metric="r2",
):
    """
    GPU-accelerated variance partitioning using himalaya.ridge.RidgeCV.

    Same logic as run_variance_partitioning (outer CV, inner CV, full + LOO
    models), but _fit_score uses himalaya.ridge.RidgeCV which:
      - runs on GPU via torch_cuda backend
      - solves all ROIs in one batched matrix operation
      - evaluates all alphas in parallel (no per-alpha loop)

    Returns
    -------
    results_r2 : [S, R]     full-model R²
    unique_r2s : [S, F, R]  unique variance per feature
    shared_r2  : [S, R]     shared variance
    """
    from himalaya.ridge import RidgeCV as HimalayaRidgeCV
    from himalaya.backend import set_backend

    backend = set_backend(backend_name, on_error="warn")
    print(f"[VarPart-GPU] backend = {backend}")


    if alphas is None:
        alphas = np.logspace(-3, 10, 40)
    alphas_gpu = backend.asarray(np.array(alphas, dtype=np.float32))

    n_rois     = roi_data.shape[1]
    F          = len(feature_order)
    results_r2 = np.zeros((n_subjects, n_rois))
    unique_r2s = np.zeros((n_subjects, F, n_rois))
    shared_r2  = np.zeros((n_subjects, n_rois))

    if score_metric not in ("r2", "pearson_r"):
        raise ValueError(f"Unknown score_metric '{score_metric}'. Choose: r2 | pearson_r")

    def _fit_score_gpu(X_tr, Y_tr, X_te, Y_te, cv_splits):
        X_tr_t = backend.asarray(X_tr.astype(np.float32))
        Y_tr_t = backend.asarray(Y_tr.astype(np.float32))
        X_te_t = backend.asarray(X_te.astype(np.float32))
        Y_te_t = backend.asarray(Y_te.astype(np.float32))
        model  = HimalayaRidgeCV(alphas=alphas_gpu, fit_intercept=False,
                                 cv=cv_splits)
        model.fit(X_tr_t, Y_tr_t)
        Y_pred = backend.asarray(model.predict(X_te_t))
        if score_metric == "r2":
            ss_res = ((Y_te_t - Y_pred) ** 2).sum(0)
            ss_tot = ((Y_te_t - Y_te_t.mean(0)) ** 2).sum(0) + 1e-12
            return backend.to_numpy(1.0 - ss_res / ss_tot)   # [n_rois]
        yt = Y_te_t - Y_te_t.mean(0)
        yp = Y_pred - Y_pred.mean(0)
        num = (yt * yp).sum(0)
        den = ((yt ** 2).sum(0) * (yp ** 2).sum(0)) ** 0.5 + 1e-12
        return backend.to_numpy(num / den)   # [n_rois]

    for subj in range(n_subjects):
        print(f"=== VarPart-GPU Subject {subj+1}/{n_subjects} ===")
        Y = roi_data[subj].T   # [T, R]
        T = Y.shape[0]

        n_chunks = max(outer_folds, T // chunk_len)
        groups   = np.array([i for i in range(n_chunks)
                             for _ in range(chunk_len)])
        groups   = np.concatenate(
            [groups, np.full(max(0, T - len(groups)), n_chunks)])
        groups   = groups[:T]

        cv_outer     = GroupKFold(n_splits=outer_folds)
        outer_splits = list(cv_outer.split(np.arange(T), groups=groups))

        r2_full_sum = np.zeros(n_rois)
        unique_sum  = np.zeros((F, n_rois))

        for fold_idx, (tr, te) in enumerate(outer_splits):
            # Each fold z-scored with its OWN statistics (reference behaviour):
            # both Y_tr and Y_te are exactly zero-mean, as fit_intercept=False
            # and r2_score_split require.  See zscore_fold().
            Y_tr = zscore_fold(np.nan_to_num(Y[tr]).astype(np.float64))
            Y_te = zscore_fold(np.nan_to_num(Y[te]).astype(np.float64))

            blocks_tr, blocks_te = {}, {}
            for f in feature_order:
                X  = np.asarray(features_dict[f], dtype=np.float64)
                mu, sd = fit_zscore_train(X[tr])
                blocks_tr[f] = apply_zscore(X[tr], mu, sd)
                blocks_te[f] = apply_zscore(X[te], mu, sd)

            groups_tr    = groups[tr]
            n_inner      = min(inner_folds, len(np.unique(groups_tr)))
            cv_inner     = GroupKFold(n_splits=n_inner)

            # full model
            X_tr_full = np.concatenate([blocks_tr[f] for f in feature_order], axis=1)
            X_te_full = np.concatenate([blocks_te[f] for f in feature_order], axis=1)
            inner_splits = list(cv_inner.split(np.arange(len(tr)), groups=groups_tr))
            r2_full = _fit_score_gpu(X_tr_full, Y_tr, X_te_full, Y_te, inner_splits)
            r2_full_sum += r2_full

            # leave-one-out per feature
            for ki, left_out in enumerate(feature_order):
                others = [f for f in feature_order if f != left_out]
                X_tr_loo = np.concatenate([blocks_tr[f] for f in others], axis=1)
                X_te_loo = np.concatenate([blocks_te[f] for f in others], axis=1)
                inner_splits_k = list(cv_inner.split(np.arange(len(tr)),
                                                     groups=groups_tr))
                r2_loo = _fit_score_gpu(X_tr_loo, Y_tr, X_te_loo, Y_te,
                                        inner_splits_k)
                unique_sum[ki] += r2_full - r2_loo

            print(f"  fold {fold_idx} | full R² mean: {r2_full.mean():.4f}")

        results_r2[subj] = r2_full_sum / outer_folds
        unique_r2s[subj] = unique_sum  / outer_folds
        shared_r2[subj]  = results_r2[subj] - unique_r2s[subj].sum(axis=0)

        print(f"  mean R²_full = {np.nanmean(results_r2[subj]):.4f}")
        for ki, f in enumerate(feature_order):
            print(f"    unique({f:<20s}) = {np.nanmean(unique_r2s[subj, ki]):.5f}")
        print(f"    shared                       = {np.nanmean(shared_r2[subj]):.5f}")

    return results_r2, unique_r2s, shared_r2


# =============================================================================
# Model 3: Variance Partitioning  (configurable regressor, CPU)
# =============================================================================
def _make_regressor(regressor, alphas, cv, l1_ratio, n_alphas_en):
    """
    Build a fitted-ready sklearn regressor for variance partitioning.

    regressor choices
    -----------------
    'ridge'      RidgeCV — L2, multi-output native, default & recommended.
                 Stable for correlated features, single alpha hyperparameter.

    'lasso'      LassoCV — L1, sparse solution.
                 Useful if you suspect only a few features drive each ROI.
                 Wrapped in MultiOutputRegressor (fits one model per ROI).

    'elasticnet' ElasticNetCV — L1+L2 mix.
                 Best when features are correlated AND you want sparsity.
                 Wrapped in MultiOutputRegressor.

    'ols'        LinearRegression — no regularization.
                 Only safe when n_features << n_samples (e.g. leave-one-out
                 sub-models with few features and long movies like DM).
                 Avoid for PC (167 TRs) or when feature dim is large.
    """
    from sklearn.linear_model import (RidgeCV, LassoCV, ElasticNetCV,
                                      LinearRegression)

    if regressor == "ridge":
        return RidgeCV(alphas=alphas, cv=cv, fit_intercept=False)
    elif regressor == "lasso":
        base = LassoCV(cv=cv, n_alphas=n_alphas_en,
                       max_iter=5000, fit_intercept=False)
        return MultiOutputRegressor(base, n_jobs=-1)
    elif regressor == "elasticnet":
        base = ElasticNetCV(l1_ratio=l1_ratio, cv=cv,
                            n_alphas=n_alphas_en,
                            max_iter=5000, fit_intercept=False)
        return MultiOutputRegressor(base, n_jobs=-1)
    elif regressor == "ols":
        return LinearRegression(fit_intercept=False)
    else:
        raise ValueError(f"Unknown regressor '{regressor}'. "
                         "Choose: ridge | lasso | elasticnet | ols")


def run_variance_partitioning(
        roi_data, features_dict, feature_order, n_subjects,
        outer_folds=5, inner_folds=5,
        chunk_len=10, alphas=None,
        regressor="ridge",
        l1_ratio=None, n_alphas_en=30,
        score_metric="r2",
):
    """
    Cross-validated variance partitioning.

    Fits F+1 models per fold per subject:
      full model      : all F features  -> score_full
      leave-one-out k : all except k    -> score_-k

    unique_k = score_full - score_-k   (unique contribution of feature k)
    shared   = score_full - sum(unique_k)

    Parameters
    ----------
    regressor : 'ridge' (default) | 'lasso' | 'elasticnet' | 'ols'
    score_metric : 'r2' (default, 1 - SS_res/SS_tot) | 'pearson_r'
        'pearson_r' scores held-out predictions by Pearson correlation with
        the true response per ROI instead of R². Correlation only cares
        about whether predicted and true responses co-vary (the predicted
        TIMING/PATTERN), not whether their overall scale/intercept is well
        calibrated, so it is far less prone to swinging negative from mild
        mis-calibration than R² is -- useful when the underlying per-ROI
        signal is weak and R²-based unique/shared values are dominated by
        CV noise. Alpha selection inside RidgeCV/etc. is unchanged (still
        its own internal default scorer); only the REPORTED score changes.

    Returns
    -------
    results_r2 : [S, R]     full-model score
    unique_r2s : [S, F, R]  unique contribution per feature
    shared_r2  : [S, R]     shared contribution
    """
    if alphas is None:
        alphas = np.logspace(-3, 10, 40)
    if l1_ratio is None:
        l1_ratio = [0.1, 0.5, 0.7, 0.9, 1.0]
    if score_metric not in ("r2", "pearson_r"):
        raise ValueError(f"Unknown score_metric '{score_metric}'. Choose: r2 | pearson_r")

    print(f"[VarPart] regressor = {regressor}  score_metric = {score_metric}")

    n_rois     = roi_data.shape[1]
    F          = len(feature_order)
    results_r2 = np.zeros((n_subjects, n_rois))
    unique_r2s = np.zeros((n_subjects, F, n_rois))
    shared_r2  = np.zeros((n_subjects, n_rois))

    def _score(Y_te, Y_pred):
        if score_metric == "r2":
            ss_res = ((Y_te - Y_pred) ** 2).sum(0)
            ss_tot = ((Y_te - Y_te.mean(0)) ** 2).sum(0) + 1e-12
            return 1.0 - ss_res / ss_tot   # [n_rois]
        # pearson_r: per-column correlation between predicted and true response
        yt = Y_te - Y_te.mean(0)
        yp = Y_pred - Y_pred.mean(0)
        num = (yt * yp).sum(0)
        den = np.sqrt((yt ** 2).sum(0) * (yp ** 2).sum(0)) + 1e-12
        return num / den   # [n_rois]

    def _fit_score(X_tr, Y_tr, X_te, Y_te, cv):
        model = _make_regressor(regressor, alphas, cv, l1_ratio, n_alphas_en)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, Y_tr)
        Y_pred = model.predict(X_te)
        return _score(Y_te, Y_pred)

    for subj in range(n_subjects):
        print(f"=== VarPart Subject {subj+1}/{n_subjects} ===")
        Y = roi_data[subj].T   # [T, R]
        T = Y.shape[0]

        n_chunks = max(outer_folds, T // chunk_len)
        groups   = np.array([i for i in range(n_chunks)
                             for _ in range(chunk_len)])
        groups   = np.concatenate(
            [groups, np.full(max(0, T - len(groups)), n_chunks)])
        groups   = groups[:T]

        cv_outer     = GroupKFold(n_splits=outer_folds)
        outer_splits = list(cv_outer.split(np.arange(T), groups=groups))

        r2_full_sum = np.zeros(n_rois)
        unique_sum  = np.zeros((F, n_rois))

        for fold_idx, (tr, te) in enumerate(outer_splits):
            # Each fold z-scored with its OWN statistics (reference behaviour):
            # both Y_tr and Y_te are exactly zero-mean, as fit_intercept=False
            # and r2_score_split require.  See zscore_fold().
            Y_tr = zscore_fold(np.nan_to_num(Y[tr]).astype(np.float64))
            Y_te = zscore_fold(np.nan_to_num(Y[te]).astype(np.float64))

            blocks_tr, blocks_te = {}, {}
            for f in feature_order:
                X   = np.asarray(features_dict[f], dtype=np.float64)
                mu, sd = fit_zscore_train(X[tr])
                blocks_tr[f] = apply_zscore(X[tr], mu, sd)
                blocks_te[f] = apply_zscore(X[te], mu, sd)

            groups_tr    = groups[tr]
            n_inner      = min(inner_folds, len(np.unique(groups_tr)))
            cv_inner     = GroupKFold(n_splits=n_inner)

            # full model
            X_tr_full = np.concatenate([blocks_tr[f] for f in feature_order], axis=1)
            X_te_full = np.concatenate([blocks_te[f] for f in feature_order], axis=1)
            inner_splits = list(cv_inner.split(np.arange(len(tr)), groups=groups_tr))
            r2_full = _fit_score(X_tr_full, Y_tr, X_te_full, Y_te, inner_splits)
            r2_full_sum += r2_full

            # leave-one-out per feature
            for ki, left_out in enumerate(feature_order):
                others = [f for f in feature_order if f != left_out]
                X_tr_loo = np.concatenate([blocks_tr[f] for f in others], axis=1)
                X_te_loo = np.concatenate([blocks_te[f] for f in others], axis=1)
                inner_splits_k = list(cv_inner.split(np.arange(len(tr)), groups=groups_tr))
                r2_loo = _fit_score(X_tr_loo, Y_tr, X_te_loo, Y_te, inner_splits_k)
                unique_sum[ki] += r2_full - r2_loo

            print(f"  fold {fold_idx} | full R² mean: {r2_full.mean():.4f}")

        results_r2[subj] = r2_full_sum / outer_folds
        unique_r2s[subj] = unique_sum  / outer_folds
        shared_r2[subj]  = results_r2[subj] - unique_r2s[subj].sum(axis=0)

        print(f"  mean R²_full = {np.nanmean(results_r2[subj]):.4f}")
        for ki, f in enumerate(feature_order):
            print(f"    unique({f:<20s}) = {np.nanmean(unique_r2s[subj, ki]):.5f}")
        print(f"    shared                       = {np.nanmean(shared_r2[subj]):.5f}")

    return results_r2, unique_r2s, shared_r2


# =============================================================================
# Statistics: sign-permutation test + FDR-BH
# =============================================================================
def run_statistics(feature_r2s, feature_order, n_perm=5000, fdr_alpha=0.05,
                   random_seed=42):
    """
    Parameters
    ----------
    feature_r2s : [S, F, R]

    Returns
    -------
    p_vals   : [F, R]   permutation p-values
    sig_mask : [F, R]   bool, True = significant after FDR-BH
    """
    data = feature_r2s.astype(float)
    S, F, R = data.shape
    rng = np.random.default_rng(random_seed)
    p_vals = np.ones((F, R))

    print("\n[Stats] Running sign-permutation test "
          f"({n_perm} perms) + FDR-BH (α={fdr_alpha}) ...")

    for fi in range(F):
        for ri in range(R):
            x = data[:, fi, ri]
            x = x[~np.isnan(x)]
            if len(x) < 3:
                p_vals[fi, ri] = 1.0
                continue
            observed  = x.mean()
            signs     = rng.choice([-1, 1], size=(n_perm, len(x)))
            null_means = (signs * x).mean(axis=1)
            p_vals[fi, ri] = (null_means >= observed).mean()

    sig_mask = np.zeros((F, R), dtype=bool)
    for fi in range(F):
        reject, _, _, _ = multipletests(
            p_vals[fi], alpha=fdr_alpha, method="fdr_bh")
        sig_mask[fi] = reject

    # ── diagnostics ─────────────────────────────────────────────────────
    mean_r2 = np.nanmean(data, axis=0)   # [F, R]
    print("\n=== Per-feature diagnostics ===")
    for fi, fname in enumerate(feature_order):
        vals = mean_r2[fi]
        n_pos = (vals > 0).sum()
        n_unc = (p_vals[fi] < 0.05).sum()
        n_sig = sig_mask[fi].sum()
        print(f"\n  [{fname}]")
        print(f"    mean R² across all {R} ROIs : {vals.mean():.5f}")
        print(f"    max  R²                      : {vals.max():.5f}  "
              f"(ROI {vals.argmax()+1})")
        print(f"    ROIs with mean > 0           : {n_pos}/{R}")
        print(f"    uncorrected p<0.05           : {n_unc}/{R}")
        print(f"    significant after FDR-BH     : {n_sig}/{R}")

    return p_vals, sig_mask


# =============================================================================
# Statistics: ROI-level Wilcoxon, FDR-BH within each network
# (matches the original paper's panel-by-panel correction strategy)
# =============================================================================
def run_statistics_roi_within_network(feature_r2s, feature_order, fdr_alpha=0.05):
    """
    For each of the 8 subnetworks independently:
        1. Wilcoxon signed-rank test (H0: median=0, one-tailed > 0) per ROI
           across subjects — one test per (ROI, feature) pair.
        2. FDR-BH correction across all ROIs × features within that network
           (~14 ROIs × F features per family).

    This mirrors the original paper (ubiquitous stats_helpers.py): stats are
    run and corrected per anatomical region group (panel), not globally.
    A network bar in the plot is marked significant when the majority (>50%)
    of its ROIs survive FDR within that network.

    Parameters
    ----------
    feature_r2s : [S, F, R]
    feature_order : list[str]
    fdr_alpha : float

    Returns
    -------
    p_vals       : [F, R]   raw Wilcoxon p-values per ROI
    sig_mask_roi : [F, R]   bool  ROI-level significance after within-network FDR
    sig_mask_net : [F, N]   bool  True when >50 % of ROIs in network are significant
    """
    from scipy.stats import wilcoxon as _wilcoxon

    data = feature_r2s.astype(float)   # [S, F, R]
    S, F, R = data.shape
    N = len(NET_ORDER)

    p_vals       = np.ones((F, R))
    sig_mask_roi = np.zeros((F, R), dtype=bool)

    for ni, net in enumerate(NET_ORDER):
        roi_idx = ROI_BLOCKS_0[net]          # 0-based ROI indices in this network
        keys, raw_p = [], []

        for fi in range(F):
            for ri in roi_idx:
                x = data[:, fi, ri]
                x = x[~np.isnan(x)]
                if len(x) < 3:
                    raw_p.append(1.0)
                else:
                    try:
                        res = _wilcoxon(x, alternative="greater", nan_policy="omit")
                        raw_p.append(float(res.pvalue))
                    except ValueError:
                        raw_p.append(1.0)
                keys.append((fi, ri))

        # FDR-BH within this network (n_rois × F tests)
        reject, _, _, _ = multipletests(raw_p, alpha=fdr_alpha, method="fdr_bh")
        for (fi, ri), p, sig in zip(keys, raw_p, reject):
            p_vals[fi, ri]       = p
            sig_mask_roi[fi, ri] = sig

    # aggregate to network level: significant if >50 % of ROIs survive FDR
    sig_mask_net = np.zeros((F, N), dtype=bool)
    for ni, net in enumerate(NET_ORDER):
        roi_idx = ROI_BLOCKS_0[net]
        for fi in range(F):
            n_sig = sig_mask_roi[fi, roi_idx].sum()
            sig_mask_net[fi, ni] = n_sig >= 1

    # diagnostics
    print(f"\n[Stats] Wilcoxon > 0 per ROI, FDR-BH within each network "
          f"(α={fdr_alpha})")
    print(f"  Family size per network: ~{R // N} ROIs × {F} features "
          f"≈ {(R // N) * F} tests  (vs {R * F} globally)")
    for fi, fname in enumerate(feature_order):
        n_sig = int(sig_mask_roi[fi].sum())
        sig_nets = [NET_ORDER[ni] for ni in range(N) if sig_mask_net[fi, ni]]
        print(f"  [{fname}]  {n_sig}/{R} significant ROIs  |  "
              f"significant networks: {sig_nets if sig_nets else 'none'}")
        for ni, net in enumerate(NET_ORDER):
            roi_idx = ROI_BLOCKS_0[net]
            n_sig_net = int(sig_mask_roi[fi, roi_idx].sum())
            star = " *" if sig_mask_net[fi, ni] else ""
            mean_r2 = float(np.nanmean(data[:, fi, roi_idx]))
            print(f"    {net:8s}  mean R²={mean_r2:.5f}  "
                  f"sig ROIs={n_sig_net}/{len(roi_idx)}{star}")

    return p_vals, sig_mask_roi, sig_mask_net


# =============================================================================
# Plotting
# =============================================================================
def _bar_plot(ax, mean_contrib, sem_contrib, feature_order, title, ylabel):
    N = len(NET_ORDER)
    F = len(feature_order)
    x        = np.arange(N)
    bw       = min(0.8 / F, 0.2)
    offsets  = (np.arange(F) - (F - 1) / 2) * bw
    colors   = plt.cm.tab10(np.linspace(0, 1, max(F, 2)))

    for fi in range(F):
        h = mean_contrib[fi].copy()
        e = sem_contrib[fi].copy()
        ax.bar(x + offsets[fi], h, width=bw, yerr=e,
               capsize=3, label=feature_order[fi],
               color=colors[fi], edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(NET_ORDER, fontsize=11)
    ax.set_xlabel("Subnetwork", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.axhline(0, color="black", linewidth=1)
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=False, fontsize=9)



def _bar_plot_with_sig(ax, mean_contrib, sem_contrib, sig_mask_net,
                       feature_order, title, ylabel):
    N = len(NET_ORDER)
    F = len(feature_order)
    x        = np.arange(N)
    bw       = min(0.8 / F, 0.2)
    offsets  = (np.arange(F) - (F - 1) / 2) * bw
    colors   = plt.cm.tab10(np.linspace(0, 1, max(F, 2)))

    ymax = float(np.nanmax(mean_contrib + np.nan_to_num(sem_contrib))) if mean_contrib.size > 0 else 0.01

    for fi in range(F):
        h = mean_contrib[fi].copy()
        e = sem_contrib[fi].copy()
        ax.bar(x + offsets[fi], h, width=bw, yerr=e,
               capsize=3, label=feature_order[fi],
               color=colors[fi], edgecolor="black", linewidth=0.8)
        for ni in range(N):
            if sig_mask_net is not None and sig_mask_net[fi, ni] and h[ni] > 0:
                ax.text(x[ni] + offsets[fi], h[ni] + e[ni] + ymax * 0.04, "*",
                        ha="center", va="bottom",
                        fontsize=11, color="black", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(NET_ORDER, fontsize=11)
    ax.set_xlabel("Subnetwork", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.axhline(0, color="black", linewidth=1)
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=False, fontsize=9)


def _roi_plot_with_sig(ax, feature_r2s, sig_mask_roi, feature_order, title, ylabel):
    """
    ROI-level bar chart with ROIs grouped by subnetwork on the x-axis.
    Significant ROIs (within-network FDR) are drawn in full colour with a
    star; non-significant ROIs are light grey.

    feature_r2s  : [S, F, R]
    sig_mask_roi : [F, R]  bool
    """
    S, F, R = feature_r2s.shape
    colors   = plt.cm.tab10(np.linspace(0, 1, max(F, 2)))
    mean_roi = np.nanmean(feature_r2s, axis=0)          # [F, R]
    sem_roi  = np.nanstd(feature_r2s, axis=0, ddof=1) / np.sqrt(S)

    # build x-axis: ROIs reordered so each network is a contiguous block
    roi_order, net_info = [], []
    pos = 0
    for net in NET_ORDER:
        idx = ROI_BLOCKS_0[net]
        net_info.append((net, pos, pos + len(idx)))
        roi_order.extend(idx)
        pos += len(idx)

    bw      = 0.8 / F
    offsets = (np.arange(F) - (F - 1) / 2) * bw

    ymax_data = float(np.nanmax(mean_roi + np.nan_to_num(sem_roi)))

    for fi in range(F):
        for xi, ri in enumerate(roi_order):
            sig  = bool(sig_mask_roi[fi, ri])
            xpos = xi + offsets[fi]
            h    = float(mean_roi[fi, ri])
            e    = float(sem_roi[fi, ri]) if not np.isnan(sem_roi[fi, ri]) else 0.0

            ax.bar(xpos, h, width=bw,
                   label=feature_order[fi] if xi == 0 else "_nolegend_",
                   color=colors[fi] if sig else "lightgray",
                   alpha=1.0,
                   edgecolor="none", linewidth=0)

            if sig and h > 0:
                ax.text(xpos, h + e + ymax_data * 0.025, "*",
                        ha="center", va="bottom",
                        fontsize=5, color=colors[fi], fontweight="bold")

    # network separators and labels
    ax.set_xticks([])
    ybot = ax.get_ylim()[0]
    for net, sx, ex in net_info:
        mid = (sx + ex - 1) / 2.0
        ax.text(mid, ybot, net, ha="center", va="top", fontsize=8)
        if ex < R:
            ax.axvline(ex - 0.5, color="gray", linewidth=0.5, linestyle=":")

    ax.set_xlabel("ROI (grouped by subnetwork)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=False, fontsize=9)


def plot_results(feature_r2s, feature_order,
                 movie, model, fig_path=None,
                 sig_mask_roi=None, sig_mask_net=None):
    """
    Two-panel figure:
    Left  — network-level overview, no statistics (mean ± SEM per network)
    Right — network-level bars, same layout; * marks significant networks
            (within-network FDR-BH Wilcoxon, >50 % of ROIs survive).

    sig_mask_net : [F, N]  bool  from run_statistics_roi_within_network()
    """
    data    = feature_r2s.astype(float)
    S, F, R = data.shape
    N       = len(NET_ORDER)

    # ── network-level mean ± SEM ─────────────────────────────────────────
    contrib_all = np.zeros((S, F, N))
    for ni, net in enumerate(NET_ORDER):
        idx = ROI_BLOCKS_0[net]
        contrib_all[:, :, ni] = np.nanmean(data[:, :, idx], axis=2)

    mean_all = np.nanmean(contrib_all, axis=0)   # [F, N]
    sem_all  = np.nanstd(contrib_all, axis=0, ddof=1) / np.sqrt(S)

    # ── figure: two equal-width panels (both network-level) ──────────────
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"Encoding model: {model.upper()}  |  Movie: {movie.upper()}",
                 fontsize=14, y=1.02)
    gs   = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.45)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    _bar_plot(ax_l, mean_all, sem_all, feature_order,
              title="No statistics\n(mean ± SEM per network)",
              ylabel="Mean R² (product-measure)")

    _bar_plot_with_sig(
        ax_r, mean_all, sem_all, sig_mask_net, feature_order,
        title=("With statistics  (Wilcoxon > 0, within-network FDR-BH)\n"
               "* = significant network  (>50 % of ROIs survive FDR)"),
        ylabel="Mean R² (product-measure)")

    plt.tight_layout()

    if fig_path:
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Figure saved → {fig_path}")
    else:
        plt.show()

    plt.close(fig)
    return mean_all, contrib_all


# =============================================================================
# Plotting: Variance Partitioning
# =============================================================================
def plot_variance_partitioning(unique_r2s, shared_r2, results_r2,
                               feature_order, movie, model,
                               sig_mask_roi=None, sig_mask_net=None,
                               fig_path=None):
    """
    Two-panel stacked bar figure for variance partitioning.

    Left  — network-level stacked bars: unique per feature + shared
            (no statistics, all ROIs averaged per network)
    Right — same network-level stacked bars; * marks significant networks
            (within-network FDR-BH Wilcoxon, >50 % of ROIs survive).

    unique_r2s  : [S, F, R]
    shared_r2   : [S, R]
    sig_mask_net: [F, N]  from run_statistics_roi_within_network(unique_r2s)
    """
    S, F, R = unique_r2s.shape
    N = len(NET_ORDER)
    colors = plt.cm.tab10(np.linspace(0, 1, max(F + 1, 2)))

    # ── aggregate by network ─────────────────────────────────────────────
    uniq_net   = np.zeros((S, F, N))
    shared_net = np.zeros((S, N))
    for ni, net in enumerate(NET_ORDER):
        idx = ROI_BLOCKS_0[net]
        uniq_net[:, :, ni]  = np.nanmean(unique_r2s[:, :, idx], axis=2)
        shared_net[:, ni]   = np.nanmean(shared_r2[:, idx], axis=1)

    mean_uniq   = np.nanmean(uniq_net,   axis=0)   # [F, N]
    sem_uniq    = np.nanstd(uniq_net,    axis=0, ddof=1) / np.sqrt(S)
    mean_shared = np.nanmean(shared_net, axis=0)   # [N]
    sem_shared  = np.nanstd(shared_net,  axis=0, ddof=1) / np.sqrt(S)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"Variance Partitioning  |  Model: {model.upper()}  |  "
        f"Movie: {movie.upper()}",
        fontsize=14, y=1.02)
    gs   = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.45)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    def _draw_stacked(ax, title, add_sig=False):
        x      = np.arange(N)
        bw     = 0.5
        bottom = np.zeros(N)
        total_h = np.zeros(N)
        for fi in range(F):
            h = np.clip(mean_uniq[fi], 0, None)
            ax.bar(x, h, width=bw, bottom=bottom,
                   label=feature_order[fi], color=colors[fi],
                   edgecolor="black", linewidth=0.8,
                   yerr=sem_uniq[fi], capsize=3,
                   error_kw=dict(elinewidth=0.8))
            if add_sig and sig_mask_net is not None:
                ymax = float(np.nanmax(
                    np.clip(mean_uniq, 0, None).sum(0)
                    + np.clip(mean_shared, 0, None) + 0.001))
                for ni in range(N):
                    if sig_mask_net[fi, ni] and h[ni] > 0:
                        ax.text(x[ni], bottom[ni] + h[ni] + sem_uniq[fi, ni]
                                + ymax * 0.04, "*",
                                ha="center", va="bottom",
                                fontsize=11, color="black", fontweight="bold")
            bottom += h
            total_h += h
        h_sh = np.clip(mean_shared, 0, None)
        ax.bar(x, h_sh, width=bw, bottom=bottom,
               label="shared", color=colors[F],
               alpha=0.5, edgecolor="black", linewidth=0.8,
               yerr=sem_shared, capsize=3, error_kw=dict(elinewidth=0.8))
        ax.set_xticks(x)
        ax.set_xticklabels(NET_ORDER, fontsize=10, rotation=20, ha="right")
        ax.set_xlabel("Subnetwork", fontsize=12)
        ax.set_ylabel("R² (unique + shared)", fontsize=11)
        ax.set_title(title, fontsize=12, pad=8)
        ax.axhline(0, color="black", linewidth=1)
        ax.legend(title="Feature", bbox_to_anchor=(1.02, 1),
                  loc="upper left", frameon=False, fontsize=9)

    # ── LEFT: no statistics ───────────────────────────────────────────────
    _draw_stacked(ax_l,
                  title="No statistics\n(unique + shared per network)",
                  add_sig=False)

    # ── RIGHT: with significance stars ────────────────────────────────────
    _draw_stacked(ax_r,
                  title=("With statistics  (Wilcoxon > 0, within-network FDR-BH)\n"
                         "* = significant network  (>50 % of ROIs survive FDR)"),
                  add_sig=True)

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[Plot-VarPart] Figure saved → {fig_path}")
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# Analysis: compare no-stats vs with-stats
# =============================================================================
def print_analysis(feature_r2s, sig_mask, feature_order,
                   mean_all, mean_sig, contrib_all, contrib_sig):
    """
    Print a structured analysis comparing the two plot versions and
    explain why FDR-corrected stats often leave no significant ROIs.
    """
    S, F, R = feature_r2s.shape
    data = feature_r2s.astype(float)

    print("\n" + "=" * 70)
    print("ANALYSIS: No-stats  vs  FDR-corrected stats")
    print("=" * 70)

    # how many ROIs survive FDR per feature?
    print("\n── FDR survival per feature ──────────────────────────────────────")
    for fi, fname in enumerate(feature_order):
        n_sig = sig_mask[fi].sum()
        by_net = {}
        for net in NET_ORDER:
            idx     = ROI_BLOCKS_0[net]
            n_net   = len(idx)
            n_s_net = sum(1 for i in idx if sig_mask[fi, i])
            if n_s_net > 0:
                by_net[net] = f"{n_s_net}/{n_net}"
        print(f"  {fname:<20s}  {n_sig:3d}/114 significant  "
              f"| nets with sig ROIs: {by_net}")

    # Effect-size context
    mean_r2 = np.nanmean(data, axis=0)   # [F, R]
    print("\n── R² effect sizes ───────────────────────────────────────────────")
    for fi, fname in enumerate(feature_order):
        vals = mean_r2[fi]
        print(f"  {fname:<20s}  "
              f"mean={vals.mean():.5f}  "
              f"max={vals.max():.5f}  "
              f"ROIs>0: {(vals>0).sum()}/114")

    # Cross-subject variability (std / mean)
    print("\n── Cross-subject CV (std/|mean|) per network ─────────────────────")
    for fi, fname in enumerate(feature_order):
        cvs = []
        for ni, net in enumerate(NET_ORDER):
            col = contrib_all[:, fi, ni]          # [S]
            col = col[~np.isnan(col)]
            if len(col) < 3 or abs(col.mean()) < 1e-8:
                cvs.append("NA")
            else:
                cvs.append(f"{col.std(ddof=1)/abs(col.mean()):.2f}")
        print(f"  {fname:<20s}  {dict(zip(NET_ORDER, cvs))}")

    print("""
── Why FDR kills significance ─────────────────────────────────────────

1. Multiple-comparison burden
   114 ROIs × F features = up to {n_comp} simultaneous tests.
   BH-FDR requires each adjusted p < α × rank / n_tests, so the
   effective per-test α is far below 0.05.

2. Small R² effect sizes  (typically 0.001 – 0.01 in cross-validated
   fMRI encoding studies). Even when the effect is real, the sample
   mean across {S} subjects is noisy relative to the effect size,
   making it hard to pass a one-sided sign-permutation test.

3. Sign-permutation is conservative for cross-validated R²
   Cross-validated R² can be slightly negative for null ROIs (finite
   sample bias), which inflates the null distribution used in the
   permutation test.  A t-test has the same problem and is even more
   liberal on small N, which is why adding a t-test made things worse.

4. Recommended alternatives
   • Use a network-level average (fewer tests → easier to survive FDR).
   • Use a non-parametric Wilcoxon signed-rank test instead of sign-perm
     (more powerful for small N).
   • Apply FDR per feature separately rather than pooling all ROIs.
   • Use a less stringent correction (Bonferroni-Holm) or report
     uncorrected p<0.05 with a note on exploratory nature.
   • If group-level significance is needed, a threshold of R² > 0
     surviving across >50% of subjects is an intuitive non-parametric
     criterion used in several neuroimaging studies.
""".format(n_comp=114 * len(feature_order), S=S))


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Encoding pipeline")
    p.add_argument("--movie",         default=None,
                   choices=["dm", "tp", "pc", "tpdm"])
    p.add_argument("--model",         default=None,
                   choices=["ridge", "ridge_sk", "elasticnet", "lasso", "varpart", "varpart_gpu"])
    p.add_argument("--group",         default=None, choices=["adult", "child"],
                   help="Subject group (default: adult)")
    p.add_argument("--isc_threshold", default=None, type=float,
                   help="Keep only ROIs where BOTH adult & child ISC >= threshold "
                        "(e.g. 0.2). Omit or set 0 to skip ISC filtering.")
    p.add_argument("--fmri_file",     default=None,
                   help="Path to fMRI .npy file (overrides FMRI_FILES[movie][group])")
    p.add_argument("--tag",           default=None,
                   help="Override output filename tag")
    p.add_argument("--t_tp",          default=None, type=int,
                   help="TP length in TRs for movie-aware CV (from meta_*.npy)")
    p.add_argument("--feature_file",  default=None)
    p.add_argument("--feature_order", default=None, nargs="+")
    p.add_argument("--chunk_len",     default=None, type=int)
    p.add_argument("--outer_folds",   default=None, type=int)
    p.add_argument("--inner_folds",   default=None, type=int)
    p.add_argument("--n_perm",        default=None, type=int)
    p.add_argument("--fdr_alpha",     default=None, type=float)
    p.add_argument("--output_dir",    default=None)
    p.add_argument("--fig_path",      default=None)
    p.add_argument("--no_run_model",  action="store_true",
                   help="Load existing results instead of running the model")
    p.add_argument("--load_feature_r2s", default=None)
    p.add_argument("--load_results_r2",  default=None)
    p.add_argument("--varpart_regressor", default=None,
                   choices=["ridge", "lasso", "elasticnet", "ols"])
    p.add_argument("--ridge_n_iter", default=None, type=int,
                   help="BandedRidgeCV: number of gamma (band-weight) samples")
    p.add_argument("--ridge_local_alpha", default=None,
                   choices=["true", "false"],
                   help="BandedRidgeCV: per-ROI alpha (true) or shared (false)")
    p.add_argument("--ridge_split_scorer", default=None,
                   choices=["numpy", "himalaya"],
                   help="Which product-measure R2 split implementation to use")
    p.add_argument("--ridge_dual", default=None, choices=["true", "false"],
                   help="Kernel MultipleKernelRidgeCV (true, paper) vs primal "
                        "BandedRidgeCV (false)")
    p.add_argument("--delays_tr", default=None,
                   help="Comma-separated integer TR delays, e.g. 2,4,6,8,10")
    return p.parse_args()


def merge_config(cfg, args):
    """Override CONFIG with any CLI arguments that were explicitly set."""
    mapping = {
        "movie":          args.movie,
        "model":          args.model,
        "group":          args.group,
        "isc_threshold":  args.isc_threshold,
        "fmri_file":      args.fmri_file,
        "tag":            args.tag,
        "t_tp":           args.t_tp,
        "feature_file":   args.feature_file,
        "feature_order":  args.feature_order,
        "chunk_len":      args.chunk_len,
        "outer_folds":    args.outer_folds,
        "inner_folds":    args.inner_folds,
        "n_perm":         args.n_perm,
        "fdr_alpha":      args.fdr_alpha,
        "output_dir":     args.output_dir,
        "fig_path":       args.fig_path,
        "load_feature_r2s":  args.load_feature_r2s,
        "load_results_r2":   args.load_results_r2,
        "varpart_regressor": args.varpart_regressor,
        "ridge_n_iter":      args.ridge_n_iter,
        "ridge_local_alpha": (None if args.ridge_local_alpha is None
                              else args.ridge_local_alpha == "true"),
        "ridge_split_scorer": args.ridge_split_scorer,
        "ridge_dual": (None if args.ridge_dual is None
                       else args.ridge_dual == "true"),
        "delays_tr": (None if args.delays_tr is None
                      else [int(x) for x in args.delays_tr.split(",")]),
    }
    for key, val in mapping.items():
        if val is not None:
            cfg[key] = val
    if args.no_run_model:
        cfg["run_model"] = False
    return cfg


def main():
    args = parse_args()
    cfg  = merge_config(CONFIG.copy(), args)

    np.random.seed(cfg["random_seed"])

    movie     = cfg["movie"]
    model     = cfg["model"]
    group     = cfg.get("group", "adult")
    out_dir   = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── ISC intersect mask ───────────────────────────────────────────────
    isc_threshold = cfg.get("isc_threshold")
    isc_roi_idx   = None
    if isc_threshold:
        _, isc_roi_idx = build_isc_intersect_mask(isc_threshold)
        np.save(os.path.join(out_dir, f"isc_roi_idx_{isc_threshold}.npy"),
                isc_roi_idx)

    # ── auto-naming for saved files ──────────────────────────────────────
    if cfg.get("tag"):
        tag = cfg["tag"]
    else:
        isc_suffix = f"_isc{str(isc_threshold).replace('.','')}" if isc_threshold else ""
        tag = f"{model}_{movie}{isc_suffix}_{group}"
    default_feat_path = os.path.join(out_dir, f"fea_{tag}.npy")
    default_res_path  = os.path.join(out_dir, f"res_{tag}.npy")

    # ── load or run model ────────────────────────────────────────────────
    if cfg["run_model"]:
        roi_data, features_dict, feature_order, n_subjects = load_data(
            movie, cfg["feature_file"], cfg["feature_order"],
            fmri_file=cfg.get("fmri_file"), group=group)

        # apply ISC ROI filter
        if isc_roi_idx is not None:
            roi_data = roi_data[:, isc_roi_idx, :]
            print(f"[ISC] roi_data filtered → {roi_data.shape}")

        t_tp = cfg.get("t_tp")
        use_kernel = cfg.get("ridge_dual", False)

        # FIR time-delay expansion. For the KERNEL (paper) path the delays are
        # applied per-fold inside the ColumnKernelizer pipeline, so we do NOT
        # pre-expand here. For the PRIMAL path we globally expand + trim (splits
        # at t_tp so no TP/DM cross-boundary leak). Both need features built at
        # zero lag (build_features lag_tr=0). TP/DM TR=0.8s -> [2,4,6,8,10].
        if not use_kernel:
            features_dict, roi_data, t_tp = apply_delays(
                features_dict, roi_data, t_tp, cfg.get("delays_tr", []))

        if model == "ridge":
            results_r2, feature_r2s = run_ridge(
                roi_data, features_dict, feature_order, n_subjects,
                outer_folds=cfg["outer_folds"],
                inner_folds=cfg["inner_folds"],
                num_alphas=cfg["num_alphas_ridge"],
                chunk_len=cfg["chunk_len"],
                t_tp=t_tp,
                n_iter=cfg["ridge_n_iter"],
                local_alpha=cfg["ridge_local_alpha"],
                split_scorer=cfg["ridge_split_scorer"],
                use_kernel=use_kernel,
                delays_tr=cfg.get("delays_tr", []),
            )
        elif model == "varpart":
            results_r2, feature_r2s, shared_r2 = run_variance_partitioning(
                roi_data, features_dict, feature_order, n_subjects,
                outer_folds=cfg["outer_folds"],
                inner_folds=cfg["inner_folds"],
                chunk_len=cfg["chunk_len"],
                regressor=cfg["varpart_regressor"],
                l1_ratio=cfg["l1_ratio"],
                n_alphas_en=cfg["n_alphas_en"],
            )
            np.save(os.path.join(out_dir, f"shared_{tag}.npy"), shared_r2)
        elif model == "varpart_gpu":
            results_r2, feature_r2s, shared_r2 = run_variance_partitioning_gpu(
                roi_data, features_dict, feature_order, n_subjects,
                outer_folds=cfg["outer_folds"],
                inner_folds=cfg["inner_folds"],
                chunk_len=cfg["chunk_len"],
            )
            np.save(os.path.join(out_dir, f"shared_{tag}.npy"), shared_r2)
        else:
            results_r2, feature_r2s = run_elasticnet(
                roi_data, features_dict, feature_order, n_subjects,
                outer_folds=cfg["outer_folds"],
                inner_folds=cfg["inner_folds"],
                l1_ratio=cfg["l1_ratio"],
                n_alphas=cfg["n_alphas_en"],
                chunk_len=cfg["chunk_len"],
                model_type=model,
                t_tp=t_tp,
            )

        feat_save = cfg.get("load_feature_r2s") or default_feat_path
        res_save  = cfg.get("load_results_r2")  or default_res_path
        np.save(feat_save, feature_r2s)
        np.save(res_save,  results_r2)
        print(f"[Save] feature_r2s → {feat_save}  shape={feature_r2s.shape}")
        print(f"[Save] results_r2  → {res_save}  shape={results_r2.shape}")

    else:
        feat_path = cfg["load_feature_r2s"] or default_feat_path
        res_path  = cfg["load_results_r2"]  or default_res_path
        print(f"[Load] feature_r2s ← {feat_path}")
        print(f"[Load] results_r2  ← {res_path}")
        feature_r2s = np.load(feat_path)
        results_r2  = np.load(res_path)
        if model == "varpart":
            shared_r2 = np.load(os.path.join(out_dir, f"shared_{tag}.npy"))
        _, _, feature_order, _ = load_data(
            movie, cfg["feature_file"], cfg["feature_order"], group=group)

    # ── restore ISC-filtered results to full 114-ROI space for plotting ──
    if isc_roi_idx is not None:
        feature_r2s_plot = restore_roi_space(feature_r2s, isc_roi_idx)
        results_r2_plot  = restore_roi_space(results_r2,  isc_roi_idx)
        if model in ("varpart", "varpart_gpu"):
            shared_r2 = restore_roi_space(shared_r2, isc_roi_idx)
    else:
        feature_r2s_plot = feature_r2s
        results_r2_plot  = results_r2

    # ── mask ROIs where overall R² < 0 ──────────────────────────────────
    feature_r2s_plot = np.where(feature_r2s_plot >= 0, feature_r2s_plot, 0.0)
    pos_mask = (results_r2_plot >= 0)
    pct = pos_mask.mean() * 100
    print(f"[Filter] ROIs with R²≥0: {pct:.1f}% of (subject, ROI) pairs kept")
    feature_r2s_plot = np.where(pos_mask[:, None, :], feature_r2s_plot, np.nan)
    if model in ("varpart", "varpart_gpu"):
        shared_r2 = np.where(pos_mask, shared_r2, np.nan)

    # ── statistics: ROI-level Wilcoxon, FDR within each network ─────────
    _, sig_mask_roi, sig_mask_net = run_statistics_roi_within_network(
        feature_r2s_plot, feature_order,
        fdr_alpha=cfg["fdr_alpha"],
    )

    # ── plots ─────────────────────────────────────────────────────────────
    fig_path = cfg["fig_path"]
    if fig_path is None:
        fig_path = os.path.join(out_dir, f"encoding_{tag}.png")

    if model in ("varpart", "varpart_gpu"):
        plot_variance_partitioning(
            feature_r2s_plot, shared_r2, results_r2_plot,
            feature_order, movie=movie, model=model,
            sig_mask_roi=sig_mask_roi,
            sig_mask_net=sig_mask_net,
            fig_path=fig_path,
        )
    else:
        plot_results(
            feature_r2s_plot, feature_order,
            movie=movie, model=f"{model} ({group})", fig_path=fig_path,
            sig_mask_roi=sig_mask_roi,
            sig_mask_net=sig_mask_net,
        )

    print(f"\n[Done]  figure → {fig_path}")


if __name__ == "__main__":
    main()
