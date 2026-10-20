"""
run_hvm_diagnostic_singledelay.py — Diagnostic re-run of the human-vs-model
variance partitioning that swaps two design choices for the ones used in the
reference paper (banded-ridge / Pearson-r methodology screenshot), to test
whether they explain why `unique_human`/`solo_human` come out so negative
under run_human_vs_model_variance_partitioning.py's FIR + 5x5-nested-CV setup:

  1. Fixed single hemodynamic shift (paper: 2 TR = 4s) instead of a 5-tap FIR
     delay expansion. TP/DM run at TR=0.8s, so the same 4s shift is 5 TR
     (5*0.8=4.0s); PC runs at TR=2.0s, so 4s is exactly 2 TR (matches the
     paper's own numbers, since PC's grid is coarser).
       DELAYS_TR = {"tp": [5], "dm": [5], "pc": [2]}
     (a single-element list -> apply_delays() degrades to a plain fixed shift,
     each feature stays 1x its original width instead of 5x.)

  2. Near-leave-one-segment-out outer CV (paper: 19 segments of 8 TR = 16s at
     PC's TR=2.0s) instead of 5 big folds. Segment length is matched in real
     time: PC keeps the paper's own 8 TR (16s); TP/DM (TR=0.8s) use the same
     16s, i.e. 20 TR. outer_folds is set to T // chunk_len for each movie so
     GroupKFold(n_splits=outer_folds == n_groups) becomes exact
     leave-one-segment-out:
       CHUNK_LEN   = {"tp": 20, "dm": 20, "pc": 8}
       OUTER_FOLDS = {"tp": 11, "dm": 36, "pc": 20}   (= T[movie] // chunk_len)

Both changes are applied TOGETHER (not isolated) per explicit request. Inner
CV (alpha selection) is left at inner_folds=5, unchanged. score_metric is
Pearson r throughout (see run_human_vs_model_variance_partitioning.py).

Everything else (feature building, z-scoring, FDR significance testing,
output format) reuses the existing "encoding_pipeline copy.py" pipeline
unchanged, so results are directly comparable to
human_vs_model_results/sweep/hvm_*_pearson_r_unique_shared.npz.

Usage:
    python run_hvm_diagnostic_singledelay.py [--movies tp,dm,pc] [--models llada,qwen,gemma] [--measures tom,valence] [--gpu]

Outputs (human_vs_model_results/diagnostic_singledelay/):
    hvm_{movie}_{model}_{measure}_sigma2.0_chunk{N}_singledelay_pearson_r_unique_shared.npz
    fig_hvm_..._singledelay.png
    hvm_summary_diagnostic_singledelay.csv
"""
import argparse
import importlib.util
import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location("encoding_pipeline_copy",
                                               os.path.join(_HERE, "encoding_pipeline copy.py"))
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)

from build_features_human_vs_model import build as build_features

FMRI_FILE = {"tp": "fmri_tp.npy", "dm": "fmri_dm.npy", "pc": "fmri_pc.npy"}

# --- The two swapped design choices (see module docstring) ---
DELAYS_TR   = {"tp": [5], "dm": [5], "pc": [2]}       # single fixed 4s shift
CHUNK_LEN   = {"tp": 20, "dm": 20, "pc": 8}            # 16s segments (paper's own 8TR/16s at PC's TR=2.0s)
OUTER_FOLDS = {"tp": 11, "dm": 36, "pc": 20}           # = T[movie] // chunk_len -> exact leave-one-segment-out
SIGMA_SEC   = 2.0                                       # unchanged baseline smoothing

OUT_DIR = os.path.join(_HERE, "human_vs_model_results", "diagnostic_singledelay")
os.makedirs(OUT_DIR, exist_ok=True)


def run_one(movie, model, measure, use_gpu=False, score_metric="pearson_r"):
    chunk_len   = CHUNK_LEN[movie]
    outer_folds = OUTER_FOLDS[movie]
    print(f"\n{'='*80}\n{movie.upper()} x {model} x {measure}  ({'GPU' if use_gpu else 'CPU'})  "
          f"[DIAGNOSTIC: single delay={DELAYS_TR[movie]}TR, chunk_len={chunk_len}TR, "
          f"outer_folds={outer_folds} (leave-one-segment-out), score_metric={score_metric}]\n{'='*80}")

    feature_file = build_features(movie, model,
                                  out_path=os.path.join(OUT_DIR, f"fea_{movie}_{model}_sigma{SIGMA_SEC}_hvm.npy"),
                                  sigma_sec=SIGMA_SEC)
    feature_order = [f"human_{measure}", f"model_{measure}"]

    roi_data, features_dict, feature_order, n_subjects = ep.load_data(
        movie, feature_file=feature_file, feature_order=feature_order,
        fmri_file=os.path.join(_HERE, FMRI_FILE[movie]))

    features_dict, roi_data, _ = ep.apply_delays(features_dict, roi_data, None, DELAYS_TR[movie])

    if use_gpu:
        results_r2, unique_r2s, shared_r2 = ep.run_variance_partitioning_gpu(
            roi_data, features_dict, feature_order, n_subjects,
            outer_folds=outer_folds, inner_folds=5, chunk_len=chunk_len, score_metric=score_metric)
    else:
        results_r2, unique_r2s, shared_r2 = ep.run_variance_partitioning(
            roi_data, features_dict, feature_order, n_subjects,
            outer_folds=outer_folds, inner_folds=5, chunk_len=chunk_len, regressor="ridge",
            score_metric=score_metric)

    p_net, sig_mask_roi, sig_mask_net = ep.run_statistics_roi_within_network(unique_r2s, feature_order)
    _, sig_mask_roi_global = ep.run_statistics(unique_r2s, feature_order)

    tag = f"{movie}_{model}_{measure}_sigma{SIGMA_SEC}_chunk{chunk_len}_singledelay_{score_metric}"
    npz_path = os.path.join(OUT_DIR, f"hvm_{tag}_unique_shared.npz")
    np.savez(npz_path, results_r2=results_r2, unique_r2s=unique_r2s, shared_r2=shared_r2,
             feature_order=np.array(feature_order),
             sig_mask_roi_network_fdr=sig_mask_roi, sig_mask_net=sig_mask_net,
             sig_mask_roi_global_fdr=sig_mask_roi_global)
    print(f"Saved -> {npz_path}")

    try:
        fig_path = os.path.join(OUT_DIR, f"fig_hvm_{tag}.png")
        ep.plot_variance_partitioning(unique_r2s, shared_r2, results_r2,
                                      feature_order, f"{movie}_{measure}", model,
                                      sig_mask_roi=sig_mask_roi, sig_mask_net=sig_mask_net,
                                      fig_path=fig_path)
        print(f"Saved -> {fig_path}")
    except Exception as e:
        print(f"[WARN] plot_variance_partitioning failed ({e}); npz results are still saved.")

    hi, mi = feature_order.index(f"human_{measure}"), feature_order.index(f"model_{measure}")
    full = float(np.nanmean(results_r2))
    unique_human = float(np.nanmean(unique_r2s[:, hi, :]))
    unique_model = float(np.nanmean(unique_r2s[:, mi, :]))
    shared = float(np.nanmean(shared_r2))
    return dict(
        movie=movie, model=model, measure=measure,
        delay_tr=DELAYS_TR[movie][0], chunk_len=chunk_len, outer_folds=outer_folds,
        full=full, solo_human=full - unique_model, solo_model=full - unique_human,
        unique_human=unique_human, unique_model=unique_model, shared=shared,
        n_sig_roi_human=int(sig_mask_roi[hi].sum()), n_sig_roi_model=int(sig_mask_roi[mi].sum()),
        n_subjects=n_subjects,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", default="tp,dm,pc")
    ap.add_argument("--models", default="llada,qwen,gemma")
    ap.add_argument("--measures", default="tom,valence")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--score-metric", default="pearson_r", choices=["pearson_r", "r2"])
    args = ap.parse_args()

    rows = []
    for movie in args.movies.split(","):
        for model in args.models.split(","):
            for measure in args.measures.split(","):
                rows.append(run_one(movie, model, measure, use_gpu=args.gpu, score_metric=args.score_metric))

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, f"hvm_summary_diagnostic_singledelay_{args.score_metric}.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
