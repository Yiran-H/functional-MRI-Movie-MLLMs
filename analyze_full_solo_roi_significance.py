"""
analyze_full_solo_roi_significance.py — ROI-level significance (sign-
permutation, 5000 resamples, FDR-BH -- matches the paper's "compare to
chance" stats section) for the three quantities that stay well-defined under
score_metric="pearson_r" WITHOUT needing an additive variance decomposition:

    full        = 2-feature (human+model) model's held-out Pearson r
    solo_human  = human-only model's held-out Pearson r
    solo_model  = model-only model's held-out Pearson r

Only pearson_r-scored files are analyzed here (r2 files are skipped) -- the
solo/full comparison is only meaningful as a Pearson r quantity per the
project's current decision to treat r2 as unreliable/near-zero for this
weak-signal data (see conversation), so there is no need to also test the r2
version of full/solo.

unique_human/unique_model/shared are differences of correlations and are NOT
tested here (they lack a variance-decomposition justification under Pearson
r). solo_human/solo_model are reconstructed EXACTLY (not approximated) from
the saved results_r2/unique_r2s arrays:
    solo_human[s, r] = results_r2[s, r] - unique_r2s[s, model_idx, r]
    solo_model[s, r] = results_r2[s, r] - unique_r2s[s, human_idx, r]
This is an exact identity (both sides are per-fold-averages of the same
per-fold quantities), not a re-fit or an approximation.

Runs run_statistics() (sign-permutation vs 0, 5000 resamples, global FDR-BH)
on all 3 quantities, for every combo found in a results directory.

Usage:
    python analyze_full_solo_roi_significance.py <results_dir> <tag_suffix>
Examples:
    python analyze_full_solo_roi_significance.py human_vs_model_results/diagnostic_delayonly delayonly
    python analyze_full_solo_roi_significance.py human_vs_model_results/diagnostic_singledelay singledelay

Output:
    <results_dir>/full_solo_roi_significance_pearson_r.csv
"""
import glob
import os
import re
import sys
import importlib.util
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("encoding_pipeline_copy",
                                               os.path.join(_HERE, "encoding_pipeline copy.py"))
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "human_vs_model_results/diagnostic_singledelay"
TAG = sys.argv[2] if len(sys.argv) > 2 else "singledelay"

FNAME_RE = re.compile(
    r"hvm_(?P<movie>tp|dm|pc)_(?P<model>llada|qwen|gemma)_(?P<measure>tom|valence)"
    r"_sigma(?P<sigma>[\d.]+)_chunk(?P<chunk>\d+)_" + TAG + r"_pearson_r_unique_shared\.npz$"
)

files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"hvm_*_{TAG}_pearson_r_unique_shared.npz")))
if not files:
    raise SystemExit(f"No pearson_r files matched in {RESULTS_DIR} with tag '{TAG}'")

rows = []
for f in files:
    m = FNAME_RE.search(os.path.basename(f))
    if not m:
        continue
    d = np.load(f, allow_pickle=True)
    feature_order = list(d["feature_order"])
    measure = m["measure"]
    hi, mi = feature_order.index(f"human_{measure}"), feature_order.index(f"model_{measure}")

    results_r2 = d["results_r2"]         # [S, R]  full (pearson r)
    unique_r2s = d["unique_r2s"]         # [S, F, R]
    solo_human = results_r2 - unique_r2s[:, mi, :]   # [S, R]
    solo_model = results_r2 - unique_r2s[:, hi, :]   # [S, R]

    stacked = np.stack([results_r2, solo_human, solo_model], axis=1)  # [S, 3, R]
    labels = ["full", "solo_human", "solo_model"]
    p_vals, sig_mask = ep.run_statistics(stacked, labels, n_perm=5000, fdr_alpha=0.05)

    for fi, lab in enumerate(labels):
        vals = stacked[:, fi, :]
        rows.append(dict(
            movie=m["movie"], model=m["model"], measure=measure, quantity=lab,
            mean_val=float(np.nanmean(vals)),
            n_roi_positive=int((np.nanmean(vals, axis=0) > 0).sum()),
            n_sig_roi_fdr=int(sig_mask[fi].sum()),
        ))

df = pd.DataFrame(rows).sort_values(["movie", "model", "measure", "quantity"]).reset_index(drop=True)
out_csv = os.path.join(RESULTS_DIR, "full_solo_roi_significance_pearson_r.csv")
df.to_csv(out_csv, index=False)
print(f"Saved -> {out_csv}")

pd.set_option("display.width", 200)
print(df.to_string(index=False))
