"""Task 3 (bootstrap): patient-level, class-stratified bootstrap CI for
macro-OvR-AUC (+ other scalar multiclass metrics) on the OOF patient_
predictions_calibrated table produced by run_calibration_united_stage.py.
Mirrors scripts/run_bootstrap_united.py.

n_boot default 750 (smaller than the binary pipeline's 2000) -- runtime
discipline, see spectra_ml.bootstrap.bootstrap_patient_metrics_multiclass
docstring for the rationale.

Usage: python scripts/run_bootstrap_united_stage.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from spectra_ml.bootstrap import bootstrap_patient_metrics_multiclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united_multiclass")
RANDOM_SEED = 42
CLASS_LABELS = (0.0, 1.0, 2.0, 3.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=750)
    parser.add_argument("--prob-prefix", default="calibrated_prob_")
    args = parser.parse_args()

    pred_path = os.path.join(OUT, "predictions", "patient_predictions_calibrated.csv")
    predictions = pd.read_csv(pred_path)
    print(f"[united_multiclass] loaded {pred_path}: {predictions['family'].nunique()} families, "
          f"{predictions['patient_id'].nunique()} patients")

    out_dir = os.path.join(OUT, "bootstrap")
    os.makedirs(out_dir, exist_ok=True)

    prob_cols = [f"{args.prob_prefix}{c}" for c in CLASS_LABELS]

    t0 = time.time()
    all_summaries = []
    for family_name, fam_df in predictions.groupby("family"):
        replicates, summary = bootstrap_patient_metrics_multiclass(fam_df, CLASS_LABELS, prob_cols=prob_cols,
                                                                     n_boot=args.n_boot, seed=RANDOM_SEED)
        summary.insert(0, "family", family_name)
        all_summaries.append(summary)
        replicates.to_csv(os.path.join(out_dir, f"replicates_{family_name}.csv"), index=False)

    summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_all.to_csv(os.path.join(out_dir, "bootstrap_summary.csv"), index=False)
    print(f"  bootstrap ({args.n_boot} iters x {predictions['family'].nunique()} families): "
          f"{time.time() - t0:.1f}s")

    print("\nmacro-OvR-AUC 95% CI (patient-level, class-stratified bootstrap):")
    for _, row in summary_all[summary_all["metric"] == "roc_auc_macro_ovr"].iterrows():
        print(f"  {row['family']:>14s}: {row['mean']:.3f}  [{row['ci95_lo']:.3f}, {row['ci95_hi']:.3f}]")

    print(f"\n  -> {out_dir}/bootstrap_summary.csv")


if __name__ == "__main__":
    main()
