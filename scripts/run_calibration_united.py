"""United-dataset calibration entry point: nested-CV classical baselines with
patient-level OOF Platt calibration per outer fold, on the pooled harmonized
3-instrument dataset. Mirrors scripts/run_calibration.py.

Usage: python scripts/run_calibration_united.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectra_ml.experiment import run_all_families_nested_cv_calibrated
from spectra_ml.plotting import plot_probability_histograms, plot_reliability_diagram
from run_experiment_united import RANDOM_SEED, load_modeling_set, united_candidate_pool  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united")


def main():
    ud, spectra, target, patient_id, instrument = load_modeling_set()

    pool = united_candidate_pool(seed=RANDOM_SEED)
    families = sorted({c.family for c in pool})
    print(f"  candidate families: {families} ({len(pool)} total candidates)")

    t0 = time.time()
    predictions, fold_metrics = run_all_families_nested_cv_calibrated(pool, spectra, target, patient_id,
                                                                        ud.waves, seed=RANDOM_SEED)
    print(f"  nested CV + calibration: {time.time() - t0:.1f}s")

    pred_dir = os.path.join(OUT, "predictions")
    calib_dir = os.path.join(OUT, "calibration")
    fig_dir = os.path.join(calib_dir, "figures")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    predictions.to_csv(os.path.join(pred_dir, "patient_predictions.csv"), index=False)
    fold_metrics.drop(columns=["chosen_params"]).to_csv(os.path.join(calib_dir, "calibration_fold_metrics.csv"),
                                                          index=False)

    metric_cols = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc",
                    "brier", "log_loss"]
    cols = [f"raw_{c}" for c in metric_cols] + [f"calibrated_{c}" for c in metric_cols]
    summary = fold_metrics.groupby("family")[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(calib_dir, "calibration_summary.csv"), index=False)

    print("\nBrier / ROC-AUC, raw -> calibrated (mean across outer folds):")
    for _, row in summary.iterrows():
        print(f"  {row['family']:>12s}: ROC-AUC {row['raw_roc_auc_mean']:.3f} -> {row['calibrated_roc_auc_mean']:.3f}"
              f"   Brier {row['raw_brier_mean']:.3f} -> {row['calibrated_brier_mean']:.3f}")

    for family_name, fam_df in predictions.groupby("family"):
        plot_reliability_diagram(fam_df["target"], fam_df["raw_probability"], fam_df["calibrated_probability"],
                                  os.path.join(fig_dir, f"reliability_{family_name}.png"),
                                  n_bins=5, title=f"Reliability: {family_name} (pooled outer-test folds)")
        plot_probability_histograms(fam_df["raw_probability"], fam_df["calibrated_probability"],
                                     os.path.join(fig_dir, f"prob_hist_{family_name}.png"))

    print(f"\n  -> {pred_dir}/patient_predictions.csv")
    print(f"  -> {calib_dir}/calibration_summary.csv")
    print(f"  -> {fig_dir}/")


if __name__ == "__main__":
    main()
