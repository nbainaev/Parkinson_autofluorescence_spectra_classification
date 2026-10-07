"""Stage 8 entry point: nested-CV classical baselines with patient-level OOF
Platt calibration per outer fold.

Usage: python scripts/run_calibration.py configs/instrument_old.yaml
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.data import build_dataset
from spectra_ml.experiment import run_all_families_nested_cv_calibrated
from spectra_ml.models.classical import default_candidate_pool, extended_candidate_pool
from spectra_ml.plotting import plot_probability_histograms, plot_reliability_diagram
from spectra_ml.qc import QCConfig, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instrument_config")
    parser.add_argument("--qc-config", default=os.path.join(ROOT, "configs", "qc.yaml"))
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()

    instrument = InstrumentConfig.from_yaml(args.instrument_config)
    qc_config = QCConfig.from_yaml(args.qc_config)
    seed = instrument.data.random_seed

    dataset = build_dataset(instrument)
    spectrum_id = [f"{pid}_{i}" for i, pid in enumerate(dataset.patient_id)]
    qc_df, hp, _ = build_qc_table(dataset.spectra, dataset.waves, dataset.patient_id, spectrum_id, qc_config)
    corrected, _ = correct_hot_pixels(dataset.spectra, hp, qc_config)

    keep = (qc_df["qc_status"].to_numpy() != "HARD_FAIL") & ~np.isnan(dataset.target)
    spectra, target, patient_id = corrected[keep], dataset.target[keep], dataset.patient_id[keep]
    _, first_idx = np.unique(patient_id, return_index=True)
    print(f"[{instrument.id}] modeling set: {keep.sum()} spectra, {len(first_idx)} patients "
          f"(park={int((target[first_idx] == 1).sum())}, control={int((target[first_idx] == 0).sum())})")

    pool = extended_candidate_pool(seed=seed) if args.extended else default_candidate_pool(seed=seed)
    families = sorted({c.family for c in pool})
    print(f"  candidate families: {families} ({len(pool)} total candidates)")

    t0 = time.time()
    predictions, fold_metrics = run_all_families_nested_cv_calibrated(pool, spectra, target, patient_id,
                                                                        dataset.waves, seed=seed)
    print(f"  nested CV + calibration: {time.time() - t0:.1f}s")

    pred_dir = os.path.join(ROOT, "outputs", instrument.id, "predictions")
    calib_dir = os.path.join(ROOT, "outputs", instrument.id, "calibration")
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
