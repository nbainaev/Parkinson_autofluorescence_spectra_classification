"""Task 3 (calibration): nested-CV multiclass classical baselines with
per-class one-vs-rest OOF Platt calibration per outer fold, on the pooled
harmonized 4-class Hoehn-Yahr stage dataset. Mirrors
scripts/run_calibration_united.py's structure.

Usage: python scripts/run_calibration_united_stage.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectra_ml.multiclass import run_all_families_nested_cv_multiclass_calibrated
from run_experiment_united_stage import CLASS_LABELS, RANDOM_SEED, load_modeling_set  # noqa: E402
from spectra_ml.models.classical import multiclass_stage_candidate_pool
from spectra_ml.united_stage import CLASS_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united_multiclass")


def main():
    usd, spectra, target, patient_id, instrument = load_modeling_set()
    waves = usd.waves

    pool = multiclass_stage_candidate_pool(seed=RANDOM_SEED)
    families = sorted({c.family for c in pool})
    print(f"  candidate families: {families} ({len(pool)} total candidates)")

    t0 = time.time()
    predictions, fold_metrics = run_all_families_nested_cv_multiclass_calibrated(
        pool, spectra, target, patient_id, waves, CLASS_LABELS, seed=RANDOM_SEED)
    print(f"  nested CV + per-class OvR calibration: {time.time() - t0:.1f}s")

    pred_dir = os.path.join(OUT, "predictions")
    calib_dir = os.path.join(OUT, "calibration")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    predictions.to_csv(os.path.join(pred_dir, "patient_predictions_calibrated.csv"), index=False)
    drop_cols = [c for c in ("chosen_params", "raw_confusion_matrix", "calibrated_confusion_matrix",
                              "calibrated_per_class_recall", "calibrated_per_class_precision")
                 if c in fold_metrics.columns]
    fold_metrics.drop(columns=drop_cols).to_csv(os.path.join(calib_dir, "calibration_fold_metrics.csv"),
                                                  index=False)
    fold_metrics.to_json(os.path.join(calib_dir, "calibration_fold_metrics_full.json"), orient="records", indent=2)

    metric_cols = ["roc_auc_macro_ovr", "balanced_accuracy", "macro_f1", "brier_multiclass", "log_loss"]
    cols = [f"raw_{c}" for c in metric_cols] + [f"calibrated_{c}" for c in metric_cols]
    summary = fold_metrics.groupby("family")[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(calib_dir, "calibration_summary.csv"), index=False)

    print("\nmacro-OvR-AUC / multiclass Brier, raw -> calibrated (mean across outer folds):")
    for _, row in summary.iterrows():
        print(f"  {row['family']:>14s}: AUC {row['raw_roc_auc_macro_ovr_mean']:.3f} -> "
              f"{row['calibrated_roc_auc_macro_ovr_mean']:.3f}   Brier "
              f"{row['raw_brier_multiclass_mean']:.3f} -> {row['calibrated_brier_multiclass_mean']:.3f}")

    print(f"\n  -> {pred_dir}/patient_predictions_calibrated.csv")
    print(f"  -> {calib_dir}/calibration_summary.csv")


if __name__ == "__main__":
    main()
