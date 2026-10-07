"""Stage 6 entry point: nested-CV classical baselines (Dummy, Logistic
Regression L1/L2, Linear SVM, PLS->Logistic Regression).

Usage: python scripts/run_experiment.py configs/instrument_old.yaml
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.data import build_dataset
from spectra_ml.experiment import run_all_families_nested_cv
from spectra_ml.models.classical import default_candidate_pool, extended_candidate_pool
from spectra_ml.qc import QCConfig, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instrument_config")
    parser.add_argument("--qc-config", default=os.path.join(ROOT, "configs", "qc.yaml"))
    parser.add_argument("--extended", action="store_true", help="Stage 7 pool: + RBF SVM, HGB, ExtraTrees")
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
    n_park_p, n_ctrl_p = int((target[first_idx] == 1).sum()), int((target[first_idx] == 0).sum())
    print(f"[{instrument.id}] modeling set: {keep.sum()} spectra, {len(first_idx)} patients "
          f"(park={n_park_p}, control={n_ctrl_p})")

    pool = extended_candidate_pool(seed=seed) if args.extended else default_candidate_pool(seed=seed)
    families = sorted({c.family for c in pool})
    print(f"  candidate families: {families} ({len(pool)} total candidates)")

    t0 = time.time()
    predictions, fold_metrics = run_all_families_nested_cv(pool, spectra, target, patient_id, dataset.waves,
                                                             seed=seed)
    print(f"  nested CV: {time.time() - t0:.1f}s")

    pred_dir = os.path.join(ROOT, "outputs", instrument.id, "predictions")
    models_dir = os.path.join(ROOT, "outputs", instrument.id, "models")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    predictions.to_csv(os.path.join(pred_dir, "patient_predictions.csv"), index=False)
    fold_metrics.drop(columns=["chosen_params"]).to_csv(os.path.join(models_dir, "fold_metrics.csv"), index=False)

    metric_cols = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc",
                    "brier", "log_loss"]
    summary = fold_metrics.groupby("family")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(ROOT, "outputs", instrument.id, "metrics.csv"), index=False)

    print("\nSummary (mean +/- std across outer folds):")
    for _, row in summary.iterrows():
        print(f"  {row['family']:>12s}: ROC-AUC={row['roc_auc_mean']:.3f}+/-{row['roc_auc_std']:.3f}  "
              f"PR-AUC={row['pr_auc_mean']:.3f}  sens={row['sensitivity_mean']:.3f}  spec={row['specificity_mean']:.3f}")
    print(f"\n  -> {pred_dir}/patient_predictions.csv")
    print(f"  -> {models_dir}/fold_metrics.csv")
    print(f"  -> {os.path.join(ROOT, 'outputs', instrument.id, 'metrics.csv')}")


if __name__ == "__main__":
    main()
