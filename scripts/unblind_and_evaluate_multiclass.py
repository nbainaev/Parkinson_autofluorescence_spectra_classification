"""Part XXIV/XXV of current_tasks/double_blind_exp.md: verify SHA256 of the
frozen artifacts and the prediction file, join `blinded_predictions.csv` with
the PRIVATE `blinding_key.csv` exactly once, and compute the primary/
secondary endpoints on the blind-test cohort. No refitting happens here.

Primary endpoint: macro one-vs-rest ROC-AUC on the 26 blind-test patients,
with a patient-level, class-stratified bootstrap 95% CI (reusing
`spectra_ml.bootstrap.bootstrap_patient_metrics_multiclass` unchanged).
Secondary: confusion matrix, per-class precision/recall/F1, balanced
accuracy. Development-only nested-CV numbers (frozen at freeze time in
dev_cv_metrics.csv) are reported alongside for direct comparison, per task
instruction -- NOT recomputed here. Per-model bootstrap-replicate AUC values
are also saved (bootstrap_auc_replicates.csv), so a report can plot an
AUC histogram per model without re-running the bootstrap.

Usage: python scripts/unblind_and_evaluate_multiclass.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from run_experiment_united_stage_v2 import CLASS_LABELS  # noqa: E402
from spectra_ml.bootstrap import bootstrap_patient_metrics_multiclass
from spectra_ml.evaluation import compute_patient_metrics_multiclass
from spectra_ml.integrity import sha256_file, verify_manifest
from spectra_ml.united_stage import CLASS_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "outputs", "blinded_multiclass")
FROZEN = os.path.join(STUDY, "frozen")
N_BOOT = 400  # modest, per task budget guidance (300-500 range)


def _assert_full_study_id_coverage(evaluated: pd.DataFrame, predictions: pd.DataFrame, key: pd.DataFrame) -> None:
    """Pure guard (no file I/O, directly unit-testable): an INNER merge
    silently DROPS any row of `predictions` whose study_id is absent from
    `key` (and vice versa) -- `validate=\"many_to_one\"` alone only rejects
    DUPLICATE keys on the right side, not missing ones, and comparing
    nunique(evaluated) to nunique(key) alone is insufficient (their counts
    can coincidentally match even after `predictions` rows were silently
    dropped) -- so all three row/patient counts must agree."""
    if len(evaluated) != len(predictions):
        raise RuntimeError(
            f"Not every blind-test prediction row matched a blinding_key study_id: "
            f"{len(predictions)} prediction rows in, {len(evaluated)} rows after the join."
        )
    if evaluated["study_id"].nunique() != key["study_id"].nunique():
        raise RuntimeError("Not every blind-test patient received a prediction from every model.")


def main():
    verify_manifest(os.path.join(FROZEN, "frozen_manifest.json"), base_dir=STUDY)
    pred_manifest_path = os.path.join(STUDY, "blinded_predictions_manifest.json")
    verify_manifest(pred_manifest_path, base_dir=STUDY)
    with open(pred_manifest_path, encoding="utf-8") as f:
        pred_meta = json.load(f)
    if pred_meta["frozen_manifest_sha256"] != sha256_file(os.path.join(FROZEN, "frozen_manifest.json")):
        raise RuntimeError("Frozen manifest changed after predictions were written -- refusing to unblind.")

    predictions = pd.read_csv(os.path.join(STUDY, "blinded_predictions.csv"), dtype={"study_id": str})
    key = pd.read_csv(os.path.join(STUDY, "private", "blinding_key.csv"),
                       dtype={"study_id": str, "real_patient_id": str})
    evaluated = predictions.merge(key[["study_id", "target", "stage_name"]], on="study_id", validate="many_to_one")
    _assert_full_study_id_coverage(evaluated, predictions, key)
    evaluated.to_csv(os.path.join(STUDY, "unblinded_predictions.csv"), index=False)

    prob_cols = [f"prob_{CLASS_NAMES[c]}" for c in CLASS_LABELS]
    reports, cms = [], {}
    boot_replicates_all = []  # per-model bootstrap-replicate AUC values, for the AUC-histogram plot
    for model, sub in evaluated.groupby("model"):
        y_true = sub["target"].to_numpy()
        prob_matrix = sub[prob_cols].to_numpy()
        metrics = compute_patient_metrics_multiclass(y_true, prob_matrix, CLASS_LABELS)
        patient_df = sub.rename(columns={"target": "target", **{c: c for c in prob_cols}}).copy()
        patient_df["patient_id"] = patient_df["study_id"]
        for c, col in zip(CLASS_LABELS, prob_cols):
            patient_df[f"prob_{c}"] = patient_df[col]
        boot_replicates, boot_summary = bootstrap_patient_metrics_multiclass(
            patient_df, CLASS_LABELS, prob_cols=[f"prob_{c}" for c in CLASS_LABELS], n_boot=N_BOOT, seed=42)
        boot_replicates_all.append(pd.DataFrame({
            "model": model, "replicate": np.arange(len(boot_replicates)),
            "roc_auc_macro_ovr": boot_replicates["roc_auc_macro_ovr"],
        }))
        auc_row = boot_summary[boot_summary["metric"] == "roc_auc_macro_ovr"].iloc[0]
        reports.append({
            "model": model, "n_patients": metrics["n_patients"],
            "roc_auc_macro_ovr": metrics["roc_auc_macro_ovr"],
            "roc_auc_macro_ovr_bootstrap_mean": auc_row["mean"],
            "roc_auc_macro_ovr_ci95_lo": auc_row["ci95_lo"], "roc_auc_macro_ovr_ci95_hi": auc_row["ci95_hi"],
            "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"],
            "brier_multiclass": metrics["brier_multiclass"], "log_loss": metrics["log_loss"],
            "per_class_precision": metrics["per_class_precision"], "per_class_recall": metrics["per_class_recall"],
            "per_class_f1": metrics["per_class_f1"], "per_class_support": metrics["per_class_support"],
        })
        cms[model] = {"confusion_matrix": metrics["confusion_matrix"],
                      "labels": [CLASS_NAMES[c] for c in CLASS_LABELS]}

    with open(os.path.join(STUDY, "confusion_matrices.json"), "w", encoding="utf-8") as f:
        json.dump(cms, f, indent=2)

    # Bootstrap-replicate AUC values per model, saved so the report can plot a per-model
    # AUC histogram (distinct from the existing bar-chart-with-CI-whiskers) without
    # re-running the (cheap, OOF-only) bootstrap.
    pd.concat(boot_replicates_all, ignore_index=True).to_csv(
        os.path.join(STUDY, "bootstrap_auc_replicates.csv"), index=False)

    table = pd.DataFrame(reports)
    dev_cv = pd.read_csv(os.path.join(STUDY, "dev_cv_metrics.csv"))
    dev_cv_slim = dev_cv[["family", "roc_auc_macro_ovr_mean", "roc_auc_macro_ovr_std",
                           "balanced_accuracy_mean", "macro_f1_mean"]].rename(columns={"family": "model"})
    comparison = table.merge(dev_cv_slim, on="model", how="left", suffixes=("_blind", "_dev_cv"))

    with open(os.path.join(STUDY, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, allow_nan=True)
    table.drop(columns=["per_class_precision", "per_class_recall", "per_class_f1", "per_class_support"]).to_csv(
        os.path.join(STUDY, "evaluation_metrics.csv"), index=False)
    comparison.to_csv(os.path.join(STUDY, "evaluation_metrics_with_dev_cv.csv"), index=False)

    print(f"Unblinded {evaluated.study_id.nunique()} blind-test patients after SHA256 integrity verification.")
    print("\nblind-test macro-OvR AUC (bootstrap 95% CI) vs development-only nested-CV AUC:")
    for _, row in comparison.iterrows():
        print(f"  {row['model']:>22s}: blind={row['roc_auc_macro_ovr']:.3f} "
              f"[{row['roc_auc_macro_ovr_ci95_lo']:.3f}, {row['roc_auc_macro_ovr_ci95_hi']:.3f}]   "
              f"dev-CV={row['roc_auc_macro_ovr_mean']:.3f}+/-{row['roc_auc_macro_ovr_std']:.3f}")
    print(f"\n  -> {STUDY}/unblinded_predictions.csv")
    print(f"  -> {STUDY}/evaluation_metrics.csv (+ .json, + _with_dev_cv.csv)")
    print(f"  -> {STUDY}/confusion_matrices.json")
    print(f"  -> {STUDY}/bootstrap_auc_replicates.csv")


if __name__ == "__main__":
    main()
