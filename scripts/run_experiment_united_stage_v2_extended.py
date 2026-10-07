"""Task 2 (united_multiclass_v2 audit-and-extension): SAME nested patient-
grouped-CV multiclass framework as scripts/run_experiment_united_stage_v2.py,
SAME modeling cohort (129 patients), SAME metrics -- extended candidate pool
(spectra_ml.models.classical.multiclass_stage_candidate_pool_v2_extended):
the original v2 pool (dummy, pca_mnlogreg, pls_mnlogreg,
linear_svm_mc_reduced) UNCHANGED, plus new nonlinear families (Random Forest,
HistGradientBoosting -- each both raw-278-wavelength and PCA(n=10)-reduced;
RBF-SVM and k-NN, PCA(n=10)-reduced only). See
spectra_ml.models.classical's module note above
multiclass_stage_candidate_pool_v2_nonlinear for the deliberate raw-feature
exception for tree ensembles and other scope trade-offs.

Deliberately a SEPARATE script/output set from run_experiment_united_stage_v2.py:
outputs/united_multiclass_v2/metrics.csv (the original 4-family run) is left
untouched; this writes metrics_extended.csv / predictions/
patient_predictions_extended.csv / models/fold_metrics_extended.csv (+ _full.json)
/ chosen_n_components_extended.csv alongside it, so the original result stays
exactly reproducible.

Usage: python scripts/run_experiment_united_stage_v2_extended.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from spectra_ml.models.classical import multiclass_stage_candidate_pool_v2_extended
from spectra_ml.multiclass import run_all_families_nested_cv_multiclass
from run_experiment_united_stage_v2 import CLASS_LABELS, RANDOM_SEED, load_modeling_set  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united_multiclass_v2")


def main():
    usd, spectra, target, patient_id, instrument = load_modeling_set()
    waves = usd.waves

    pool = multiclass_stage_candidate_pool_v2_extended(seed=RANDOM_SEED)
    families = sorted({c.family for c in pool})
    print(f"  candidate families (extended): {families} ({len(pool)} total candidates)")

    t0 = time.time()
    predictions, fold_metrics = run_all_families_nested_cv_multiclass(
        pool, spectra, target, patient_id, waves, CLASS_LABELS, seed=RANDOM_SEED)
    elapsed = time.time() - t0
    print(f"  nested CV (extended): {elapsed:.1f}s")

    pred_dir = os.path.join(OUT, "predictions")
    models_dir = os.path.join(OUT, "models")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    predictions.to_csv(os.path.join(pred_dir, "patient_predictions_extended.csv"), index=False)
    fold_metrics.drop(columns=["chosen_params", "confusion_matrix", "per_class_precision", "per_class_recall",
                                "per_class_f1", "per_class_support"]).to_csv(
        os.path.join(models_dir, "fold_metrics_extended.csv"), index=False)
    fold_metrics.to_json(os.path.join(models_dir, "fold_metrics_extended_full.json"), orient="records", indent=2)

    metric_cols = ["roc_auc_macro_ovr", "balanced_accuracy", "macro_f1", "brier_multiclass", "log_loss"]
    summary = fold_metrics.groupby("family")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(OUT, "metrics_extended.csv"), index=False)

    n_comp_rows = []
    for _, row in fold_metrics.iterrows():
        params = row["chosen_params"]
        if isinstance(params, dict) and "n_components" in params:
            n_comp_rows.append({"family": row["family"], "outer_fold": row["outer_fold"],
                                 "n_components": params["n_components"]})
    pd.DataFrame(n_comp_rows).to_csv(os.path.join(OUT, "chosen_n_components_extended.csv"), index=False)

    # confusion-matrix summary (argmax over all outer-fold OOF patient predictions pooled per family)
    import numpy as np
    from spectra_ml.evaluation import compute_patient_metrics_multiclass
    cm_rows = []
    prob_cols = [f"raw_prob_{c}" for c in CLASS_LABELS]
    for family, sub in predictions.groupby("family"):
        m = compute_patient_metrics_multiclass(sub["target"].to_numpy(), sub[prob_cols].to_numpy(), CLASS_LABELS)
        cm_rows.append({"family": family, "confusion_matrix": m["confusion_matrix"],
                         "confusion_matrix_labels": m["confusion_matrix_labels"],
                         "per_class_recall": m["per_class_recall"]})
    pd.DataFrame(cm_rows).to_json(os.path.join(OUT, "confusion_matrices_extended.json"), orient="records", indent=2)

    with open(os.path.join(OUT, "runtime.txt"), "a", encoding="utf-8") as f:
        f.write(f"run_experiment_united_stage_v2_extended.py: {elapsed:.1f}s\n")

    print("\nSummary (mean +/- std across outer folds), extended pool:")
    for _, row in summary.sort_values("roc_auc_macro_ovr_mean", ascending=False).iterrows():
        print(f"  {row['family']:>28s}: macro-AUC(OvR)={row['roc_auc_macro_ovr_mean']:.3f}"
              f"+/-{row['roc_auc_macro_ovr_std']:.3f}  bal_acc={row['balanced_accuracy_mean']:.3f}"
              f"  macroF1={row['macro_f1_mean']:.3f}")
    print(f"\n  -> {pred_dir}/patient_predictions_extended.csv")
    print(f"  -> {models_dir}/fold_metrics_extended.csv (+ _full.json)")
    print(f"  -> {os.path.join(OUT, 'metrics_extended.csv')}")
    print(f"  -> {os.path.join(OUT, 'chosen_n_components_extended.csv')}")
    print(f"  -> {os.path.join(OUT, 'confusion_matrices_extended.json')}")


if __name__ == "__main__":
    main()
