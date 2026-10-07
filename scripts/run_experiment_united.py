"""United-dataset classification entry point: nested-CV classical baselines on
the pooled, harmonized, 3-instrument dataset (spectra_ml.united). Same family
set as outputs/old's original comparison (dummy, logreg, linear_svm,
pls_logreg, rbf_svm, hgb, extratrees) -- built directly (not via
extended_candidate_pool, which also includes random_forest, not part of the
OLD comparison being mirrored here).

Scope: a single non-blinded, all-data pooled nested-CV analysis (every
labeled patient used for both training and evaluation across outer folds,
like OLD's original study) -- NOT the Phase-2 blinded validation protocol.
Instrument id is metadata only and is never concatenated into the feature
matrix passed to any candidate.

Usage: python scripts/run_experiment_united.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.experiment import run_all_families_nested_cv
from spectra_ml.models.base import ModelCandidate
from spectra_ml.models.classical import (build_dummy_candidates, build_extra_trees_candidates,
                                          build_gradient_boosting_candidates, build_linear_svm_candidates,
                                          build_logreg_candidates, build_pls_logreg_candidates,
                                          build_rbf_svm_candidates)
from spectra_ml.qc import HARD_FAIL
from spectra_ml.united import build_united_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_SEED = 42
OUT = os.path.join(ROOT, "outputs", "united")


def united_candidate_pool(seed: int = 42) -> list[ModelCandidate]:
    """dummy, logreg, linear_svm, pls_logreg, rbf_svm, hgb, extratrees -- same
    7-family set as outputs/old/metrics.csv (extended_candidate_pool minus
    random_forest, which was never part of OLD's reported comparison)."""
    return (
        build_dummy_candidates()
        + build_logreg_candidates(seed=seed)
        + build_linear_svm_candidates(seed=seed)
        + build_pls_logreg_candidates(seed=seed)
        + build_rbf_svm_candidates(seed=seed)
        + build_gradient_boosting_candidates(seed=seed)
        + build_extra_trees_candidates(seed=seed)
    )


def load_modeling_set():
    ud = build_united_dataset()
    keep = (ud.qc_status != HARD_FAIL) & ~np.isnan(ud.target)
    spectra, target, patient_id = ud.spectra[keep], ud.target[keep], ud.patient_id[keep]
    instrument = ud.instrument[keep]  # metadata only -- never fed to any model
    _, first_idx = np.unique(patient_id, return_index=True)
    n_park_p = int((target[first_idx] == 1).sum())
    n_ctrl_p = int((target[first_idx] == 0).sum())
    print(f"[united] modeling set: {keep.sum()} spectra, {len(first_idx)} patients "
          f"(park={n_park_p}, control={n_ctrl_p})")
    for role in ("old", "new_pd", "new_control"):
        m = instrument == role
        pid_role = patient_id[m]
        print(f"    {role}: {m.sum()} spectra, {len(np.unique(pid_role))} patients")
    return ud, spectra, target, patient_id, instrument


def main():
    ud, spectra, target, patient_id, instrument = load_modeling_set()

    pool = united_candidate_pool(seed=RANDOM_SEED)
    families = sorted({c.family for c in pool})
    print(f"  candidate families: {families} ({len(pool)} total candidates)")

    t0 = time.time()
    predictions, fold_metrics = run_all_families_nested_cv(pool, spectra, target, patient_id, ud.waves,
                                                             seed=RANDOM_SEED)
    print(f"  nested CV: {time.time() - t0:.1f}s")

    pred_dir = os.path.join(OUT, "predictions")
    models_dir = os.path.join(OUT, "models")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    predictions.to_csv(os.path.join(pred_dir, "patient_predictions.csv"), index=False)
    fold_metrics.drop(columns=["chosen_params"]).to_csv(os.path.join(models_dir, "fold_metrics.csv"), index=False)

    metric_cols = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc",
                    "brier", "log_loss"]
    summary = fold_metrics.groupby("family")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(OUT, "metrics.csv"), index=False)

    print("\nSummary (mean +/- std across outer folds):")
    for _, row in summary.iterrows():
        print(f"  {row['family']:>12s}: ROC-AUC={row['roc_auc_mean']:.3f}+/-{row['roc_auc_std']:.3f}  "
              f"PR-AUC={row['pr_auc_mean']:.3f}  sens={row['sensitivity_mean']:.3f}  spec={row['specificity_mean']:.3f}")
    print(f"\n  -> {pred_dir}/patient_predictions.csv")
    print(f"  -> {models_dir}/fold_metrics.csv")
    print(f"  -> {os.path.join(OUT, 'metrics.csv')}")


if __name__ == "__main__":
    main()
