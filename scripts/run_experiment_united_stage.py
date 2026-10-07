"""Task 3: multiclass (4-class Hoehn-Yahr PD stage: control/stage1/stage2/
stage3) classification entry point -- nested patient-grouped CV on the pooled,
harmonized, 3-instrument dataset (spectra_ml.united_stage).

Scope (runtime discipline, see current_tasks): dummy(stratified) + multinomial
logreg + linear SVM (OvR, calibrated) + PLS+multinomial-logreg only. Neural
nets, tree ensembles, and extended hyperparameter grids are explicitly OUT of
scope for this task -- see spectra_ml.models.classical.multiclass_stage_
candidate_pool. Instrument id is metadata only, never a model feature.

Usage: python scripts/run_experiment_united_stage.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.models.base import ModelCandidate
from spectra_ml.models.classical import multiclass_stage_candidate_pool
from spectra_ml.multiclass import run_all_families_nested_cv_multiclass
from spectra_ml.qc import HARD_FAIL
from spectra_ml.united import build_united_dataset
from spectra_ml.united_stage import CLASS_NAMES, build_united_stage_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_SEED = 42
OUT = os.path.join(ROOT, "outputs", "united_multiclass")
CLASS_LABELS = (0.0, 1.0, 2.0, 3.0)


def load_modeling_set():
    ud = build_united_dataset()
    usd = build_united_stage_dataset(ud)
    keep = (usd.qc_status != HARD_FAIL) & ~np.isnan(usd.stage_class)
    spectra, target, patient_id = usd.spectra[keep], usd.stage_class[keep], usd.patient_id[keep]
    instrument = usd.instrument[keep]  # metadata only -- never fed to any model
    _, first_idx = np.unique(patient_id, return_index=True)
    counts = {CLASS_NAMES[c]: int((target[first_idx] == c).sum()) for c in CLASS_LABELS}
    print(f"[united_multiclass] modeling set: {keep.sum()} spectra, {len(first_idx)} patients {counts}")
    for role in ("old", "new_pd", "new_control"):
        m = instrument == role
        pid_role = patient_id[m]
        print(f"    {role}: {m.sum()} spectra, {len(np.unique(pid_role))} patients")
    return usd, spectra, target, patient_id, instrument


def main():
    usd, spectra, target, patient_id, instrument = load_modeling_set()
    waves = usd.waves

    pool = multiclass_stage_candidate_pool(seed=RANDOM_SEED)
    families = sorted({c.family for c in pool})
    print(f"  candidate families: {families} ({len(pool)} total candidates)")

    t0 = time.time()
    predictions, fold_metrics = run_all_families_nested_cv_multiclass(
        pool, spectra, target, patient_id, waves, CLASS_LABELS, seed=RANDOM_SEED)
    print(f"  nested CV: {time.time() - t0:.1f}s")

    pred_dir = os.path.join(OUT, "predictions")
    models_dir = os.path.join(OUT, "models")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    predictions.to_csv(os.path.join(pred_dir, "patient_predictions.csv"), index=False)
    fold_metrics.drop(columns=["chosen_params", "confusion_matrix", "per_class_precision", "per_class_recall",
                                "per_class_f1", "per_class_support"]).to_csv(
        os.path.join(models_dir, "fold_metrics.csv"), index=False)
    fold_metrics.to_json(os.path.join(models_dir, "fold_metrics_full.json"), orient="records", indent=2)

    metric_cols = ["roc_auc_macro_ovr", "balanced_accuracy", "macro_f1", "brier_multiclass", "log_loss"]
    summary = fold_metrics.groupby("family")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(OUT, "metrics.csv"), index=False)

    print("\nSummary (mean +/- std across outer folds):")
    for _, row in summary.iterrows():
        print(f"  {row['family']:>14s}: macro-AUC(OvR)={row['roc_auc_macro_ovr_mean']:.3f}"
              f"+/-{row['roc_auc_macro_ovr_std']:.3f}  bal_acc={row['balanced_accuracy_mean']:.3f}"
              f"  macroF1={row['macro_f1_mean']:.3f}")
    print(f"\n  -> {pred_dir}/patient_predictions.csv")
    print(f"  -> {models_dir}/fold_metrics.csv (+ fold_metrics_full.json)")
    print(f"  -> {os.path.join(OUT, 'metrics.csv')}")


if __name__ == "__main__":
    main()
