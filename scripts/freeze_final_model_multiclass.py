"""Part XVII/XX/XXII of current_tasks/double_blind_exp.md, adapted to the
4-class multiclass problem: freeze the 3 linear-only families
(`pca_mnlogreg`, `pls_mnlogreg`, `linear_svm_mc_reduced`; `linear_svm_mc_reduced`
freshly re-verified post the sample-weight-routing bugfix in
GroupAwareCalibratedLinearSVC) plus a `dummy_stratified` floor baseline, using
DEVELOPMENT data ONLY (outputs/blinded_multiclass/development_manifest.csv).
No calibration (standing instruction for this multiclass thread).

Two distinct uses of development-only CV, kept explicit (Part XX: model
selection only within development):
  1. "dev-CV" numbers for the report: full nested (outer+inner) patient-
     grouped CV restricted to development, via the SAME
     `spectra_ml.multiclass.run_all_families_nested_cv_multiclass` used by
     the original (non-blinded) united_multiclass_v2 study -- gives an
     unbiased estimate comparable in methodology to that study's own
     numbers, just on the smaller (103-patient) development-only cohort.
  2. Final hyperparameter selection + refit: a SEPARATE single round of
     patient-grouped CV over the whole development set is used only to pick
     each family's winning candidate (n_components / C), then that candidate
     is refit ONCE on the FULL development set -- the artifact that actually
     gets frozen and used for blind prediction. This mirrors
     `_select_best_candidate_multiclass`'s own inner-fold mechanics exactly
     (reused unchanged, just fed a flat set of folds via a placeholder
     NestedFold instead of a real outer split).

Writes:
  outputs/blinded_multiclass/dev_cv_metrics.csv          (nested-CV, development-only, comparable to metrics.csv)
  outputs/blinded_multiclass/frozen/final_model_<family>.joblib
  outputs/blinded_multiclass/frozen/wavelength_grid.npy
  outputs/blinded_multiclass/frozen/k_new_pd_to_old.npy / k_new_control_to_old.npy / transfer_functions_meta.json
    (frozen COPIES of the harmonization_v2 curves used by predict_blinded_multiclass.py, so blind prediction
     never reads the mutable outputs/harmonization_v2/ working directory directly)
  outputs/blinded_multiclass/frozen/final_protocol.json  (chosen candidate + params + selection AUC per family)
  outputs/blinded_multiclass/frozen/frozen_manifest.json (SHA256 of every artifact above)

Usage: python scripts/freeze_final_model_multiclass.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd

from run_experiment_united_stage_v2 import CLASS_LABELS, load_modeling_set  # noqa: E402
from spectra_ml.evaluation import patient_balanced_sample_weight
from spectra_ml.experiment import _fit_with_sample_weight
from spectra_ml.integrity import write_manifest
from spectra_ml.models.classical import multiclass_stage_candidate_pool_v2, multiclass_stage_candidate_pool_v2_nonlinear_reduced
from spectra_ml.models.cnn1d import build_cnn1d_multiclass_candidates
from spectra_ml.models.mlp import build_mlp_multiclass_candidates
from spectra_ml.multiclass import _select_best_candidate_multiclass, run_all_families_nested_cv_multiclass
from spectra_ml.preprocessing import build_preprocessing_pipeline
from spectra_ml.splitting import Fold, NestedFold, group_stratified_splits_multiclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "outputs", "blinded_multiclass")
FROZEN = os.path.join(STUDY, "frozen")
HARMONIZATION_V2 = os.path.join(ROOT, "outputs", "harmonization_v2")
SEED = 42


def _development_subset():
    """Rebuild the deterministic full modeling set, then restrict every array
    to the patient_ids recorded in development_manifest.csv (decoupled from
    prepare_blinded_study_multiclass.py's own split call, matching the
    project's existing final_study convention of trusting the manifest file
    as the single source of truth for who is 'development')."""
    usd, spectra, target, patient_id, instrument = load_modeling_set()
    dev = pd.read_csv(os.path.join(STUDY, "development_manifest.csv"), dtype={"patient_id": str})
    dev_ids = set(dev["patient_id"])
    keep = np.isin(patient_id, list(dev_ids))
    if keep.sum() == 0 or len(np.unique(patient_id[keep])) != len(dev_ids):
        raise ValueError("development_manifest.csv patient_ids do not fully match the rebuilt modeling set -- "
                          "investigate before freezing (run prepare_blinded_study_multiclass.py first).")
    return usd.waves, spectra[keep], target[keep], patient_id[keep], instrument[keep]


def _select_and_refit(family_name, candidates, spectra, target, patient_id, waves, seed):
    """Pick the winning candidate for `family_name` via a flat round of
    patient-grouped CV over the WHOLE development set (Part XX: selection may
    only use development), then refit that single candidate's preprocessing +
    model on ALL of development. Single-candidate families (dummy) skip
    selection entirely, matching `_select_best_candidate_multiclass`'s own
    len(candidates) == 1 short-circuit used elsewhere in this project."""
    if len(candidates) == 1:
        best, selection_mean_auc, selection_scores = candidates[0], float("nan"), []
    else:
        folds: list[Fold] = group_stratified_splits_multiclass(patient_id, target, CLASS_LABELS, seed=seed)
        placeholder_outer = NestedFold(0, np.array([], dtype=int), np.array([], dtype=int), folds)
        best, selection_mean_auc, selection_scores = _select_best_candidate_multiclass(
            candidates, spectra, target, patient_id, waves, placeholder_outer, CLASS_LABELS)

    prep = build_preprocessing_pipeline(best.preprocessing_variant, wavelengths=waves)
    Xt = prep.fit_transform(spectra)
    model = best.build_fn()
    sw = patient_balanced_sample_weight(patient_id)
    _fit_with_sample_weight(model, Xt, target, sw, patient_id=patient_id,
                             requires_patient_groups=best.requires_patient_groups)

    return {
        "family": family_name, "chosen_name": best.name, "chosen_params": best.params,
        "preprocessing_variant": best.preprocessing_variant,
        "selection_mean_auc_within_development": selection_mean_auc,
        "selection_fold_aucs": selection_scores, "n_development_patients": int(len(np.unique(patient_id))),
    }, prep, model, best


def main():
    os.makedirs(FROZEN, exist_ok=True)
    waves, spectra, target, patient_id, instrument = _development_subset()

    # Extended pool (task 3 of the nonlinear/neural extension): original 4 linear families
    # (dummy, pca_mnlogreg, pls_mnlogreg, linear_svm_mc_reduced) + task 1's runtime-trimmed
    # PCA-reduced nonlinear families (rf/hgb/rbf-svm/knn, all forced onto preprocessing
    # variant "D") + task 2's new multiclass MLP/CNN1D torch candidates (1 fixed config
    # each, also variant "D"). Additive: multiclass_stage_candidate_pool_v2 itself is
    # untouched, so the original 4-family linear-only run stays reproducible via that
    # function alone if ever needed again.
    pool = (
        multiclass_stage_candidate_pool_v2(seed=SEED)
        + multiclass_stage_candidate_pool_v2_nonlinear_reduced(seed=SEED)
        + build_mlp_multiclass_candidates(seed=SEED)
        + build_cnn1d_multiclass_candidates(seed=SEED)
    )
    by_family: dict[str, list] = {}
    for c in pool:
        by_family.setdefault(c.family, []).append(c)
    print(f"development-only pool: families={sorted(by_family)} n_patients={len(np.unique(patient_id))}")

    # 1) development-only nested CV numbers, for report comparison against the blind test.
    dev_predictions, dev_fold_metrics = run_all_families_nested_cv_multiclass(
        pool, spectra, target, patient_id, waves, CLASS_LABELS, seed=SEED)
    dev_predictions.to_csv(os.path.join(STUDY, "dev_cv_patient_predictions.csv"), index=False)
    metric_cols = ["roc_auc_macro_ovr", "balanced_accuracy", "macro_f1", "brier_multiclass", "log_loss"]
    dev_summary = dev_fold_metrics.groupby("family")[metric_cols].agg(["mean", "std"])
    dev_summary.columns = [f"{a}_{b}" for a, b in dev_summary.columns]
    dev_summary = dev_summary.reset_index()
    dev_summary.to_csv(os.path.join(STUDY, "dev_cv_metrics.csv"), index=False)
    print("\ndevelopment-only nested-CV summary (mean +/- std across outer folds):")
    for _, row in dev_summary.iterrows():
        print(f"  {row['family']:>20s}: macro-AUC(OvR)={row['roc_auc_macro_ovr_mean']:.3f}"
              f"+/-{row['roc_auc_macro_ovr_std']:.3f}  bal_acc={row['balanced_accuracy_mean']:.3f}")

    # 2) select + refit final frozen artifact per family, on the FULL development set.
    artifacts = [os.path.join(STUDY, "dev_cv_metrics.csv"), os.path.join(STUDY, "dev_cv_patient_predictions.csv")]
    protocol = {}
    for family_name, candidates in by_family.items():
        summary, prep, model, best = _select_and_refit(family_name, candidates, spectra, target, patient_id,
                                                          waves, SEED)
        bundle = {"family": family_name, "preprocessing": prep, "model": model, "class_labels": list(CLASS_LABELS),
                  "chosen_name": best.name, "chosen_params": best.params}
        path = os.path.join(FROZEN, f"final_model_{family_name}.joblib")
        joblib.dump(bundle, path)
        artifacts.append(path)
        protocol[family_name] = summary
        print(f"  froze {family_name}: chosen={best.name} params={best.params} "
              f"selection_auc={summary['selection_mean_auc_within_development']}")

    np.save(os.path.join(FROZEN, "wavelength_grid.npy"), waves)
    artifacts.append(os.path.join(FROZEN, "wavelength_grid.npy"))

    # Frozen COPIES of the harmonization curves, in their own subdirectory using their
    # ORIGINAL filenames (k_new_pd_to_old.npy etc.) so build_united_dataset(transfer_dir=...)
    # can be pointed directly at them at predict time (Part XXII: predict must only touch
    # frozen artifacts, never the mutable outputs/harmonization_v2/ working directory).
    harmonization_frozen = os.path.join(FROZEN, "harmonization")
    os.makedirs(harmonization_frozen, exist_ok=True)
    for name in ("k_new_pd_to_old.npy", "k_new_control_to_old.npy", "transfer_functions_meta.json",
                 "wavelength_grid.npy"):
        src, dst = os.path.join(HARMONIZATION_V2, name), os.path.join(harmonization_frozen, name)
        shutil.copyfile(src, dst)
        artifacts.append(dst)

    protocol_path = os.path.join(FROZEN, "final_protocol.json")
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump({
            "random_seed": SEED, "calibration": "disabled (standing instruction for this multiclass thread)",
            "harmonization_curves_source": "outputs/harmonization_v2/ (frozen copies included in this directory)",
            "development_manifest_sha256_note": "verified by frozen_manifest.json below",
            "families": protocol,
        }, f, indent=2, default=str)
    artifacts.append(protocol_path)

    write_manifest(artifacts, os.path.join(FROZEN, "frozen_manifest.json"), base_dir=STUDY, extra={
        "state": "FROZEN_BEFORE_BLIND_PREDICTION", "random_seed": SEED, "families": sorted(by_family),
    })
    print(f"\nFrozen {len(artifacts)} artifacts -> {os.path.join(FROZEN, 'frozen_manifest.json')}")


if __name__ == "__main__":
    main()
