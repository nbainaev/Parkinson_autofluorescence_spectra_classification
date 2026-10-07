"""Task 3 (interpretability, v2): per-class-vs-rest linear coefficients in
wavelength space for the two PLS/PCA + multinomial-logreg families
(pca_mnlogreg, pls_mnlogreg), on the CORRECTED, re-harmonized pooled 4-class
Hoehn-Yahr stage dataset. Mirrors scripts/run_interpretability_united_stage.py's
refit-per-fold structure (same pattern, adapted: the raw-278-dim "mn_logreg"
family from the prior round is replaced by "pca_mnlogreg" here, since every
non-dummy family now goes through PLS/PCA -- see
spectra_ml.models.classical.multiclass_stage_candidate_pool_v2).

Scope note (unchanged from the prior round): grouped permutation importance
and the linear_svm_mc_reduced family's own interpretability are intentionally
NOT reproduced here -- runtime discipline + the CalibratedClassifierCV-wrapped
SVM does not expose a single refit coefficient per fold the same simple way
logistic regression does (see GroupAwareCalibratedLinearSVC); the two PLS/PCA
+ logreg families already answer the scientifically interesting question (do
the same hypothesis bands separate each stage from control).

Usage: python scripts/run_interpretability_united_stage_v2.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from spectra_ml.evaluation import patient_balanced_sample_weight
from spectra_ml.experiment import _fit_with_sample_weight
from spectra_ml.interpretability import aggregate_wavelength_regions, back_project_through_scaler, fold_stability
from spectra_ml.models.classical import build_pca_multinomial_logreg_candidates, build_pls_multinomial_logreg_candidates
from spectra_ml.multiclass import _select_best_candidate_multiclass
from spectra_ml.preprocessing import build_preprocessing_pipeline
from spectra_ml.reduction import PCAFeatureExtractor, PLSFeatureExtractor
from spectra_ml.splitting import nested_group_stratified_splits_multiclass
from spectra_ml.united_stage import CLASS_NAMES
from run_experiment_united_stage_v2 import CLASS_LABELS, RANDOM_SEED, load_modeling_set  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united_multiclass_v2", "interpretability")

# hypothesis bands from the OLD-only binary study (current_tasks), nm
HYPOTHESIS_BANDS = [(467.0, 500.0), (416.0, 425.0), (537.0, 561.0)]

# family_name -> name of the reduction step inside its Pipeline(reduce_step, logreg)
REDUCE_STEP_NAME = {"pca_mnlogreg": "pca", "pls_mnlogreg": "pls"}


def refit_best_per_fold(candidates, spectra, target, patient_id, waves, seed):
    nested = nested_group_stratified_splits_multiclass(patient_id, target, CLASS_LABELS, seed=seed)
    results = []
    for outer in nested:
        best, inner_mean, _ = _select_best_candidate_multiclass(candidates, spectra, target, patient_id, waves,
                                                                   outer, CLASS_LABELS)

        X_tr = spectra[outer.outer_train_idx]
        y_tr = target[outer.outer_train_idx]
        pid_tr = patient_id[outer.outer_train_idx]

        prep = build_preprocessing_pipeline(best.preprocessing_variant, wavelengths=waves)
        Xt_tr = prep.fit_transform(X_tr)
        model = best.build_fn()
        sw = patient_balanced_sample_weight(pid_tr)
        _fit_with_sample_weight(model, Xt_tr, y_tr, sw)

        results.append({"outer_fold": outer.outer_fold_index, "candidate": best, "preprocessing_pipeline": prep,
                         "model": model, "inner_cv_auc": inner_mean})
    return results


def hypothesis_band_hits(region_df: pd.DataFrame) -> list[dict]:
    hits = []
    for lo, hi in HYPOTHESIS_BANDS:
        sub = region_df[(region_df["wavelength_start"] <= hi) & (region_df["wavelength_end"] >= lo)]
        if len(sub):
            rank = region_df["sum_abs_importance"].rank(ascending=False)
            best_rank = int(rank.loc[sub.index].min())
            hits.append({"band_lo": lo, "band_hi": hi, "best_rank_among_regions": best_rank,
                         "n_regions_total": len(region_df)})
    return hits


def main():
    usd, spectra, target, patient_id, instrument = load_modeling_set()
    waves = usd.waves
    seed = RANDOM_SEED

    os.makedirs(OUT, exist_ok=True)

    families = {
        "pca_mnlogreg": build_pca_multinomial_logreg_candidates(seed=seed),
        "pls_mnlogreg": build_pls_multinomial_logreg_candidates(seed=seed),
    }

    summary_rows = []
    for family_name, candidates in families.items():
        reduce_step = REDUCE_STEP_NAME[family_name]
        t0 = time.time()
        fold_results = refit_best_per_fold(candidates, spectra, target, patient_id, waves, seed)
        print(f"[{family_name}] refit {len(fold_results)} folds in {time.time() - t0:.1f}s "
              f"(preprocessing chosen: {[r['candidate'].preprocessing_variant for r in fold_results]}, "
              f"n_components chosen: {[r['candidate'].params.get('n_components') for r in fold_results]})")

        for k, cls in enumerate(CLASS_LABELS):
            class_name = CLASS_NAMES[cls]
            coef_per_fold, backprojected_flags = [], []
            for r in fold_results:
                prep, model = r["preprocessing_pipeline"], r["model"]
                reducer = model.named_steps[reduce_step]  # PCAFeatureExtractor or PLSFeatureExtractor
                logreg = model.named_steps["logreg"]
                classes_ = list(logreg.classes_)
                row = classes_.index(cls)
                logreg_coef = np.asarray(logreg.coef_[row]).ravel()
                coef_in_prep_space = reducer.back_project_coefficients(logreg_coef)
                coef_bp, was_bp = back_project_through_scaler(coef_in_prep_space, prep)
                coef_per_fold.append(coef_bp)
                backprojected_flags.append(was_bp)

            coef_matrix = np.stack(coef_per_fold)  # (n_folds, n_wave)
            stability = fold_stability(coef_matrix, waves, top_k=20)
            stability["any_fold_not_backprojected"] = not all(backprojected_flags)
            stability.to_csv(os.path.join(OUT, f"coefficient_stability_{family_name}_class{int(cls)}_{class_name}.csv"),
                              index=False)

            region_importance = aggregate_wavelength_regions(coef_matrix.mean(axis=0), waves, region_size=5,
                                                               signed_values=coef_matrix.mean(axis=0))
            region_importance.to_csv(
                os.path.join(OUT, f"coefficient_region_importance_{family_name}_class{int(cls)}_{class_name}.csv"),
                index=False)

            hits = hypothesis_band_hits(region_importance)
            for h in hits:
                summary_rows.append({"family": family_name, "class": class_name, **h})
            print(f"  class {class_name} (vs rest): backprojected={all(backprojected_flags)}  "
                  f"hypothesis-band ranks: {hits}")

    band_summary = pd.DataFrame(summary_rows)
    band_summary.to_csv(os.path.join(OUT, "hypothesis_band_replication_summary.csv"), index=False)
    print(f"\n  -> {OUT}/coefficient_stability_<family>_class<k>_<name>.csv")
    print(f"  -> {OUT}/coefficient_region_importance_<family>_class<k>_<name>.csv")
    print(f"  -> {OUT}/hypothesis_band_replication_summary.csv")


if __name__ == "__main__":
    main()
