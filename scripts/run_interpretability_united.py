"""United-dataset interpretability entry point: linear_svm and pls_logreg
coefficient-in-wavelength-space + grouped permutation importance, on the
pooled harmonized 3-instrument dataset. Mirrors scripts/run_interpretability.py.

Usage: python scripts/run_interpretability_united.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from spectra_ml.experiment import _fit_with_sample_weight, _select_best_candidate
from spectra_ml.evaluation import patient_balanced_sample_weight
from spectra_ml.interpretability import (aggregate_wavelength_regions, back_project_through_scaler,
                                          fold_stability, group_permutation_importance,
                                          linear_coefficients_in_feature_space)
from spectra_ml.plotting import plot_coefficient_stability, plot_permutation_importance
from spectra_ml.models.classical import build_linear_svm_candidates, build_pls_logreg_candidates
from spectra_ml.preprocessing import build_preprocessing_pipeline
from spectra_ml.reduction import PLSFeatureExtractor
from spectra_ml.splitting import nested_group_stratified_splits
from run_experiment_united import RANDOM_SEED, load_modeling_set  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united", "interpretability")


def refit_best_per_fold(family_name, candidates, spectra, target, patient_id, waves, seed):
    nested = nested_group_stratified_splits(patient_id, target, seed=seed)
    results = []
    for outer in nested:
        best, inner_mean, _ = _select_best_candidate(candidates, spectra, target, patient_id, waves, outer)

        X_tr = spectra[outer.outer_train_idx]
        y_tr = target[outer.outer_train_idx]
        pid_tr = patient_id[outer.outer_train_idx]
        X_te = spectra[outer.outer_test_idx]
        y_te = target[outer.outer_test_idx]
        pid_te = patient_id[outer.outer_test_idx]

        prep = build_preprocessing_pipeline(best.preprocessing_variant, wavelengths=waves)
        Xt_tr = prep.fit_transform(X_tr)
        model = best.build_fn()
        sw = patient_balanced_sample_weight(pid_tr)
        _fit_with_sample_weight(model, Xt_tr, y_tr, sw)

        results.append({
            "outer_fold": outer.outer_fold_index, "candidate": best, "preprocessing_pipeline": prep,
            "model": model, "X_test_raw": X_te, "y_test": y_te, "pid_test": pid_te,
            "inner_cv_auc": inner_mean,
        })
    return results


def main():
    ud, spectra, target, patient_id, instrument = load_modeling_set()
    waves = ud.waves
    seed = RANDOM_SEED

    os.makedirs(OUT, exist_ok=True)
    fig_dir = os.path.join(OUT, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    families = {
        "linear_svm": build_linear_svm_candidates(seed=seed),
        "pls_logreg": build_pls_logreg_candidates(seed=seed),
    }

    for family_name, candidates in families.items():
        t0 = time.time()
        fold_results = refit_best_per_fold(family_name, candidates, spectra, target, patient_id, waves, seed)
        print(f"[{family_name}] refit {len(fold_results)} folds in {time.time() - t0:.1f}s "
              f"(preprocessing chosen: {[r['candidate'].preprocessing_variant for r in fold_results]})")

        coef_per_fold, backprojected_flags = [], []
        perm_rows = []
        vip_per_fold = []

        for r in fold_results:
            prep, model = r["preprocessing_pipeline"], r["model"]

            if family_name == "linear_svm":
                coef = linear_coefficients_in_feature_space(model)
                coef_bp, was_bp = back_project_through_scaler(coef, prep)
            else:  # pls_logreg: Pipeline(pls, logreg)
                pls: PLSFeatureExtractor = model.named_steps["pls"]
                logreg_coef = linear_coefficients_in_feature_space(model.named_steps["logreg"])
                coef_in_prep_space = pls.back_project_coefficients(logreg_coef)
                coef_bp, was_bp = back_project_through_scaler(coef_in_prep_space, prep)
                vip_per_fold.append(pls.vip_scores())

            coef_per_fold.append(coef_bp)
            backprojected_flags.append(was_bp)

            has_df = r["candidate"].has_decision_function
            imp = group_permutation_importance(prep, model, r["X_test_raw"], r["y_test"], r["pid_test"], waves,
                                                group_size=5, n_repeats=15, seed=seed + r["outer_fold"],
                                                has_decision_function=has_df)
            imp["outer_fold"] = r["outer_fold"]
            perm_rows.append(imp)

        coef_matrix = np.stack(coef_per_fold)  # (n_folds, n_wave)
        stability = fold_stability(coef_matrix, waves, top_k=20)
        stability["any_fold_not_backprojected"] = not all(backprojected_flags)
        stability.to_csv(os.path.join(OUT, f"coefficient_stability_{family_name}.csv"), index=False)

        region_importance = aggregate_wavelength_regions(coef_matrix.mean(axis=0), waves, region_size=5,
                                                           signed_values=coef_matrix.mean(axis=0))
        region_importance.to_csv(os.path.join(OUT, f"coefficient_region_importance_{family_name}.csv"),
                                  index=False)

        if vip_per_fold:
            vip_matrix = np.stack(vip_per_fold)
            vip_stability = fold_stability(vip_matrix, waves, top_k=20)
            vip_stability.to_csv(os.path.join(OUT, f"vip_stability_{family_name}.csv"), index=False)

        perm_df = pd.concat(perm_rows, ignore_index=True)
        perm_df.to_csv(os.path.join(OUT, f"permutation_importance_{family_name}.csv"), index=False)
        perm_stability = perm_df.groupby(["wavelength_start", "wavelength_end"])["importance_mean"].agg(
            ["mean", "std", "count"]).reset_index()
        perm_stability.to_csv(os.path.join(OUT, f"permutation_importance_stability_{family_name}.csv"),
                               index=False)

        plot_coefficient_stability(stability, os.path.join(fig_dir, f"coefficient_stability_{family_name}.png"),
                                    family=family_name, backprojected=all(backprojected_flags))
        plot_permutation_importance(perm_stability, os.path.join(fig_dir, f"permutation_importance_{family_name}.png"),
                                     family=family_name)

        print(f"  backprojected to raw units: {all(backprojected_flags)} "
              f"(variant chosen per fold: {[r['candidate'].preprocessing_variant for r in fold_results]})")
        print(f"  -> {OUT}/coefficient_stability_{family_name}.csv")
        print(f"  -> {OUT}/permutation_importance_{family_name}.csv")


if __name__ == "__main__":
    main()
