"""Stage 10: interpretability for the two linear-model families that actually
carry signal (linear_svm, pls_logreg) -- build_project_and_exp.md sections
22-25, 27-29. Model-agnostic (grouped permutation importance) and
family-specific (linear coefficients, PLS VIP + back-projection) methods,
both computed per outer fold and then aggregated for stability (section 29).

Every method here operates on a model already refit for one outer fold (using
that fold's already-selected hyperparameters/preprocessing from Stage 6-9 --
no new hyperparameter search happens here) and evaluates on that fold's own
held-out outer_test, so nothing here can leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from spectra_ml.evaluation import aggregate_to_patient
from spectra_ml.preprocessing import SpectralStandardScaler


def linear_coefficients_in_feature_space(fitted_model) -> np.ndarray:
    """LogisticRegression or LinearSVC -- both expose .coef_ (1, n_features)."""
    return np.asarray(fitted_model.coef_).ravel()


def back_project_through_scaler(coef: np.ndarray, fitted_preprocessing_pipeline) -> tuple[np.ndarray, bool]:
    """If the pipeline's final step is SpectralStandardScaler (a fixed per-wavelength
    z = (x - mean_)/std_), rescale coef by 1/std_ so it's in raw-intensity units
    (d(score)/dx = coef/std_). SNV (variant B, used by most winning folds here) is
    a PER-SPECTRUM normalization, not a fixed per-feature linear map, so there is
    no single matrix that undoes it -- in that case the coefficient stays in
    SNV-normalized-feature space and this returns backprojected=False so callers
    don't silently mislabel units."""
    last_step = fitted_preprocessing_pipeline.steps[-1][1]
    if isinstance(last_step, SpectralStandardScaler):
        return coef / (last_step.std_ + 1e-12), True
    return coef, False


def rank_biserial_like_rank(coef: np.ndarray) -> np.ndarray:
    """1-indexed rank by |coefficient|, largest first."""
    order = np.argsort(-np.abs(coef))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(coef) + 1)
    return ranks


def group_permutation_importance(fitted_preprocessing_pipeline, fitted_model, X_raw: np.ndarray,
                                  y: np.ndarray, patient_id: np.ndarray, waves: np.ndarray,
                                  group_size: int = 5, n_repeats: int = 20, seed: int = 42,
                                  has_decision_function: bool = False) -> pd.DataFrame:
    """Section 25/27: permute each contiguous wavelength GROUP (not single
    wavelengths, since neighbors are highly correlated) across PATIENTS on
    held-out data, and measure the drop in patient-level ROC-AUC. Permuting at
    the raw-spectrum level (before preprocessing) so any preprocessing that
    depends on the full row (SNV) is recomputed consistently with the permuted
    values, not bypassed."""
    rng = np.random.default_rng(seed)
    n_wave = X_raw.shape[1]
    group_starts = list(range(0, n_wave, group_size))

    def score(X):
        Xt = fitted_preprocessing_pipeline.transform(X)
        if has_decision_function:
            raw = 1 / (1 + np.exp(-fitted_model.decision_function(Xt)))
        else:
            raw = fitted_model.predict_proba(Xt)[:, 1]
        patient_df = aggregate_to_patient(raw, patient_id, y)
        return roc_auc_score(patient_df["target"], patient_df["probability"])

    baseline = score(X_raw)

    rows = []
    for start in group_starts:
        end = min(start + group_size, n_wave)
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            X_perm = X_raw.copy()
            perm_idx = rng.permutation(X_perm.shape[0])
            X_perm[:, start:end] = X_perm[perm_idx][:, start:end]
            drops[r] = baseline - score(X_perm)
        rows.append({
            "wavelength_start": float(waves[start]), "wavelength_end": float(waves[end - 1]),
            "importance_mean": float(drops.mean()), "importance_std": float(drops.std()),
            "baseline_auc": float(baseline),
        })
    return pd.DataFrame(rows)


def aggregate_wavelength_regions(values: np.ndarray, waves: np.ndarray, region_size: int = 5,
                                  signed_values: np.ndarray | None = None) -> pd.DataFrame:
    """Group point-wise |values| (e.g. |coefficient| or mean_abs_attribution) into
    contiguous wavelength regions (section 28)."""
    n = len(values)
    rows = []
    for start in range(0, n, region_size):
        end = min(start + region_size, n)
        rows.append({
            "wavelength_start": float(waves[start]), "wavelength_end": float(waves[end - 1]),
            "sum_abs_importance": float(np.sum(np.abs(values[start:end]))),
            "mean_signed": float(np.mean(signed_values[start:end])) if signed_values is not None else np.nan,
        })
    return pd.DataFrame(rows)


def fold_stability(per_fold_values: np.ndarray, waves: np.ndarray, top_k: int = 20) -> pd.DataFrame:
    """per_fold_values: (n_folds, n_wave) array (e.g. |coefficient| per fold, after
    aligning to a common wavelength grid). Section 29: mean/std, 95% CI across
    folds, and the fraction of folds where each wavelength is in the top-K by
    |value|."""
    n_folds, n_wave = per_fold_values.shape
    top_k_masks = np.zeros_like(per_fold_values, dtype=bool)
    for f in range(n_folds):
        top_idx = np.argsort(-np.abs(per_fold_values[f]))[:top_k]
        top_k_masks[f, top_idx] = True

    return pd.DataFrame({
        "wavelength": waves,
        "mean": per_fold_values.mean(axis=0),
        "std": per_fold_values.std(axis=0),
        "ci95_lo": np.percentile(per_fold_values, 2.5, axis=0),
        "ci95_hi": np.percentile(per_fold_values, 97.5, axis=0),
        "fraction_folds_in_topK": top_k_masks.mean(axis=0),
    })
