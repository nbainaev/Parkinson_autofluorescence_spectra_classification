"""Multiclass (N>2 classes, e.g. Hoehn-Yahr PD stage 0-3) nested-CV experiment
runner. Additive sibling of spectra_ml.experiment (which is binary-only: single
probability column, binary roc_auc_score, sigmoid-squashed decision_function).
Reuses spectra_ml.experiment's generic (class-count-agnostic) helpers directly
(`_fit_with_sample_weight`) rather than reimplementing them -- only the parts
that are genuinely shaped by "how many classes" (probability matrix instead of
a single column, macro OvR AUC for model selection, multiclass metrics/
calibration) are new here.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from spectra_ml.calibration import PlattCalibrator
from spectra_ml.evaluation import aggregate_to_patient_multiclass, compute_patient_metrics_multiclass, macro_ovr_auc
from spectra_ml.evaluation import patient_balanced_sample_weight
from spectra_ml.experiment import _fit_with_sample_weight
from spectra_ml.models.base import ModelCandidate
from spectra_ml.preprocessing import build_preprocessing_pipeline
from spectra_ml.splitting import Fold, NestedFold, nested_group_stratified_splits_multiclass


def _fit_predict_proba_multiclass(candidate: ModelCandidate, X_train, y_train, pid_train, X_eval, waves,
                                   class_labels):
    """Every multiclass candidate in spectra_ml.models.classical's multiclass
    pool exposes predict_proba (native for mn_logreg/pls_mnlogreg/pca_mnlogreg/
    random_forest/hgb/knn, via GroupAwareCalibratedLinearSVC for linear_svm_mc*)
    EXCEPT the rbf_svm family (united_multiclass_v2 task 2 extension), which has
    no predict_proba without a CalibratedClassifierCV wrapper (deliberately not
    added -- calibration is disabled this round). For has_decision_function
    candidates only, mirror spectra_ml.experiment._fit_predict_scores's binary
    convention: take the raw decision_function (shape (n, n_classes) for a
    multiclass estimator with decision_function_shape='ovr') and squash each
    column independently with a sigmoid -- a rank-preserving [0,1] transform
    used ONLY for macro OvR-AUC ranking/model-selection and for the multiclass
    metrics that need a "probability"-shaped input (Brier/log loss), NOT a
    calibrated probability (columns need not sum to 1)."""
    prep = build_preprocessing_pipeline(candidate.preprocessing_variant, wavelengths=waves)
    Xt_train = prep.fit_transform(X_train)
    Xt_eval = prep.transform(X_eval)

    model = candidate.build_fn()
    sw = patient_balanced_sample_weight(pid_train)
    _fit_with_sample_weight(model, Xt_train, y_train, sw, patient_id=pid_train,
                             requires_patient_groups=candidate.requires_patient_groups)

    classes_ = list(model.classes_)
    if getattr(candidate, "has_decision_function", False):
        raw = model.decision_function(Xt_eval)
        raw = np.asarray(raw)
        if raw.ndim == 1:  # degenerate 2-class decision_function edge case, not expected for this 4-class task
            raw = np.column_stack([-raw, raw])
        probs = 1.0 / (1.0 + np.exp(-raw))
    else:
        probs = model.predict_proba(Xt_eval)
    # reindex columns to the canonical class_labels order (model.classes_ should
    # already be sorted ascending for numeric labels, but don't assume it silently)
    col_order = [classes_.index(c) for c in class_labels]
    probs = probs[:, col_order]
    return probs, model


def _select_best_candidate_multiclass(candidates: list[ModelCandidate], spectra, target, patient_id, waves,
                                       outer: NestedFold, class_labels) -> tuple[ModelCandidate, float, list[float]]:
    best_candidate, best_mean_score, best_inner_scores = None, -np.inf, []
    for candidate in candidates:
        inner_scores = []
        for inner in outer.inner_folds:
            X_tr, y_tr, pid_tr = spectra[inner.train_idx], target[inner.train_idx], patient_id[inner.train_idx]
            X_val, y_val = spectra[inner.test_idx], target[inner.test_idx]
            pid_val = patient_id[inner.test_idx]
            probs, _ = _fit_predict_proba_multiclass(candidate, X_tr, y_tr, pid_tr, X_val, waves, class_labels)
            patient_df = aggregate_to_patient_multiclass(probs, pid_val, y_val, class_labels)
            prob_cols = [f"prob_{c}" for c in class_labels]
            inner_scores.append(macro_ovr_auc(patient_df["target"].to_numpy(), patient_df[prob_cols].to_numpy(),
                                               class_labels))
        mean_score = float(np.mean(inner_scores))
        if mean_score > best_mean_score:
            best_candidate, best_mean_score, best_inner_scores = candidate, mean_score, inner_scores
    return best_candidate, best_mean_score, best_inner_scores


def run_family_nested_cv_multiclass(family_name: str, candidates: list[ModelCandidate], spectra: np.ndarray,
                                     target: np.ndarray, patient_id: np.ndarray, waves: np.ndarray, class_labels,
                                     seed: int = 42, outer_n_splits: int | None = None,
                                     inner_n_splits: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert all(c.family == family_name for c in candidates), "all candidates must belong to family_name"
    class_labels = list(class_labels)
    prob_cols = [f"prob_{c}" for c in class_labels]

    nested = nested_group_stratified_splits_multiclass(patient_id, target, class_labels, outer_n_splits,
                                                          inner_n_splits, seed=seed)

    pred_rows, summary_rows = [], []
    for outer in nested:
        if len(candidates) == 1:
            best, inner_mean, inner_scores = candidates[0], np.nan, []
        else:
            best, inner_mean, inner_scores = _select_best_candidate_multiclass(
                candidates, spectra, target, patient_id, waves, outer, class_labels)

        X_tr = spectra[outer.outer_train_idx]
        y_tr = target[outer.outer_train_idx]
        pid_tr = patient_id[outer.outer_train_idx]
        X_te = spectra[outer.outer_test_idx]
        y_te = target[outer.outer_test_idx]
        pid_te = patient_id[outer.outer_test_idx]

        probs_te, _ = _fit_predict_proba_multiclass(best, X_tr, y_tr, pid_tr, X_te, waves, class_labels)
        patient_df = aggregate_to_patient_multiclass(probs_te, pid_te, y_te, class_labels)
        patient_df["outer_fold"] = outer.outer_fold_index
        patient_df["family"] = family_name
        patient_df["model"] = best.name
        patient_df["preprocessing"] = best.preprocessing_variant
        patient_df.rename(columns={c: f"raw_{c}" for c in prob_cols}, inplace=True)
        pred_rows.append(patient_df)

        metrics = compute_patient_metrics_multiclass(patient_df["target"].to_numpy(),
                                                       patient_df[[f"raw_{c}" for c in prob_cols]].to_numpy(),
                                                       class_labels)
        summary_rows.append({
            "family": family_name, "outer_fold": outer.outer_fold_index, "chosen_model": best.name,
            "preprocessing": best.preprocessing_variant, "chosen_params": best.params,
            "inner_cv_mean_auc": inner_mean, "n_inner_folds": len(inner_scores),
            **{k: v for k, v in metrics.items() if k not in ("per_class_precision", "per_class_recall",
                                                                "per_class_f1", "per_class_support",
                                                                "confusion_matrix", "confusion_matrix_labels")},
            "per_class_precision": metrics["per_class_precision"], "per_class_recall": metrics["per_class_recall"],
            "per_class_f1": metrics["per_class_f1"], "per_class_support": metrics["per_class_support"],
            "confusion_matrix": metrics["confusion_matrix"],
        })

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(summary_rows)


def run_all_families_nested_cv_multiclass(candidate_pool: list[ModelCandidate], spectra: np.ndarray,
                                           target: np.ndarray, patient_id: np.ndarray, waves: np.ndarray,
                                           class_labels, seed: int = 42, outer_n_splits: int | None = None,
                                           inner_n_splits: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_family: dict[str, list[ModelCandidate]] = defaultdict(list)
    for c in candidate_pool:
        by_family[c.family].append(c)

    all_preds, all_summaries = [], []
    for family_name, candidates in by_family.items():
        preds, summary = run_family_nested_cv_multiclass(family_name, candidates, spectra, target, patient_id,
                                                            waves, class_labels, seed=seed,
                                                            outer_n_splits=outer_n_splits,
                                                            inner_n_splits=inner_n_splits)
        all_preds.append(preds)
        all_summaries.append(summary)

    return pd.concat(all_preds, ignore_index=True), pd.concat(all_summaries, ignore_index=True)


def _oof_patient_scores_multiclass(candidate: ModelCandidate, spectra, target, patient_id, waves, class_labels,
                                    inner_folds: list[Fold]) -> pd.DataFrame:
    rows = []
    for inner in inner_folds:
        X_tr, y_tr, pid_tr = spectra[inner.train_idx], target[inner.train_idx], patient_id[inner.train_idx]
        X_val, y_val, pid_val = spectra[inner.test_idx], target[inner.test_idx], patient_id[inner.test_idx]
        probs, _ = _fit_predict_proba_multiclass(candidate, X_tr, y_tr, pid_tr, X_val, waves, class_labels)
        rows.append(aggregate_to_patient_multiclass(probs, pid_val, y_val, class_labels))
    return pd.concat(rows, ignore_index=True)


def _calibrate_multiclass_ovr(oof: pd.DataFrame, class_labels) -> dict:
    """Per-class one-vs-rest Platt calibration (documented choice: a full
    joint softmax/Dirichlet calibration is nontrivial and not needed for this
    scope): fit one PlattCalibrator per class on that class's OOF raw
    probability vs (target==class) indicator, then renormalize the resulting
    per-class calibrated probabilities to sum to 1 across classes so the
    output stays a valid probability simplex."""
    calibrators = {}
    for c in class_labels:
        raw = oof[f"prob_{c}"].to_numpy()
        y_bin = (oof["target"].to_numpy() == c).astype(int)
        calibrators[c] = PlattCalibrator().fit(raw, y_bin)
    return calibrators


def _apply_multiclass_calibration(probs: np.ndarray, calibrators: dict, class_labels) -> np.ndarray:
    cal = np.column_stack([calibrators[c].predict_proba(probs[:, k]) for k, c in enumerate(class_labels)])
    row_sum = cal.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    return cal / row_sum


def run_family_nested_cv_multiclass_calibrated(family_name: str, candidates: list[ModelCandidate],
                                                spectra: np.ndarray, target: np.ndarray, patient_id: np.ndarray,
                                                waves: np.ndarray, class_labels, seed: int = 42,
                                                outer_n_splits: int | None = None,
                                                inner_n_splits: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert all(c.family == family_name for c in candidates), "all candidates must belong to family_name"
    class_labels = list(class_labels)
    prob_cols = [f"prob_{c}" for c in class_labels]

    nested = nested_group_stratified_splits_multiclass(patient_id, target, class_labels, outer_n_splits,
                                                          inner_n_splits, seed=seed)

    pred_rows, summary_rows = [], []
    for outer in nested:
        if len(candidates) == 1:
            best, inner_mean, inner_scores = candidates[0], np.nan, []
        else:
            best, inner_mean, inner_scores = _select_best_candidate_multiclass(
                candidates, spectra, target, patient_id, waves, outer, class_labels)

        oof = _oof_patient_scores_multiclass(best, spectra, target, patient_id, waves, class_labels,
                                              outer.inner_folds)
        calibrators = _calibrate_multiclass_ovr(oof, class_labels)

        X_tr = spectra[outer.outer_train_idx]
        y_tr = target[outer.outer_train_idx]
        pid_tr = patient_id[outer.outer_train_idx]
        X_te = spectra[outer.outer_test_idx]
        y_te = target[outer.outer_test_idx]
        pid_te = patient_id[outer.outer_test_idx]

        probs_te, _ = _fit_predict_proba_multiclass(best, X_tr, y_tr, pid_tr, X_te, waves, class_labels)
        patient_df = aggregate_to_patient_multiclass(probs_te, pid_te, y_te, class_labels)
        raw_probs = patient_df[prob_cols].to_numpy()
        cal_probs = _apply_multiclass_calibration(raw_probs, calibrators, class_labels)

        patient_df.rename(columns={c: f"raw_{c}" for c in prob_cols}, inplace=True)
        for k, c in enumerate(class_labels):
            patient_df[f"calibrated_prob_{c}"] = cal_probs[:, k]
        patient_df["outer_fold"] = outer.outer_fold_index
        patient_df["family"] = family_name
        patient_df["model"] = best.name
        patient_df["preprocessing"] = best.preprocessing_variant
        pred_rows.append(patient_df)

        raw_metrics = compute_patient_metrics_multiclass(patient_df["target"].to_numpy(), raw_probs, class_labels)
        cal_metrics = compute_patient_metrics_multiclass(patient_df["target"].to_numpy(), cal_probs, class_labels)
        plain_keys = ["n_patients", "roc_auc_macro_ovr", "balanced_accuracy", "macro_f1", "brier_multiclass",
                      "log_loss"]
        summary_rows.append({
            "family": family_name, "outer_fold": outer.outer_fold_index, "chosen_model": best.name,
            "preprocessing": best.preprocessing_variant, "chosen_params": best.params,
            "inner_cv_mean_auc": inner_mean, "n_inner_folds": len(inner_scores), "n_oof_patients": len(oof),
            **{f"raw_{k}": raw_metrics[k] for k in plain_keys},
            **{f"calibrated_{k}": cal_metrics[k] for k in plain_keys},
            "raw_confusion_matrix": raw_metrics["confusion_matrix"],
            "calibrated_confusion_matrix": cal_metrics["confusion_matrix"],
            "calibrated_per_class_recall": cal_metrics["per_class_recall"],
            "calibrated_per_class_precision": cal_metrics["per_class_precision"],
        })

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(summary_rows)


def run_all_families_nested_cv_multiclass_calibrated(candidate_pool: list[ModelCandidate], spectra: np.ndarray,
                                                       target: np.ndarray, patient_id: np.ndarray,
                                                       waves: np.ndarray, class_labels, seed: int = 42,
                                                       outer_n_splits: int | None = None,
                                                       inner_n_splits: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_family: dict[str, list[ModelCandidate]] = defaultdict(list)
    for c in candidate_pool:
        by_family[c.family].append(c)

    all_preds, all_summaries = [], []
    for family_name, candidates in by_family.items():
        preds, summary = run_family_nested_cv_multiclass_calibrated(
            family_name, candidates, spectra, target, patient_id, waves, class_labels, seed=seed,
            outer_n_splits=outer_n_splits, inner_n_splits=inner_n_splits)
        all_preds.append(preds)
        all_summaries.append(summary)

    return pd.concat(all_preds, ignore_index=True), pd.concat(all_summaries, ignore_index=True)
