"""Stage 6+: nested-CV experiment runner for classical models (build_project_and_exp.md
sections 9-13, 32). For each model FAMILY (dummy/logreg/linear_svm/pls_logreg/...):
for each outer fold, use that family's inner folds to pick the best
(preprocessing variant, hyperparameters) by mean inner patient-level ROC-AUC,
refit on the full outer_train, evaluate once on the untouched outer_test.
Families are evaluated independently so section 30's model-vs-model comparison
table is apples-to-apples (same outer folds, each family gets its own model
selection).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from spectra_ml.calibration import PlattCalibrator
from spectra_ml.evaluation import aggregate_to_patient, compute_patient_metrics, patient_balanced_sample_weight
from spectra_ml.models.base import ModelCandidate
from spectra_ml.preprocessing import build_preprocessing_pipeline
from spectra_ml.splitting import Fold, NestedFold, nested_group_stratified_splits


def _fit_with_sample_weight(model, X, y, sample_weight, patient_id=None, requires_patient_groups=False):
    """A bare estimator takes `sample_weight=` directly; a sklearn Pipeline needs it
    routed to its final step (`<stepname>__sample_weight=`); some estimators (e.g.
    PLSRegression, used inside the pls_logreg Pipeline via PLSFeatureExtractor)
    don't support it at all -- fall back to unweighted fit rather than dropping
    the whole candidate. Torch models (MLP/CNN) additionally take `patient_id=`
    for their own internal group-aware early-stopping split (section 14: that
    split must never be, or derive from, the outer_test)."""
    from sklearn.pipeline import Pipeline

    if requires_patient_groups:
        model.fit(X, y, sample_weight=sample_weight, patient_id=patient_id)
        return
    if isinstance(model, Pipeline):
        last_step = model.steps[-1][0]
        try:
            model.fit(X, y, **{f"{last_step}__sample_weight": sample_weight})
            return
        except (TypeError, ValueError):
            model.fit(X, y)
            return
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)


def _fit_predict_scores(candidate: ModelCandidate, X_train, y_train, pid_train, X_eval, waves):
    prep = build_preprocessing_pipeline(candidate.preprocessing_variant, wavelengths=waves)
    Xt_train = prep.fit_transform(X_train)
    Xt_eval = prep.transform(X_eval)

    model = candidate.build_fn()
    sw = patient_balanced_sample_weight(pid_train)
    _fit_with_sample_weight(model, Xt_train, y_train, sw, patient_id=pid_train,
                             requires_patient_groups=candidate.requires_patient_groups)

    if candidate.has_decision_function:
        scores = model.decision_function(Xt_eval)
        # rank-preserving squash to [0,1] for metrics that need a "probability"-shaped
        # input (Brier/log loss); NOT a calibrated probability -- that's Stage 8.
        scores = 1 / (1 + np.exp(-scores))
    else:
        scores = model.predict_proba(Xt_eval)[:, 1]
    return scores, model


def _select_best_candidate(candidates: list[ModelCandidate], spectra, target, patient_id, waves,
                            outer: NestedFold) -> tuple[ModelCandidate, float, list[float]]:
    best_candidate, best_mean_score, best_inner_scores = None, -np.inf, []
    for candidate in candidates:
        inner_scores = []
        for inner in outer.inner_folds:
            X_tr, y_tr, pid_tr = spectra[inner.train_idx], target[inner.train_idx], patient_id[inner.train_idx]
            X_val, y_val = spectra[inner.test_idx], target[inner.test_idx]
            pid_val = patient_id[inner.test_idx]
            scores, _ = _fit_predict_scores(candidate, X_tr, y_tr, pid_tr, X_val, waves)
            patient_df = aggregate_to_patient(scores, pid_val, y_val)
            inner_scores.append(roc_auc_score(patient_df["target"], patient_df["probability"]))
        mean_score = float(np.mean(inner_scores))
        if mean_score > best_mean_score:
            best_candidate, best_mean_score, best_inner_scores = candidate, mean_score, inner_scores
    return best_candidate, best_mean_score, best_inner_scores


def run_family_nested_cv(family_name: str, candidates: list[ModelCandidate], spectra: np.ndarray,
                          target: np.ndarray, patient_id: np.ndarray, waves: np.ndarray,
                          seed: int = 42, outer_n_splits: int | None = None,
                          inner_n_splits: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (patient_predictions, fold_summary) for one model family, evaluated
    across all outer folds."""
    assert all(c.family == family_name for c in candidates), "all candidates must belong to family_name"

    nested = nested_group_stratified_splits(patient_id, target, outer_n_splits, inner_n_splits, seed=seed)

    pred_rows, summary_rows = [], []
    for outer in nested:
        if len(candidates) == 1:
            best, inner_mean, inner_scores = candidates[0], np.nan, []
        else:
            best, inner_mean, inner_scores = _select_best_candidate(candidates, spectra, target, patient_id,
                                                                       waves, outer)

        X_tr = spectra[outer.outer_train_idx]
        y_tr = target[outer.outer_train_idx]
        pid_tr = patient_id[outer.outer_train_idx]
        X_te = spectra[outer.outer_test_idx]
        y_te = target[outer.outer_test_idx]
        pid_te = patient_id[outer.outer_test_idx]

        scores_te, _ = _fit_predict_scores(best, X_tr, y_tr, pid_tr, X_te, waves)
        patient_df = aggregate_to_patient(scores_te, pid_te, y_te)
        patient_df["outer_fold"] = outer.outer_fold_index
        patient_df["family"] = family_name
        patient_df["model"] = best.name
        patient_df["preprocessing"] = best.preprocessing_variant
        patient_df.rename(columns={"probability": "raw_probability"}, inplace=True)
        pred_rows.append(patient_df)

        metrics = compute_patient_metrics(patient_df["target"].to_numpy(), patient_df["raw_probability"].to_numpy())
        summary_rows.append({
            "family": family_name, "outer_fold": outer.outer_fold_index, "chosen_model": best.name,
            "preprocessing": best.preprocessing_variant, "chosen_params": best.params,
            "inner_cv_mean_auc": inner_mean, "n_inner_folds": len(inner_scores),
            **metrics,
        })

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(summary_rows)


def run_all_families_nested_cv(candidate_pool: list[ModelCandidate], spectra: np.ndarray, target: np.ndarray,
                                patient_id: np.ndarray, waves: np.ndarray, seed: int = 42,
                                outer_n_splits: int | None = None,
                                inner_n_splits: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_family: dict[str, list[ModelCandidate]] = defaultdict(list)
    for c in candidate_pool:
        by_family[c.family].append(c)

    all_preds, all_summaries = [], []
    for family_name, candidates in by_family.items():
        preds, summary = run_family_nested_cv(family_name, candidates, spectra, target, patient_id, waves,
                                                seed=seed, outer_n_splits=outer_n_splits,
                                                inner_n_splits=inner_n_splits)
        all_preds.append(preds)
        all_summaries.append(summary)

    return pd.concat(all_preds, ignore_index=True), pd.concat(all_summaries, ignore_index=True)


def _oof_patient_scores(candidate: ModelCandidate, spectra, target, patient_id, waves,
                         inner_folds: list[Fold]) -> pd.DataFrame:
    """Group-aware out-of-fold patient-level scores for ONE fixed candidate across
    outer_train (section 18): each inner fold's held-out patients get a score from
    a model trained only on the other inner folds. Concatenating across inner
    folds covers every outer_train patient exactly once."""
    rows = []
    for inner in inner_folds:
        X_tr, y_tr, pid_tr = spectra[inner.train_idx], target[inner.train_idx], patient_id[inner.train_idx]
        X_val, y_val, pid_val = spectra[inner.test_idx], target[inner.test_idx], patient_id[inner.test_idx]
        scores, _ = _fit_predict_scores(candidate, X_tr, y_tr, pid_tr, X_val, waves)
        rows.append(aggregate_to_patient(scores, pid_val, y_val))
    return pd.concat(rows, ignore_index=True)


def run_family_nested_cv_calibrated(family_name: str, candidates: list[ModelCandidate], spectra: np.ndarray,
                                     target: np.ndarray, patient_id: np.ndarray, waves: np.ndarray,
                                     seed: int = 42, outer_n_splits: int | None = None,
                                     inner_n_splits: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 8 version of run_family_nested_cv: per outer fold, after selecting
    hyperparameters via inner CV (as before), also use those SAME inner folds to
    build OOF patient-level scores for the winning candidate over the whole
    outer_train, fit a Platt calibrator on that OOF table, refit the winning
    candidate on the full outer_train, predict outer_test, and apply the
    calibrator. The calibrator never sees outer_test labels (section 18)."""
    assert all(c.family == family_name for c in candidates), "all candidates must belong to family_name"

    nested = nested_group_stratified_splits(patient_id, target, outer_n_splits, inner_n_splits, seed=seed)

    pred_rows, summary_rows = [], []
    for outer in nested:
        if len(candidates) == 1:
            best, inner_mean, inner_scores = candidates[0], np.nan, []
        else:
            best, inner_mean, inner_scores = _select_best_candidate(candidates, spectra, target, patient_id,
                                                                       waves, outer)

        oof = _oof_patient_scores(best, spectra, target, patient_id, waves, outer.inner_folds)
        calibrator = PlattCalibrator().fit(oof["probability"].to_numpy(), oof["target"].to_numpy())

        X_tr = spectra[outer.outer_train_idx]
        y_tr = target[outer.outer_train_idx]
        pid_tr = patient_id[outer.outer_train_idx]
        X_te = spectra[outer.outer_test_idx]
        y_te = target[outer.outer_test_idx]
        pid_te = patient_id[outer.outer_test_idx]

        scores_te, _ = _fit_predict_scores(best, X_tr, y_tr, pid_tr, X_te, waves)
        patient_df = aggregate_to_patient(scores_te, pid_te, y_te)
        patient_df.rename(columns={"probability": "raw_probability"}, inplace=True)
        patient_df["calibrated_probability"] = calibrator.predict_proba(patient_df["raw_probability"].to_numpy())
        patient_df["outer_fold"] = outer.outer_fold_index
        patient_df["family"] = family_name
        patient_df["model"] = best.name
        patient_df["preprocessing"] = best.preprocessing_variant
        pred_rows.append(patient_df)

        raw_metrics = compute_patient_metrics(patient_df["target"].to_numpy(), patient_df["raw_probability"].to_numpy())
        cal_metrics = compute_patient_metrics(patient_df["target"].to_numpy(), patient_df["calibrated_probability"].to_numpy())
        summary_rows.append({
            "family": family_name, "outer_fold": outer.outer_fold_index, "chosen_model": best.name,
            "preprocessing": best.preprocessing_variant, "chosen_params": best.params,
            "inner_cv_mean_auc": inner_mean, "n_inner_folds": len(inner_scores), "n_oof_patients": len(oof),
            **{f"raw_{k}": v for k, v in raw_metrics.items()},
            **{f"calibrated_{k}": v for k, v in cal_metrics.items()},
        })

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(summary_rows)


def run_all_families_nested_cv_calibrated(candidate_pool: list[ModelCandidate], spectra: np.ndarray,
                                           target: np.ndarray, patient_id: np.ndarray, waves: np.ndarray,
                                           seed: int = 42, outer_n_splits: int | None = None,
                                           inner_n_splits: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_family: dict[str, list[ModelCandidate]] = defaultdict(list)
    for c in candidate_pool:
        by_family[c.family].append(c)

    all_preds, all_summaries = [], []
    for family_name, candidates in by_family.items():
        preds, summary = run_family_nested_cv_calibrated(family_name, candidates, spectra, target, patient_id,
                                                            waves, seed=seed, outer_n_splits=outer_n_splits,
                                                            inner_n_splits=inner_n_splits)
        all_preds.append(preds)
        all_summaries.append(summary)

    return pd.concat(all_preds, ignore_index=True), pd.concat(all_summaries, ignore_index=True)


# ---------------------------------------------------------------------------
# double_blind_exp.md Parts XX-XXII: "fit once on all development data" path.
# Selects hyperparameters via a single (non-nested) group-stratified CV over
# the WHOLE development set, fits an OOF Platt calibrator on those same folds
# for the winning candidate, then refits the winning candidate's preprocessing
# + model on ALL development data -- this refit model is what gets frozen and
# is the only thing predict_blinded.py ever calls.
# ---------------------------------------------------------------------------

def select_calibrate_and_refit(family_name: str, candidates: list[ModelCandidate], spectra: np.ndarray,
                                target: np.ndarray, patient_id: np.ndarray, waves: np.ndarray, seed: int = 42,
                                calibrate: bool = True, n_splits: int | None = None) -> dict:
    """Development-only model freeze for one family (no outer_test/held-out
    generalization estimate is produced here -- that estimate, if wanted for
    reporting, should come from spectra_ml.experiment.run_family_nested_cv*
    run separately; THIS function's only job is to pick one final
    hyperparameter configuration and produce one final fitted pipeline).

    Returns a dict: {family, chosen_candidate, preprocessing (fitted on ALL
    development data), model (fitted on ALL development data), calibrator
    (PlattCalibrator fit on OOF development predictions, or None if
    calibrate=False), oof_patient_predictions (DataFrame: patient_id, target,
    raw_probability[, calibrated_probability]), selection_mean_auc,
    selection_fold_aucs, cv_folds_used}.
    """
    from spectra_ml.splitting import Fold, NestedFold, group_stratified_splits

    folds = group_stratified_splits(patient_id, target, n_splits=n_splits, seed=seed)
    fake_outer = NestedFold(outer_fold_index=-1, outer_train_idx=np.arange(len(patient_id)),
                             outer_test_idx=np.array([], dtype=int), inner_folds=folds)

    if len(candidates) == 1:
        best, mean_auc, fold_aucs = candidates[0], np.nan, []
    else:
        best, mean_auc, fold_aucs = _select_best_candidate(candidates, spectra, target, patient_id, waves,
                                                             fake_outer)

    oof = _oof_patient_scores(best, spectra, target, patient_id, waves, folds)
    oof = oof.rename(columns={"probability": "raw_probability"})

    calibrator = None
    if calibrate:
        calibrator = PlattCalibrator().fit(oof["raw_probability"].to_numpy(), oof["target"].to_numpy())
        oof["calibrated_probability"] = calibrator.predict_proba(oof["raw_probability"].to_numpy())

    # -- refit chosen candidate's preprocessing + model on ALL development data --
    prep = build_preprocessing_pipeline(best.preprocessing_variant, wavelengths=waves)
    Xt = prep.fit_transform(spectra)
    model = best.build_fn()
    sw = patient_balanced_sample_weight(patient_id)
    _fit_with_sample_weight(model, Xt, target, sw, patient_id=patient_id,
                             requires_patient_groups=best.requires_patient_groups)

    return {
        "family": family_name, "chosen_candidate": best, "preprocessing": prep, "model": model,
        "calibrator": calibrator, "oof_patient_predictions": oof, "selection_mean_auc": mean_auc,
        "selection_fold_aucs": fold_aucs, "cv_folds_used": len(folds),
    }
