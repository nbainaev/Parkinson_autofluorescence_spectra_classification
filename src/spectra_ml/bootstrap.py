"""Stage 9: patient-level bootstrap uncertainty (build_project_and_exp.md
section 21). Operates entirely on the already-computed OOF patient_predictions
table -- NO refitting, so it stays cheap (seconds, not minutes) regardless of
how expensive the underlying model was to train. Bootstrap resampling is
stratified by class and always at the PATIENT level (never resampling a
patient's individual spectra as if independent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spectra_ml.evaluation import compute_patient_metrics

METRIC_COLS = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc",
               "brier", "log_loss", "calibration_slope", "calibration_intercept"]


def _stratified_resample_indices(target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap draw: resample WITHIN each class with replacement, at the
    same size as that class, then concatenate -- keeps class balance stable
    across replicates instead of letting it drift (or occasionally vanish)."""
    idx_parts = []
    for cls in (0.0, 1.0):
        cls_idx = np.where(target == cls)[0]
        idx_parts.append(cls_idx[rng.integers(0, len(cls_idx), len(cls_idx))])
    return np.concatenate(idx_parts)


def bootstrap_patient_metrics(patient_df: pd.DataFrame, prob_col: str = "calibrated_probability",
                               n_boot: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """patient_df: one row per patient (patient_id, target, prob_col). Returns
    (replicates, summary): replicates has one row per bootstrap iteration (all
    metrics), summary has mean/median/std/95% CI per metric."""
    y = patient_df["target"].to_numpy()
    p = patient_df[prob_col].to_numpy()
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n_boot):
        idx = _stratified_resample_indices(y, rng)
        rows.append(compute_patient_metrics(y[idx], p[idx]))
    replicates = pd.DataFrame(rows)

    summary_rows = []
    for col in METRIC_COLS:
        vals = replicates[col].dropna().to_numpy()
        if len(vals) == 0:
            summary_rows.append({"metric": col, "mean": np.nan, "median": np.nan, "std": np.nan,
                                  "ci95_lo": np.nan, "ci95_hi": np.nan})
            continue
        summary_rows.append({
            "metric": col, "mean": float(vals.mean()), "median": float(np.median(vals)),
            "std": float(vals.std()), "ci95_lo": float(np.percentile(vals, 2.5)),
            "ci95_hi": float(np.percentile(vals, 97.5)),
        })
    summary = pd.DataFrame(summary_rows)
    return replicates, summary


def paired_bootstrap_compare(patient_df_a: pd.DataFrame, patient_df_b: pd.DataFrame,
                              prob_col: str = "calibrated_probability", metric: str = "roc_auc",
                              n_boot: int = 2000, seed: int = 42) -> dict:
    """Compare two models on the SAME patients using the SAME bootstrap draws each
    iteration (paired), so the comparison isn't confounded by independent
    resampling noise (section 21: "для сравнения двух моделей использовать
    paired bootstrap на одних и тех же пациентах")."""
    merged = patient_df_a[["patient_id", "target", prob_col]].merge(
        patient_df_b[["patient_id", prob_col]], on="patient_id", suffixes=("_a", "_b"), validate="one_to_one",
    )
    if len(merged) != len(patient_df_a) or len(merged) != len(patient_df_b):
        raise ValueError(
            f"patient_df_a ({len(patient_df_a)} patients) and patient_df_b ({len(patient_df_b)} patients) "
            f"must cover the exact same patient set for a paired comparison; only {len(merged)} matched."
        )
    y = merged["target"].to_numpy()
    p_a = merged[f"{prob_col}_a"].to_numpy()
    p_b = merged[f"{prob_col}_b"].to_numpy()
    rng = np.random.default_rng(seed)

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stratified_resample_indices(y, rng)
        m_a = compute_patient_metrics(y[idx], p_a[idx])[metric]
        m_b = compute_patient_metrics(y[idx], p_b[idx])[metric]
        deltas[b] = m_a - m_b

    deltas = deltas[~np.isnan(deltas)]
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p-value: fraction of the OTHER tail, doubled, capped at 1
    p_value = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "metric": metric, "n_patients": len(merged), "n_boot": len(deltas),
        "mean_delta": float(deltas.mean()), "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        "p_value_approx": float(min(p_value, 1.0)), "significant_at_0.05": bool(ci_lo > 0 or ci_hi < 0),
    }


# ---------------------------------------------------------------------------
# Multiclass (N>2 classes) sibling, additive -- binary bootstrap_patient_
# metrics/paired_bootstrap_compare above are unchanged.
# ---------------------------------------------------------------------------

def _stratified_resample_indices_multiclass(target: np.ndarray, class_labels, rng: np.random.Generator) -> np.ndarray:
    idx_parts = []
    for cls in class_labels:
        cls_idx = np.where(target == cls)[0]
        idx_parts.append(cls_idx[rng.integers(0, len(cls_idx), len(cls_idx))])
    return np.concatenate(idx_parts)


def bootstrap_patient_metrics_multiclass(patient_df: pd.DataFrame, class_labels, prob_cols=None,
                                          n_boot: int = 750, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """patient_df: one row per patient (patient_id, target, prob_<class> for
    each class in class_labels). Patient-level, class-stratified bootstrap of
    the macro-OvR-AUC (and other compute_patient_metrics_multiclass scalar
    metrics). n_boot default is deliberately smaller than the binary
    pipeline's 2000 (500-1000 range, per task runtime-discipline guidance);
    750 chosen as a reasonable middle point -- CI half-widths from bootstrap
    percentiles shrink slowly (~1/sqrt(n_boot)), so 750 vs 2000 changes CI
    precision only marginally for the purpose of a descriptive report."""
    from spectra_ml.evaluation import compute_patient_metrics_multiclass

    class_labels = list(class_labels)
    if prob_cols is None:
        prob_cols = [f"prob_{c}" for c in class_labels]
    y = patient_df["target"].to_numpy()
    p = patient_df[prob_cols].to_numpy()
    rng = np.random.default_rng(seed)

    scalar_keys = ["n_patients", "roc_auc_macro_ovr", "balanced_accuracy", "macro_f1", "brier_multiclass",
                   "log_loss"]
    rows = []
    for _ in range(n_boot):
        idx = _stratified_resample_indices_multiclass(y, class_labels, rng)
        m = compute_patient_metrics_multiclass(y[idx], p[idx], class_labels)
        rows.append({k: m[k] for k in scalar_keys})
    replicates = pd.DataFrame(rows)

    summary_rows = []
    for col in scalar_keys:
        vals = replicates[col].dropna().to_numpy()
        if len(vals) == 0:
            summary_rows.append({"metric": col, "mean": np.nan, "median": np.nan, "std": np.nan,
                                  "ci95_lo": np.nan, "ci95_hi": np.nan})
            continue
        summary_rows.append({
            "metric": col, "mean": float(vals.mean()), "median": float(np.median(vals)),
            "std": float(vals.std()), "ci95_lo": float(np.percentile(vals, 2.5)),
            "ci95_hi": float(np.percentile(vals, 97.5)),
        })
    summary = pd.DataFrame(summary_rows)
    return replicates, summary
