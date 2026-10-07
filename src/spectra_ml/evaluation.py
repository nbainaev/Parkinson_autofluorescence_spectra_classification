"""Stage 6+ shared evaluation utilities: patient-balanced sample weights
(section 11), spectrum->patient aggregation (section 17), and patient-level
metrics (section 20). All metrics in this module are computed on PATIENT-level
aggregated predictions, per the spec's "the primary result is
P(Parkinson | patient), one probability per patient" requirement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                              log_loss, matthews_corrcoef, roc_auc_score)

from spectra_ml.calibration import calibration_slope_intercept


def patient_balanced_sample_weight(patient_id: np.ndarray, class_weight: dict | None = None,
                                    target: np.ndarray | None = None) -> np.ndarray:
    """weight_i = 1 / n_spectra_for_patient(i), optionally * class_weight[target_i],
    so a patient with 16 spectra doesn't outweigh one with 1 (section 11)."""
    counts = pd.Series(patient_id).map(pd.Series(patient_id).value_counts())
    weight = 1.0 / counts.to_numpy()
    if class_weight is not None:
        if target is None:
            raise ValueError("target is required when class_weight is given")
        weight = weight * np.array([class_weight[t] for t in target])
    return weight


def aggregate_to_patient(spectrum_scores: np.ndarray, patient_id: np.ndarray, target: np.ndarray,
                          method: str = "mean") -> pd.DataFrame:
    """spectrum_scores: per-spectrum probabilities (or logits). Returns one row
    per patient: patient_id, target, probability (or score)."""
    df = pd.DataFrame({"patient_id": patient_id, "target": target, "score": spectrum_scores})
    agg_fn = "mean" if method == "mean" else "median"
    out = df.groupby("patient_id", sort=False).agg(target=("target", "first"), probability=("score", agg_fn))
    return out.reset_index()


def aggregate_logit_to_patient(spectrum_probs: np.ndarray, patient_id: np.ndarray, target: np.ndarray,
                                eps: float = 1e-6) -> pd.DataFrame:
    """Sensitivity variant: aggregate in logit space then sigmoid back, instead
    of averaging probabilities directly (section 17)."""
    p = np.clip(spectrum_probs, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    agg = aggregate_to_patient(logit, patient_id, target, method="mean")
    agg["probability"] = 1 / (1 + np.exp(-agg["probability"]))
    return agg


def compute_patient_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Primary patient-level metrics (section 20). Requires both classes present."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    cal_slope, cal_intercept = calibration_slope_intercept(y_true, y_prob) if len(set(y_true.tolist())) > 1 \
        else (np.nan, np.nan)

    return {
        "n_patients": len(y_true),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]) if len(set(y_true.tolist())) > 1 else np.nan,
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
        "threshold": threshold,
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold selection MUST be fit on training/OOF data only, never on
    outer test (section 20)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


# ---------------------------------------------------------------------------
# Multiclass (N>2 classes) siblings, additive -- binary aggregate_to_patient /
# compute_patient_metrics above are unchanged and still used by every binary
# pipeline (outputs/old, outputs/united, outputs/final_study*).
# ---------------------------------------------------------------------------

def aggregate_to_patient_multiclass(spectrum_probs: np.ndarray, patient_id: np.ndarray, target: np.ndarray,
                                     class_labels, method: str = "mean") -> pd.DataFrame:
    """spectrum_probs: (n_spectra, n_classes) per-spectrum class probabilities,
    columns already in `class_labels` order. Returns one row per patient:
    patient_id, target, prob_<label> for each class in class_labels (mean/
    median aggregated across that patient's spectra, matching the binary
    aggregate_to_patient convention)."""
    class_labels = list(class_labels)
    spectrum_probs = np.asarray(spectrum_probs)
    if spectrum_probs.shape[1] != len(class_labels):
        raise ValueError(f"spectrum_probs has {spectrum_probs.shape[1]} columns, expected {len(class_labels)}")
    df = pd.DataFrame({"patient_id": patient_id, "target": target})
    for k, c in enumerate(class_labels):
        df[f"prob_{c}"] = spectrum_probs[:, k]
    agg_fn = "mean" if method == "mean" else "median"
    agg_dict = {"target": ("target", "first")}
    agg_dict.update({f"prob_{c}": (f"prob_{c}", agg_fn) for c in class_labels})
    out = df.groupby("patient_id", sort=False).agg(**agg_dict)
    return out.reset_index()


def macro_ovr_auc(y_true: np.ndarray, prob_matrix: np.ndarray, class_labels) -> float:
    """Macro one-vs-rest ROC-AUC computed as the plain (unweighted) mean of
    each class's own binary AUC (class-k column vs (y_true == class_k)),
    bypassing sklearn's `roc_auc_score(..., multi_class="ovr")` convenience
    wrapper. Mathematically IDENTICAL to that wrapper's own `average="macro"`
    output whenever `prob_matrix` rows already sum to 1 (verified: OvR-macro
    AUC is defined as exactly this per-class average) -- but unlike the
    wrapper, this does not require rows to sum to 1, which matters for the
    rbf_svm multiclass family (spectra_ml.models.classical.build_rbf_svm_
    multiclass_candidates): its "probabilities" are independent per-class
    sigmoid-squashed OvR decision_function values (deliberately NOT
    renormalized -- see that function's own docstring), which sklearn>=1.5's
    stricter multiclass roc_auc_score input validation rejects outright
    ("Target scores need to be probabilities ... sum up to 1.0") even though
    the AUC ranking itself (computed independently per class-column) does not
    mathematically depend on that normalization at all."""
    class_labels = list(class_labels)
    y_true = np.asarray(y_true)
    prob_matrix = np.asarray(prob_matrix)
    aucs = [roc_auc_score((y_true == c).astype(int), prob_matrix[:, k]) for k, c in enumerate(class_labels)]
    return float(np.mean(aucs))


def compute_patient_metrics_multiclass(y_true: np.ndarray, prob_matrix: np.ndarray, class_labels) -> dict:
    """Primary patient-level multiclass metrics: macro one-vs-rest ROC-AUC,
    balanced accuracy, macro F1, per-class precision/recall/F1, and the full
    confusion matrix (argmax prediction). Requires every class in
    `class_labels` to be present in y_true (raises via sklearn otherwise, same
    contract as the binary compute_patient_metrics requiring both classes)."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    class_labels = list(class_labels)
    y_true = np.asarray(y_true)
    prob_matrix = np.asarray(prob_matrix)
    y_pred = np.array(class_labels)[np.argmax(prob_matrix, axis=1)]

    present = set(np.unique(y_true).tolist())
    expected = set(float(c) for c in class_labels)
    roc_auc_macro_ovr = np.nan
    log_loss_val = np.nan
    if present == expected:
        roc_auc_macro_ovr = macro_ovr_auc(y_true, prob_matrix, class_labels)
        log_loss_val = log_loss(y_true, prob_matrix, labels=class_labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, zero_division=0,
    )
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # multiclass Brier score: mean squared error between one-hot targets and
    # predicted probabilities, averaged over classes (standard generalization
    # of the binary Brier score).
    onehot = np.zeros_like(prob_matrix)
    for k, c in enumerate(class_labels):
        onehot[:, k] = (y_true == c).astype(float)
    brier_multiclass = float(np.mean(np.sum((prob_matrix - onehot) ** 2, axis=1)))

    return {
        "n_patients": len(y_true),
        "roc_auc_macro_ovr": float(roc_auc_macro_ovr) if not np.isnan(roc_auc_macro_ovr) else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1,
        "brier_multiclass": brier_multiclass,
        "log_loss": float(log_loss_val) if not np.isnan(log_loss_val) else np.nan,
        "per_class_precision": {str(c): float(p) for c, p in zip(class_labels, precision)},
        "per_class_recall": {str(c): float(r) for c, r in zip(class_labels, recall)},
        "per_class_f1": {str(c): float(f) for c, f in zip(class_labels, f1)},
        "per_class_support": {str(c): int(s) for c, s in zip(class_labels, support)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [str(c) for c in class_labels],
    }
