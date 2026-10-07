"""Stage 8 (+19 metrics): patient-level probability calibration (build_project_and_exp.md
section 18). Platt/sigmoid scaling is the primary method; the calibrator is fit
on out-of-fold (OOF), patient-level scores from the outer TRAINING set only,
and never sees outer-test labels (section 18's fold recipe).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """1D logistic regression of y on logit(raw_probability) -- classic Platt
    scaling, applicable uniformly to classical and neural models alike
    (section 18: "использовать единую реализацию для classical models и
    neural networks")."""

    def __init__(self):
        self.model_ = None

    def fit(self, raw_prob: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        x = safe_logit(np.asarray(raw_prob)).reshape(-1, 1)
        # near-unregularized: Platt scaling is a 2-parameter fit, not meant to be
        # penalized the way a high-dimensional model would be
        self.model_ = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.model_.fit(x, y)
        return self

    def predict_proba(self, raw_prob: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("PlattCalibrator must be fit before predict_proba.")
        x = safe_logit(np.asarray(raw_prob)).reshape(-1, 1)
        return self.model_.predict_proba(x)[:, 1]


def calibration_slope_intercept(y_true: np.ndarray, raw_prob: np.ndarray) -> tuple[float, float]:
    """Fit y ~ intercept + slope * logit(p); slope=1, intercept=0 is perfect
    calibration. Slope<1 -> probabilities too extreme; slope>1 -> too flat."""
    x = safe_logit(np.asarray(raw_prob)).reshape(-1, 1)
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(x, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 5,
                                strategy: str = "uniform") -> float:
    """ECE with a SMALL default bin count -- section 19 warns ECE is sensitive to
    bin count on small samples. `strategy`: "uniform" (equal-width bins) or
    "quantile" (equal-count bins, more stable for skewed probability distributions)."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    n = len(prob)
    if strategy == "quantile":
        edges = np.quantile(prob, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (prob >= lo) & (prob <= hi if hi == edges[-1] else prob < hi)
        if not in_bin.any():
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = prob[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
