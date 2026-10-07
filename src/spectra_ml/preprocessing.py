"""Stage 3: reusable spectral preprocessing transformers (build_project_and_exp.md
section 6). All are sklearn-compatible (fit/transform/get_params/set_params via
BaseEstimator+TransformerMixin) so they compose in sklearn.Pipeline and inherit
correct fit-on-train-only semantics from Pipeline/cross_val machinery.

Row-wise transformers (SNV, VectorNormalizer, AreaNormalizer, SavitzkyGolay,
Derivative) have no learned state -- `fit` is a no-op (they cannot leak between
folds by construction). SpectralStandardScaler DOES learn per-wavelength
mean/std from `fit` and must only ever be fit on a training fold.

NaNs: these transformers assume spectra have already passed Stage 2 QC and
hot-pixel correction (no interior NaNs). They raise on NaN input rather than
silently propagating it, since scipy.signal.savgol_filter and sklearn scalers
give hard-to-diagnose garbage on unhandled NaN.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

_EPS = 1e-12


def _check_no_nan(X: np.ndarray, name: str) -> np.ndarray:
    X = check_array(X, dtype=float)
    if np.isnan(X).any():
        raise ValueError(
            f"{name}: input contains NaN. Run QC (spectra_ml.qc) and hot-pixel "
            "correction / drop HARD_FAIL spectra first."
        )
    return X


class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    """Smoothing only (deriv=0). See DerivativeTransformer for deriv>0."""

    def __init__(self, window_length: int = 11, polyorder: int = 3):
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X, y=None):
        X = _check_no_nan(X, "SavitzkyGolayTransformer")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = _check_no_nan(X, "SavitzkyGolayTransformer")
        return savgol_filter(X, window_length=self.window_length, polyorder=self.polyorder, deriv=0, axis=1)


class DerivativeTransformer(BaseEstimator, TransformerMixin):
    """Savitzky-Golay derivative (order 1 or 2), evaluated on a uniform index
    grid (delta=1 sample -> derivative is in intensity/sample units, not
    intensity/nm; rescale by mean wavelength spacing upstream if physical
    units are needed)."""

    def __init__(self, window_length: int = 11, polyorder: int = 3, order: int = 1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.order = order

    def fit(self, X, y=None):
        X = _check_no_nan(X, "DerivativeTransformer")
        if self.order < 1:
            raise ValueError("DerivativeTransformer.order must be >= 1 (use SavitzkyGolayTransformer for order 0).")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = _check_no_nan(X, "DerivativeTransformer")
        return savgol_filter(X, window_length=self.window_length, polyorder=self.polyorder, deriv=self.order, axis=1)


class SNVTransformer(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: per-spectrum (row-wise) mean-center, unit-std."""

    def fit(self, X, y=None):
        X = _check_no_nan(X, "SNVTransformer")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = _check_no_nan(X, "SNVTransformer")
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        return (X - mean) / (std + _EPS)


class VectorNormalizer(BaseEstimator, TransformerMixin):
    """Per-spectrum L2-norm normalization."""

    def fit(self, X, y=None):
        X = _check_no_nan(X, "VectorNormalizer")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = _check_no_nan(X, "VectorNormalizer")
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norm + _EPS)


class AreaNormalizer(BaseEstimator, TransformerMixin):
    """Per-spectrum normalization by area under the curve (trapezoidal, using
    `wavelengths` if given, else unit sample spacing)."""

    def __init__(self, wavelengths: np.ndarray | None = None):
        self.wavelengths = wavelengths

    def fit(self, X, y=None):
        X = _check_no_nan(X, "AreaNormalizer")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = _check_no_nan(X, "AreaNormalizer")
        x_axis = self.wavelengths if self.wavelengths is not None else np.arange(X.shape[1])
        area = np.trapezoid(np.abs(X), x_axis, axis=1, )[:, None]
        return X / (area + _EPS)


class SpectralStandardScaler(BaseEstimator, TransformerMixin):
    """Per-wavelength (column-wise) standardization. Learns mean/std in `fit`
    -- must be fit on the training fold only inside CV (build_project_and_exp.md
    section 6/37: never fit a scaler on the whole dataset before CV)."""

    def fit(self, X, y=None):
        X = _check_no_nan(X, "SpectralStandardScaler")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ["mean_", "std_"])
        X = _check_no_nan(X, "SpectralStandardScaler")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        return (X - self.mean_) / (self.std_ + _EPS)


# ---------------------------------------------------------------------------
# Preprocessing branches (build_project_and_exp.md section 6, variants A-D).
# Preprocessing choice is model selection and must be made only inside CV
# (configs/qc.yaml pattern repeated: this module only builds candidate
# pipelines, it never picks one).
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(variant: str, wavelengths: np.ndarray | None = None,
                                  sg_window: int = 11, sg_polyorder: int = 3) -> "Pipeline":
    from sklearn.pipeline import Pipeline

    if variant == "A":  # minimal: raw -> StandardScaler
        steps = [("scaler", SpectralStandardScaler())]
    elif variant == "B":  # SG smoothing -> SNV
        steps = [
            ("sg", SavitzkyGolayTransformer(sg_window, sg_polyorder)),
            ("snv", SNVTransformer()),
        ]
    elif variant == "C":  # SG first derivative -> StandardScaler
        steps = [
            ("deriv", DerivativeTransformer(sg_window, sg_polyorder, order=1)),
            ("scaler", SpectralStandardScaler()),
        ]
    elif variant == "D":  # vector/area normalization -> StandardScaler
        steps = [
            ("norm", AreaNormalizer(wavelengths=wavelengths)),
            ("scaler", SpectralStandardScaler()),
        ]
    elif variant == "E":  # SNV only -- current_tasks/double_blind_exp.md Part XIX's fixed final-model branch:
        # QC -> wavelength harmonization -> area-based instrument transfer -> SNV -> model. Those first three
        # steps happen upstream (spectra_ml.harmonization, before this pipeline ever sees the data), so this
        # variant is deliberately just the last step, with NO additional smoothing/derivative/scaling that
        # would reopen preprocessing choices after the frozen protocol (Part XIX: "не менять порядок после
        # просмотра blind-test performance").
        steps = [("snv", SNVTransformer())]
    else:
        raise ValueError(f"unknown preprocessing variant {variant!r}, expected one of A/B/C/D/E")
    return Pipeline(steps)
