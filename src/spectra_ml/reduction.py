"""Stage 6 (used by): PCA / PLS feature extractors (build_project_and_exp.md
section 7). Both are sklearn-compatible fit/transform; the caller is
responsible for only ever calling `.fit()` on a training fold (same
convention as spectra_ml.preprocessing.SpectralStandardScaler)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted


class PCAFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X, y=None):
        n_components = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.model_ = PCA(n_components=n_components, random_state=0)
        self.model_.fit(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "model_")
        return self.model_.transform(X)

    def back_project_coefficients(self, linear_coef: np.ndarray) -> np.ndarray:
        """PCA sibling of PLSFeatureExtractor.back_project_coefficients: map a
        downstream linear model's coefficients (in PCA-component space) back to
        original (wavelength) feature space, so they are never reported as
        wavelength importances directly (section 23). PCA's transform is
        X_new = (X - mean_) @ components_.T (mean-centering does not affect the
        gradient), so d(coef @ X_new)/dX = coef @ components_ -- i.e. the same
        shape/derivation as PLS's x_rotations_ back-projection, using PCA's own
        components_ matrix instead."""
        check_is_fitted(self, "model_")
        return (linear_coef.reshape(1, -1) @ self.model_.components_).ravel()


class PLSFeatureExtractor(BaseEstimator, TransformerMixin):
    """Supervised feature extractor -- MUST be fit only on training data (it
    uses y), never on validation/test or the whole dataset before CV."""

    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def fit(self, X, y):
        if y is None:
            raise ValueError("PLSFeatureExtractor.fit requires y (supervised); do not fit on unlabeled/test data.")
        n_components = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.model_ = PLSRegression(n_components=n_components)
        self.model_.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "model_")
        return self.model_.transform(X)

    def vip_scores(self) -> np.ndarray:
        """Variable Importance in Projection -- for interpretability (section 23),
        NOT to be confused with the downstream linear model's own coefficients."""
        check_is_fitted(self, "model_")
        t = self.model_.x_scores_
        w = self.model_.x_weights_
        q = self.model_.y_loadings_
        p, h = w.shape
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        vip = np.sqrt(p * (w ** 2 @ s) / total_s).ravel()
        return vip

    def back_project_coefficients(self, linear_coef: np.ndarray) -> np.ndarray:
        """Map a downstream linear model's coefficients (in PLS-component space)
        back to original wavelength space, so they are never reported as
        wavelength importances directly (section 23)."""
        check_is_fitted(self, "model_")
        return (self.model_.x_rotations_ @ linear_coef.reshape(-1, 1)).ravel()
