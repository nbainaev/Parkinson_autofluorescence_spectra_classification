"""Regression tests locking in the united_multiclass_v2 audit (task 1)
findings in spectra_ml.models.classical / spectra_ml.multiclass.
"""
import warnings

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from spectra_ml.models.classical import GroupAwareCalibratedLinearSVC
from spectra_ml.reduction import PCAFeatureExtractor, PLSFeatureExtractor


def _toy_multiclass(seed=0, n=60, n_features=20, n_classes=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = rng.integers(0, n_classes, size=n)
    patient_id = np.repeat(np.arange(n // 2), 2)
    sample_weight = rng.uniform(0.5, 1.5, size=n)
    return X, y, patient_id, sample_weight


class TestGroupAwareCalibratedLinearSVCSampleWeightRouting:
    """Bug found in the united_multiclass_v2 audit (task 1, item 4): when
    `reducer_kind` is set, the base estimator handed to CalibratedClassifierCV
    is a Pipeline([("reduce", ...), ("svc", ...)]); a bare `sample_weight=`
    kwarg to CalibratedClassifierCV.fit is silently NOT forwarded to a
    Pipeline's own .fit (sklearn's own has_fit_parameter-style check inspects
    the literal `fit` signature, which for Pipeline never names `sample_weight`
    -- it only warns and uses the weight for the sigmoid calibration step).
    The fix explicitly routes it to `svc__sample_weight` instead, which
    Pipeline.fit understands."""

    def test_reduced_svc_step_actually_receives_sample_weight(self, monkeypatch):
        received = []
        orig_fit = LinearSVC.fit

        def spy_fit(self, X, y, sample_weight=None):
            received.append(sample_weight is not None)
            return orig_fit(self, X, y, sample_weight=sample_weight)

        monkeypatch.setattr(LinearSVC, "fit", spy_fit)

        X, y, patient_id, sw = _toy_multiclass()
        model = GroupAwareCalibratedLinearSVC(reducer_kind="pca", n_components=5, seed=0)
        model.fit(X, y, sample_weight=sw, patient_id=patient_id)

        assert len(received) > 0, "LinearSVC.fit was never called"
        assert all(received), "sample_weight was not forwarded to the wrapped LinearSVC's own .fit()"

    def test_no_sample_weight_dropped_warning_for_reduced_variant(self):
        X, y, patient_id, sw = _toy_multiclass()
        model = GroupAwareCalibratedLinearSVC(reducer_kind="pca", n_components=5, seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y, sample_weight=sw, patient_id=patient_id)
        sw_dropped = [w for w in caught if "does not appear to accept sample_weight" in str(w.message)]
        assert not sw_dropped, f"sample_weight silently dropped: {[str(w.message) for w in sw_dropped]}"

    def test_unreduced_variant_still_works_unchanged(self):
        """reducer_kind=None (bare LinearSVC base estimator) already routed
        sample_weight correctly before this fix -- must remain unaffected."""
        X, y, patient_id, sw = _toy_multiclass()
        model = GroupAwareCalibratedLinearSVC(reducer_kind=None, seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y, sample_weight=sw, patient_id=patient_id)
        sw_dropped = [w for w in caught if "does not appear to accept sample_weight" in str(w.message)]
        assert not sw_dropped
        probs = model.predict_proba(X)
        assert probs.shape == (X.shape[0], 4)


class TestReductionComponentCountNotSilentlyCapped:
    """united_multiclass_v2 audit task 1, item 3: PLS=8 / PCA=10 component
    counts must actually be honored (not silently collapsed by an off-by-one
    or min(n_components, n_features)-style bug) whenever the training fold is
    large enough to support them -- confirmed here with fold sizes comparable
    to the smallest outer-fold training set in the real pipeline (129
    patients, smallest class stage2 n=14)."""

    def test_pca_uses_requested_n_components_when_fold_is_large_enough(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 278))  # comparable to a real outer-fold training set
        pca = PCAFeatureExtractor(n_components=10).fit(X)
        assert pca.model_.n_components_ == 10

    def test_pls_uses_requested_n_components_when_fold_is_large_enough(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 278))
        y = rng.integers(0, 4, size=100).astype(float)
        pls = PLSFeatureExtractor(n_components=8).fit(X, y)
        assert pls.model_.n_components == 8

    def test_pca_caps_only_when_fold_is_genuinely_too_small(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 278))  # n_samples - 1 = 5 < requested 10
        pca = PCAFeatureExtractor(n_components=10).fit(X)
        assert pca.model_.n_components_ == 5


class TestNonlinearReducedPoolRuntimeTrim:
    """Task 1 of the nonlinear/neural extension: multiclass_stage_candidate_
    pool_v2_nonlinear_reduced must (a) force every candidate onto
    preprocessing_variants=("D",) regardless of each builder's own
    DEFAULT_PREPROCESSING_VARIANTS=("A","B"), (b) exclude the raw-278-dim
    RF/HistGB siblings entirely, and (c) apply the mandated grid trims
    (RF max_depth_grid=(8,) only, RBF-SVM gamma_grid=("scale",) only)."""

    def test_all_candidates_forced_to_variant_d(self):
        from spectra_ml.models.classical import multiclass_stage_candidate_pool_v2_nonlinear_reduced
        cands = multiclass_stage_candidate_pool_v2_nonlinear_reduced()
        assert len(cands) > 0
        assert all(c.preprocessing_variant == "D" for c in cands)

    def test_raw_278dim_tree_siblings_excluded(self):
        from spectra_ml.models.classical import multiclass_stage_candidate_pool_v2_nonlinear_reduced
        families = {c.family for c in multiclass_stage_candidate_pool_v2_nonlinear_reduced()}
        assert "random_forest_raw" not in families
        assert "hgb_raw" not in families
        assert families == {"random_forest_reduced_pca", "hgb_reduced_pca", "rbf_svm", "knn"}

    def test_rf_reduced_max_depth_grid_trimmed_to_8_only(self):
        from spectra_ml.models.classical import multiclass_stage_candidate_pool_v2_nonlinear_reduced
        rf_cands = [c for c in multiclass_stage_candidate_pool_v2_nonlinear_reduced()
                    if c.family == "random_forest_reduced_pca"]
        assert all(c.params["max_depth"] == 8 for c in rf_cands)

    def test_rbf_svm_gamma_grid_trimmed_to_scale_only(self):
        from spectra_ml.models.classical import multiclass_stage_candidate_pool_v2_nonlinear_reduced
        svm_cands = [c for c in multiclass_stage_candidate_pool_v2_nonlinear_reduced() if c.family == "rbf_svm"]
        assert all(c.params["gamma"] == "scale" for c in svm_cands)
        assert len(svm_cands) == 2  # C_grid=(1.0, 10.0) x gamma_grid=("scale",)


class TestMacroOvrAucMatchesSklearnWhenNormalized:
    """spectra_ml.evaluation.macro_ovr_auc (used by _select_best_candidate_
    multiclass and compute_patient_metrics_multiclass) must reproduce
    sklearn's own roc_auc_score(..., multi_class="ovr", average="macro")
    exactly when probabilities DO sum to 1, and must not raise/require
    normalization when they do not (the rbf_svm multiclass family's
    sigmoid-squashed decision_function columns never sum to 1 by
    construction -- sklearn's own multiclass roc_auc_score rejects that
    input outright on this project's sklearn version)."""

    def test_matches_sklearn_on_normalized_probabilities(self):
        from sklearn.metrics import roc_auc_score

        from spectra_ml.evaluation import macro_ovr_auc

        rng = np.random.default_rng(0)
        y = rng.integers(0, 4, size=200)
        probs = rng.dirichlet(alpha=[1, 1, 1, 1], size=200)
        expected = roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=[0, 1, 2, 3])
        assert np.isclose(macro_ovr_auc(y, probs, [0, 1, 2, 3]), expected)

    def test_does_not_require_rows_to_sum_to_one(self):
        from spectra_ml.evaluation import macro_ovr_auc

        rng = np.random.default_rng(0)
        y = rng.integers(0, 4, size=200)
        raw = rng.normal(size=(200, 4))
        probs = 1.0 / (1.0 + np.exp(-raw))  # independent per-class sigmoid -- rows do NOT sum to 1
        assert not np.allclose(probs.sum(axis=1), 1.0)
        value = macro_ovr_auc(y, probs, [0, 1, 2, 3])
        assert 0.0 <= value <= 1.0
