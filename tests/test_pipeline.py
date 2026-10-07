import numpy as np
import pytest

from spectra_ml.evaluation import aggregate_to_patient, compute_patient_metrics, patient_balanced_sample_weight
from spectra_ml.experiment import _fit_with_sample_weight, run_all_families_nested_cv, run_family_nested_cv
from spectra_ml.models.classical import build_dummy_candidates, build_logreg_candidates, build_pls_logreg_candidates
from spectra_ml.reduction import PCAFeatureExtractor, PLSFeatureExtractor


def make_synthetic_dataset(n_park=40, n_control=40, spectra_per_patient=6, n_wave=30, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    waves = np.linspace(400, 680, n_wave)
    patient_id, target, spectra = [], [], []
    for cls, n in [(1, n_park), (0, n_control)]:
        for p in range(n):
            k = rng.integers(3, spectra_per_patient + 1)
            base = rng.normal(size=n_wave) * 0.3
            base[5:10] += signal * cls  # localized signal so a real model can beat Dummy
            for _ in range(k):
                spectra.append(base + rng.normal(scale=0.2, size=n_wave))
                patient_id.append(f"{'P' if cls else 'C'}{p}")
                target.append(float(cls))
    return np.array(spectra), np.array(patient_id), np.array(target), waves


class TestReductionLeakage:
    def test_pca_transform_does_not_refit(self):
        rng = np.random.default_rng(0)
        train = rng.normal(size=(30, 10))
        test = rng.normal(loc=5, size=(10, 10))
        pca = PCAFeatureExtractor(n_components=3).fit(train)
        out_train = pca.transform(train)
        out_test = pca.transform(test)
        # same fitted components used regardless of what's transformed
        assert np.allclose(pca.model_.transform(test), out_test)
        assert out_train.shape == (30, 3)

    def test_pls_requires_y(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 10))
        with pytest.raises(ValueError):
            PLSFeatureExtractor(n_components=2).fit(X, None)

    def test_pls_fit_only_uses_given_data(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(20, 10))
        y_train = rng.integers(0, 2, 20).astype(float)
        X_test = rng.normal(loc=3, size=(5, 10))
        pls = PLSFeatureExtractor(n_components=2).fit(X_train, y_train)
        out_test = pls.transform(X_test)
        assert out_test.shape == (5, 2)


class TestEvaluationUtils:
    def test_patient_balanced_weight_sums_equal_per_patient(self):
        pid = np.array(["a", "a", "a", "a", "b", "b"])
        w = patient_balanced_sample_weight(pid)
        assert np.isclose(w[pid == "a"].sum(), 1.0)
        assert np.isclose(w[pid == "b"].sum(), 1.0)

    def test_aggregate_to_patient_mean(self):
        pid = np.array(["a", "a", "b"])
        target = np.array([1.0, 1.0, 0.0])
        scores = np.array([0.2, 0.8, 0.5])
        out = aggregate_to_patient(scores, pid, target)
        row_a = out[out["patient_id"] == "a"].iloc[0]
        assert np.isclose(row_a["probability"], 0.5)

    def test_compute_patient_metrics_perfect_separation(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_patient_metrics(y_true, y_prob)
        assert m["roc_auc"] == 1.0
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0


class TestFitWithSampleWeight:
    def test_bare_estimator_uses_sample_weight(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(40, 5)), rng.integers(0, 2, 40).astype(float)
        sw = rng.uniform(0.1, 1, 40)
        model = LogisticRegression()
        _fit_with_sample_weight(model, X, y, sw)
        assert hasattr(model, "coef_")

    def test_pipeline_routes_sample_weight_to_final_step(self):
        """Regression test: a bare `model.fit(X, y, sample_weight=sw)` on a Pipeline
        raises ValueError (not TypeError), which an except-TypeError-only fallback
        does not catch -- hit this for real running pls_logreg in Stage 6."""
        candidates = build_pls_logreg_candidates(preprocessing_variants=("A",), n_components_grid=(2,), C_grid=(1.0,))
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(40, 10)), rng.integers(0, 2, 40).astype(float)
        sw = rng.uniform(0.1, 1, 40)
        model = candidates[0].build_fn()
        _fit_with_sample_weight(model, X, y, sw)  # must not raise
        assert hasattr(model.named_steps["logreg"], "coef_")


class TestNestedCVIntegration:
    def test_dummy_family_predictions_cover_every_patient_once(self):
        spectra, pid, target, waves = make_synthetic_dataset()
        candidates = build_dummy_candidates()
        preds, summary = run_family_nested_cv("dummy", candidates, spectra, target, pid, waves, seed=1)
        assert set(preds["patient_id"]) == set(pid)
        assert preds["patient_id"].value_counts().eq(1).all()  # one outer-test appearance each
        assert len(summary) >= 2  # >=2 outer folds

    def test_dummy_roc_auc_near_chance(self):
        spectra, pid, target, waves = make_synthetic_dataset()
        candidates = build_dummy_candidates()
        preds, summary = run_family_nested_cv("dummy", candidates, spectra, target, pid, waves, seed=1)
        assert 0.3 <= summary["roc_auc"].mean() <= 0.7

    def test_logreg_beats_dummy_on_separable_synthetic_signal(self):
        spectra, pid, target, waves = make_synthetic_dataset(signal=3.0, seed=2)
        pool = build_dummy_candidates() + build_logreg_candidates(preprocessing_variants=("A",), C_grid=(1.0,), seed=2)
        preds, summary = run_all_families_nested_cv(pool, spectra, target, pid, waves, seed=2)
        dummy_auc = summary.loc[summary["family"] == "dummy", "roc_auc"].mean()
        logreg_auc = summary.loc[summary["family"] == "logreg", "roc_auc"].mean()
        assert logreg_auc > dummy_auc

    def test_no_patient_appears_in_train_and_test_within_a_family_run(self):
        """Integration-level leakage check on the experiment runner itself, not
        just spectra_ml.splitting in isolation."""
        spectra, pid, target, waves = make_synthetic_dataset(seed=3)
        candidates = build_logreg_candidates(preprocessing_variants=("A",), C_grid=(1.0,), seed=3)
        preds, summary = run_family_nested_cv("logreg", candidates, spectra, target, pid, waves, seed=3)
        # every patient tested exactly once across all outer folds -> no double counting / no gaps
        assert sorted(preds["patient_id"]) == sorted(np.unique(pid))
