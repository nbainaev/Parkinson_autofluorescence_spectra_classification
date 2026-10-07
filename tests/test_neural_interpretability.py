import numpy as np
import torch

from spectra_ml.models.mlp import MLPBinaryClassifier
from spectra_ml.neural_interpretability import (aggregate_attribution_to_patient, compute_baselines,
                                                  global_attribution_summary, gradient_shap_attribution,
                                                  integrated_gradients)


def train_toy_model(seed=0, n_wave=15):
    rng = np.random.default_rng(seed)
    n = 60
    X = rng.normal(size=(n, n_wave))
    y = (X[:, 3] + X[:, 4] > 0).astype(float)  # signal concentrated in features 3,4
    pid = np.array([f"p{i}" for i in range(n)])
    model = MLPBinaryClassifier(hidden_dims=(16,), max_epochs=200, patience=20, random_state=seed)
    model.fit(X, y, sample_weight=np.ones(n), patient_id=pid)
    return model, X, y, pid


class TestIntegratedGradients:
    def test_completeness_axiom(self):
        """Sum of attributions over features should approximately equal
        F(x) - F(baseline) (the defining property of Integrated Gradients)."""
        model, X, y, pid = train_toy_model(seed=0)
        baseline = np.zeros(X.shape[1])
        attribution = integrated_gradients(model.module_, X[:10], baseline, steps=200)

        with torch.no_grad():
            logit_x = model.module_(torch.tensor(X[:10], dtype=torch.float32)).squeeze(-1).numpy()
            logit_base = model.module_(torch.tensor(np.broadcast_to(baseline, (10, X.shape[1])).copy(),
                                                       dtype=torch.float32)).squeeze(-1).numpy()
        completeness_error = np.abs(attribution.sum(axis=1) - (logit_x - logit_base))
        assert np.all(completeness_error < 0.05)

    def test_signal_features_have_higher_attribution(self):
        model, X, y, pid = train_toy_model(seed=1)
        baseline = np.zeros(X.shape[1])
        attribution = integrated_gradients(model.module_, X, baseline, steps=100)
        mean_abs = np.abs(attribution).mean(axis=0)
        signal_importance = mean_abs[[3, 4]].mean()
        other_importance = np.delete(mean_abs, [3, 4]).mean()
        assert signal_importance > other_importance

    def test_zero_attribution_at_baseline_itself(self):
        model, X, y, pid = train_toy_model(seed=2)
        baseline = X[0].copy()
        attribution = integrated_gradients(model.module_, X[:1], baseline, steps=50)
        assert np.allclose(attribution, 0, atol=1e-6)

    def test_conv_reshape_path_runs(self):
        from spectra_ml.models.cnn1d import CNN1DBinaryClassifier
        rng = np.random.default_rng(3)
        n_wave = 20
        X = rng.normal(size=(30, n_wave))
        y = (X[:, 5] > 0).astype(float)
        pid = np.array([f"p{i}" for i in range(30)])
        model = CNN1DBinaryClassifier(channels=(8,), kernel_sizes=(5,), max_epochs=30, patience=10, random_state=3)
        model.fit(X, y, sample_weight=np.ones(30), patient_id=pid)
        attribution = integrated_gradients(model.module_, X, np.zeros(n_wave), steps=20, reshape_for_conv=True)
        assert attribution.shape == X.shape
        assert np.isfinite(attribution).all()


class TestGradientShap:
    def test_signal_features_have_higher_attribution(self):
        model, X, y, pid = train_toy_model(seed=1)
        rng = np.random.default_rng(0)
        baseline_samples = X[rng.choice(len(X), 20, replace=False)]
        attribution = gradient_shap_attribution(model.module_, X, baseline_samples, n_samples=30, seed=1)
        mean_abs = np.abs(attribution).mean(axis=0)
        signal_importance = mean_abs[[3, 4]].mean()
        other_importance = np.delete(mean_abs, [3, 4]).mean()
        assert signal_importance > other_importance

    def test_agrees_with_integrated_gradients_on_signal_location(self):
        """Not identical (different baseline distributions/methods), but both
        methods should rank the true signal features similarly on top."""
        model, X, y, pid = train_toy_model(seed=8)
        rng = np.random.default_rng(0)
        baseline_samples = X[rng.choice(len(X), 20, replace=False)]

        ig_attr = integrated_gradients(model.module_, X, np.zeros(X.shape[1]), steps=100)
        gs_attr = gradient_shap_attribution(model.module_, X, baseline_samples, n_samples=30, seed=8)

        ig_rank = np.argsort(-np.abs(ig_attr).mean(axis=0))
        gs_rank = np.argsort(-np.abs(gs_attr).mean(axis=0))
        # true signal features (3, 4) should be in both methods' top-5
        assert {3, 4}.issubset(set(ig_rank[:5].tolist()))
        assert {3, 4}.issubset(set(gs_rank[:5].tolist()))

    def test_conv_reshape_path_runs(self):
        from spectra_ml.models.cnn1d import CNN1DBinaryClassifier
        rng = np.random.default_rng(3)
        n_wave = 20
        X = rng.normal(size=(30, n_wave))
        y = (X[:, 5] > 0).astype(float)
        pid = np.array([f"p{i}" for i in range(30)])
        model = CNN1DBinaryClassifier(channels=(8,), kernel_sizes=(5,), max_epochs=30, patience=10, random_state=3)
        model.fit(X, y, sample_weight=np.ones(30), patient_id=pid)
        baseline_samples = X[:10]
        attribution = gradient_shap_attribution(model.module_, X, baseline_samples, n_samples=10, seed=3,
                                                  reshape_for_conv=True)
        assert attribution.shape == X.shape
        assert np.isfinite(attribution).all()


class TestBaselinesAndAggregation:
    def test_compute_baselines_shapes(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(40, 12))
        baselines = compute_baselines(X_train, n_wave=12)
        assert set(baselines) == {"zero", "train_mean", "train_median"}
        for b in baselines.values():
            assert b.shape == (12,)
        assert np.allclose(baselines["zero"], 0)

    def test_aggregate_attribution_to_patient(self):
        attribution = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 10.0]])
        patient_id = np.array(["a", "a", "b"])
        mean_df, abs_mean_df = aggregate_attribution_to_patient(attribution, patient_id)
        assert np.allclose(mean_df.loc["a"].to_numpy(), [2.0, 3.0])
        assert np.allclose(mean_df.loc["b"].to_numpy(), [10.0, 10.0])

    def test_global_attribution_summary(self):
        waves = np.array([400.0, 500.0])
        patient_mean = np.array([[1.0, -1.0], [3.0, -3.0]])
        patient_abs_mean = np.abs(patient_mean)
        out = global_attribution_summary(patient_mean, patient_abs_mean, waves)
        assert list(out["wavelength"]) == [400.0, 500.0]
        assert out["mean_attribution"].iloc[0] == 2.0
        assert out["mean_abs_attribution"].iloc[1] == 2.0
