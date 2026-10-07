import numpy as np
import pytest

from spectra_ml.preprocessing import (AreaNormalizer, DerivativeTransformer, SavitzkyGolayTransformer,
                                       SNVTransformer, SpectralStandardScaler, VectorNormalizer,
                                       build_preprocessing_pipeline)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def waves():
    return np.linspace(400, 680, 150)


@pytest.fixture
def smooth_spectra(waves, rng):
    """5 smooth Gaussian-bump spectra, no noise."""
    n = 5
    centers = rng.uniform(450, 630, size=n)
    return np.stack([100 + 500 * np.exp(-((waves - c) ** 2) / (2 * 30 ** 2)) for c in centers])


class TestSNVTransformer:
    def test_row_mean_zero_std_one(self, smooth_spectra):
        out = SNVTransformer().fit_transform(smooth_spectra)
        assert np.allclose(out.mean(axis=1), 0, atol=1e-8)
        assert np.allclose(out.std(axis=1), 1, atol=1e-6)

    def test_rejects_nan(self, smooth_spectra):
        X = smooth_spectra.copy()
        X[0, 0] = np.nan
        with pytest.raises(ValueError):
            SNVTransformer().fit_transform(X)


class TestVectorNormalizer:
    def test_unit_l2_norm(self, smooth_spectra):
        out = VectorNormalizer().fit_transform(smooth_spectra)
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1, atol=1e-8)


class TestAreaNormalizer:
    def test_unit_area(self, smooth_spectra, waves):
        out = AreaNormalizer(wavelengths=waves).fit_transform(smooth_spectra)
        areas = np.trapezoid(np.abs(out), waves, axis=1)
        assert np.allclose(areas, 1, atol=1e-6)

    def test_default_axis_is_sample_index(self, smooth_spectra):
        # no wavelengths given -> unit sample spacing, still normalizes consistently
        out = AreaNormalizer().fit_transform(smooth_spectra)
        areas = np.trapezoid(np.abs(out), axis=1)
        assert np.allclose(areas, 1, atol=1e-6)


class TestSavitzkyGolayTransformer:
    def test_reproduces_low_order_polynomial(self):
        x = np.arange(101.0)
        quad = (0.5 * x ** 2 - 3 * x + 7)[None, :]
        out = SavitzkyGolayTransformer(window_length=11, polyorder=3).fit_transform(quad)
        # interior points should be reproduced almost exactly (SG with polyorder>=2 is exact on a quadratic)
        assert np.allclose(out[0, 10:-10], quad[0, 10:-10], atol=1e-6)

    def test_reduces_noise_variance(self, rng):
        x = np.linspace(0, 10, 200)
        clean = np.sin(x)
        noisy = clean + rng.normal(scale=0.3, size=x.shape)
        out = SavitzkyGolayTransformer(window_length=15, polyorder=3).fit_transform(noisy[None, :])
        resid_before = np.var(noisy - clean)
        resid_after = np.var(out[0] - clean)
        assert resid_after < resid_before


class TestDerivativeTransformer:
    def test_first_derivative_of_line_is_constant(self):
        x = np.arange(101.0)
        line = (3.0 * x + 5.0)[None, :]
        out = DerivativeTransformer(window_length=11, polyorder=2, order=1).fit_transform(line)
        assert np.allclose(out[0, 10:-10], 3.0, atol=1e-6)

    def test_rejects_order_zero(self, smooth_spectra):
        with pytest.raises(ValueError):
            DerivativeTransformer(order=0).fit(smooth_spectra)


class TestSpectralStandardScaler:
    def test_train_stats_applied_to_test_not_refit(self, rng):
        train = rng.normal(loc=10, scale=2, size=(50, 20))
        test = rng.normal(loc=50, scale=5, size=(10, 20))  # deliberately shifted distribution

        scaler = SpectralStandardScaler().fit(train)
        train_out = scaler.transform(train)
        test_out = scaler.transform(test)

        assert np.allclose(train_out.mean(axis=0), 0, atol=1e-8)
        assert np.allclose(train_out.std(axis=0), 1, atol=1e-6)
        # test data was NOT standardized to its own distribution (no leakage: scaler kept train mean/std)
        assert not np.allclose(test_out.mean(axis=0), 0, atol=1)

    def test_transform_before_fit_raises(self, smooth_spectra):
        with pytest.raises(Exception):
            SpectralStandardScaler().transform(smooth_spectra)

    def test_feature_count_mismatch_raises(self, smooth_spectra):
        scaler = SpectralStandardScaler().fit(smooth_spectra)
        with pytest.raises(ValueError):
            scaler.transform(smooth_spectra[:, :-1])

    def test_rejects_nan(self, smooth_spectra):
        X = smooth_spectra.copy()
        X[0, 0] = np.nan
        with pytest.raises(ValueError):
            SpectralStandardScaler().fit(X)


class TestBuildPreprocessingPipeline:
    @pytest.mark.parametrize("variant", ["A", "B", "C", "D"])
    def test_fit_transform_roundtrip(self, variant, smooth_spectra, waves):
        pipe = build_preprocessing_pipeline(variant, wavelengths=waves)
        out = pipe.fit_transform(smooth_spectra)
        assert out.shape[0] == smooth_spectra.shape[0]
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("variant", ["A", "B", "C", "D"])
    def test_transform_after_separate_fit_works(self, variant, smooth_spectra, waves):
        """Regression test: sklearn>=1.8's Pipeline.transform() requires every step to
        expose a fitted marker; a step whose fit() returns self without setting any
        trailing-underscore attribute makes the whole Pipeline look unfitted even
        after fit() succeeded (hit this for real with variant B in Stage 6)."""
        pipe = build_preprocessing_pipeline(variant, wavelengths=waves)
        pipe.fit(smooth_spectra)
        out = pipe.transform(smooth_spectra)
        assert np.isfinite(out).all()

    def test_unknown_variant_raises(self, waves):
        with pytest.raises(ValueError):
            build_preprocessing_pipeline("Z", wavelengths=waves)

    def test_pipeline_fits_only_on_train_fold(self, rng, waves):
        """sklearn.Pipeline should keep the scaler's learned stats from
        fit(train) when transforming a differently-distributed test set --
        exercises the same leakage-safety property end-to-end through a
        Pipeline, not just the bare transformer."""
        smooth_spectra_train = 100 + 500 * rng.normal(size=(20, len(waves))).cumsum(axis=1)
        smooth_spectra_test = 1000 + 500 * rng.normal(size=(5, len(waves))).cumsum(axis=1)

        pipe = build_preprocessing_pipeline("A", wavelengths=waves)
        pipe.fit(smooth_spectra_train)
        train_out = pipe.transform(smooth_spectra_train)
        test_out = pipe.transform(smooth_spectra_test)

        assert np.allclose(train_out.mean(axis=0), 0, atol=1e-6)
        assert not np.allclose(test_out.mean(axis=0), 0, atol=1)
