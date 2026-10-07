import numpy as np
import pytest

from spectra_ml.calibration import PlattCalibrator, calibration_slope_intercept, expected_calibration_error, safe_logit


class TestSafeLogit:
    def test_clips_extremes(self):
        p = np.array([0.0, 1.0, 0.5])
        x = safe_logit(p)
        assert np.isfinite(x).all()

    def test_monotonic(self):
        p = np.linspace(0.01, 0.99, 20)
        x = safe_logit(p)
        assert np.all(np.diff(x) > 0)


class TestPlattCalibrator:
    def test_improves_overconfident_probabilities(self):
        """Simulate an overconfident model (extreme probabilities that don't match
        true frequencies) and check Platt scaling pulls them toward calibrated."""
        rng = np.random.default_rng(0)
        n = 400
        true_logit = rng.normal(scale=1.0, size=n)
        y = (rng.random(n) < 1 / (1 + np.exp(-true_logit))).astype(float)
        # overconfident: raw score is the true logit scaled up 3x, squashed to prob
        raw_prob = 1 / (1 + np.exp(-3 * true_logit))

        cal = PlattCalibrator().fit(raw_prob, y)
        calibrated = cal.predict_proba(raw_prob)

        raw_slope, _ = calibration_slope_intercept(y, raw_prob)
        cal_slope, _ = calibration_slope_intercept(y, calibrated)
        assert abs(cal_slope - 1.0) < abs(raw_slope - 1.0)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            PlattCalibrator().predict_proba(np.array([0.5, 0.6]))

    def test_predict_proba_never_refits(self):
        """predict_proba on a differently-distributed 'test' set must not change
        the calibrator's learned parameters (i.e. it applies a FIXED mapping
        learned once from 'train', never refits itself against test data)."""
        rng = np.random.default_rng(0)
        n = 400
        true_logit = rng.normal(scale=1.0, size=n)
        train_y = (rng.random(n) < 1 / (1 + np.exp(-true_logit))).astype(float)
        train_prob = 1 / (1 + np.exp(-3 * true_logit))

        cal = PlattCalibrator().fit(train_prob, train_y)
        coef_before = cal.model_.coef_.copy()
        intercept_before = cal.model_.intercept_.copy()

        test_prob = np.array([0.9, 0.95, 0.99])  # far outside train's typical range
        cal.predict_proba(test_prob)

        assert np.array_equal(cal.model_.coef_, coef_before)
        assert np.array_equal(cal.model_.intercept_, intercept_before)


class TestCalibrationMetrics:
    def test_perfect_calibration_slope_one_intercept_zero(self):
        rng = np.random.default_rng(2)
        n = 2000
        logit = rng.normal(size=n)
        p = 1 / (1 + np.exp(-logit))
        y = (rng.random(n) < p).astype(float)
        slope, intercept = calibration_slope_intercept(y, p)
        assert abs(slope - 1.0) < 0.15
        assert abs(intercept) < 0.15

    def test_single_class_returns_nan(self):
        y = np.zeros(10)
        p = np.linspace(0.1, 0.9, 10)
        slope, intercept = calibration_slope_intercept(y, p)
        assert np.isnan(slope) and np.isnan(intercept)

    def test_ece_zero_for_perfect_calibration(self):
        rng = np.random.default_rng(3)
        n = 5000
        p = rng.uniform(0, 1, n)
        y = (rng.random(n) < p).astype(float)
        ece = expected_calibration_error(y, p, n_bins=10)
        assert ece < 0.05

    def test_ece_high_for_bad_calibration(self):
        n = 100
        p = np.full(n, 0.9)
        y = np.zeros(n)  # confidently wrong every time
        ece = expected_calibration_error(y, p, n_bins=5)
        assert ece > 0.8
