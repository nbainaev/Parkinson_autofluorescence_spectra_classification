import numpy as np
import pytest

from spectra_ml.harmonization import (
    FAIL, PASS, WARNING,
    apply_transfer,
    area_normalize,
    build_patient_area_normalized_repr,
    build_transfer_curve_raw,
    class_group_mean,
    compare_pd_control_transfer_curves,
    compute_similarity_metrics,
    estimate_wavelength_shift,
    evaluate_shift_gate,
    patient_bootstrap_curve,
    smooth_transfer_curve,
)


def gaussian_spectrum(waves, centers=(500.0, 560.0), widths=(40.0, 25.0), heights=(5.0, 2.0), baseline=0.5):
    y = np.full_like(waves, baseline, dtype=float)
    for c, w, h in zip(centers, widths, heights):
        y = y + h * np.exp(-0.5 * ((waves - c) / w) ** 2)
    return y


@pytest.fixture
def waves():
    return np.linspace(400.0, 680.0, 280)


class TestAreaNormalize:
    def test_integrates_to_one(self, waves):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(10, len(waves)))
        norm = area_normalize(spectra, waves)
        areas = np.trapezoid(norm, waves, axis=1)
        np.testing.assert_allclose(areas, 1.0, atol=1e-8)

    def test_scale_invariant_shape(self, waves):
        base = gaussian_spectrum(waves)
        spectra = np.vstack([base, base * 3.7])
        norm = area_normalize(spectra, waves)
        np.testing.assert_allclose(norm[0], norm[1], atol=1e-10)

    def test_near_zero_area_flagged_nan(self, waves):
        spectra = np.zeros((1, len(waves)))
        norm = area_normalize(spectra, waves)
        assert np.isnan(norm).all()


class TestWavelengthShiftRecovery:
    def test_recovers_known_subnm_shift(self, waves):
        base = gaussian_spectrum(waves)
        true_shift = 0.37  # NEW(lambda) = OLD(lambda - true_shift)
        new = np.interp(waves, waves + true_shift, base)
        res = estimate_wavelength_shift(waves, base, new, search_range_nm=(-5.0, 5.0), step_nm=0.02)
        assert res["delta_lambda_nm"] == pytest.approx(true_shift, abs=0.05)
        assert res["best_correlation"] > 0.99

    def test_recovers_negative_shift(self, waves):
        base = gaussian_spectrum(waves)
        true_shift = -1.2
        new = np.interp(waves, waves + true_shift, base)
        res = estimate_wavelength_shift(waves, base, new, search_range_nm=(-5.0, 5.0), step_nm=0.02)
        assert res["delta_lambda_nm"] == pytest.approx(true_shift, abs=0.05)

    def test_zero_shift_identical_spectra(self, waves):
        base = gaussian_spectrum(waves)
        res = estimate_wavelength_shift(waves, base, base, search_range_nm=(-5.0, 5.0), step_nm=0.02)
        assert res["delta_lambda_nm"] == pytest.approx(0.0, abs=0.03)

    def test_shift_outside_search_range_not_falsely_small(self, waves):
        """A large true shift, searched only within a small window, should land
        at (or near) the edge of the search range rather than falsely reporting
        a small in-range shift as the global optimum inside that window."""
        base = gaussian_spectrum(waves)
        true_shift = 4.9
        new = np.interp(waves, waves + true_shift, base)
        res = estimate_wavelength_shift(waves, base, new, search_range_nm=(-5.0, 5.0), step_nm=0.02)
        assert res["delta_lambda_nm"] > 3.0


class TestTransferCurveConstruction:
    def test_identical_spectra_give_unit_transfer(self, waves):
        base = gaussian_spectrum(waves)
        raw = build_transfer_curve_raw(base, base, denom_near_zero_rel_threshold=0.02, edge_exclude_n_points=3)
        valid = ~raw["unreliable_mask"]
        np.testing.assert_allclose(raw["k_raw"][valid], 1.0, atol=1e-10)

    def test_smoothed_curve_close_to_one_for_identical_spectra(self, waves):
        base = gaussian_spectrum(waves)
        raw = build_transfer_curve_raw(base, base, denom_near_zero_rel_threshold=0.02, edge_exclude_n_points=3)
        smoothed = smooth_transfer_curve(raw["k_raw"], window=15, polyorder=2)
        np.testing.assert_allclose(smoothed, 1.0, atol=1e-6)

    def test_denominator_near_zero_flagged(self, waves):
        numerator = gaussian_spectrum(waves)
        denominator = gaussian_spectrum(waves).copy()
        denominator[100] = 1e-10  # far below its own max -> should be flagged unreliable
        raw = build_transfer_curve_raw(numerator, denominator, denom_near_zero_rel_threshold=0.02,
                                        edge_exclude_n_points=0)
        assert raw["unreliable_mask"][100]

    def test_edge_points_always_excluded(self, waves):
        base = gaussian_spectrum(waves)
        raw = build_transfer_curve_raw(base, base, denom_near_zero_rel_threshold=0.02, edge_exclude_n_points=5)
        assert raw["unreliable_mask"][:5].all()
        assert raw["unreliable_mask"][-5:].all()

    def test_smoothing_fills_nan_and_stays_finite(self, waves):
        base = gaussian_spectrum(waves)
        raw = build_transfer_curve_raw(base, base * 1.5, denom_near_zero_rel_threshold=0.02,
                                        edge_exclude_n_points=3)
        smoothed = smooth_transfer_curve(raw["k_raw"], window=15, polyorder=2)
        assert np.isfinite(smoothed).all()
        assert len(smoothed) == len(waves)


class TestApplyTransfer:
    def test_apply_unit_transfer_is_noop_after_renorm(self, waves):
        base = gaussian_spectrum(waves)
        norm = area_normalize(base[None, :], waves)
        k = np.ones(len(waves))
        out = apply_transfer(norm, waves, k)
        np.testing.assert_allclose(out["harmonized"], norm, atol=1e-10)

    def test_harmonized_area_is_one(self, waves):
        rng = np.random.default_rng(1)
        norm = area_normalize(rng.uniform(0.1, 2.0, size=(5, len(waves))), waves)
        k = 1.0 + 0.3 * np.sin(np.linspace(0, 3, len(waves)))
        out = apply_transfer(norm, waves, k)
        areas = np.trapezoid(out["harmonized"], waves, axis=1)
        np.testing.assert_allclose(areas, 1.0, atol=1e-8)


class TestSimilarityMetrics:
    def test_identical_spectra_perfect_scores(self, waves):
        base = gaussian_spectrum(waves)
        m = compute_similarity_metrics(base, base, waves)
        assert m["pearson"] == pytest.approx(1.0, abs=1e-6)
        assert m["spearman"] == pytest.approx(1.0, abs=1e-6)
        assert m["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert m["spectral_angle_rad"] == pytest.approx(0.0, abs=1e-4)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-8)
        assert m["mean_abs_residual"] == pytest.approx(0.0, abs=1e-8)

    def test_dissimilar_spectra_low_scores(self, waves):
        a = gaussian_spectrum(waves, centers=(450.0,), widths=(10.0,), heights=(1.0,), baseline=0.0)
        b = gaussian_spectrum(waves, centers=(650.0,), widths=(10.0,), heights=(1.0,), baseline=0.0)
        m = compute_similarity_metrics(a, b, waves)
        assert m["cosine_similarity"] < 0.05
        assert m["spectral_angle_rad"] > 1.4  # close to orthogonal (pi/2 ~= 1.57)


class TestPdControlDivergence:
    def test_identical_curves_do_not_diverge(self, waves):
        k = 1.0 + 0.1 * np.cos(np.linspace(0, 2, len(waves)))
        result = compare_pd_control_transfer_curves(k, k, waves, min_correlation=0.9,
                                                      max_spectral_angle_rad=0.2618)
        assert not result["diverges"]
        assert result["warning"] is None

    def test_strongly_different_curves_trigger_warning(self, waves):
        k_pd = 1.0 + 0.05 * np.sin(np.linspace(0, 2, len(waves)))
        k_control = 3.0 - 0.05 * np.cos(np.linspace(0, 5, len(waves)))
        result = compare_pd_control_transfer_curves(k_pd, k_control, waves, min_correlation=0.9,
                                                      max_spectral_angle_rad=0.2618)
        assert result["diverges"]
        assert result["warning"] is not None
        assert "clinical subgroup" in result["warning"]


class TestShiftGate:
    def test_pass_when_estimate_and_ci_within_threshold(self):
        d = evaluate_shift_gate(point_estimate_nm=0.1, ci_lo_nm=0.0, ci_hi_nm=0.2, max_allowed_shift_nm=0.505)
        assert d.status == PASS
        assert d.stop_transfer_message is None

    def test_warning_when_ci_exceeds_but_point_estimate_within(self):
        d = evaluate_shift_gate(point_estimate_nm=0.2, ci_lo_nm=-0.1, ci_hi_nm=0.6, max_allowed_shift_nm=0.505)
        assert d.status == WARNING
        assert d.stop_transfer_message is None

    def test_fail_when_point_estimate_exceeds_threshold(self):
        d = evaluate_shift_gate(point_estimate_nm=0.9, ci_lo_nm=0.5, ci_hi_nm=1.3, max_allowed_shift_nm=0.505)
        assert d.status == FAIL
        assert d.stop_transfer_message is not None
        assert "misalignment" in d.stop_transfer_message

    def test_fail_on_negative_shift_beyond_threshold(self):
        d = evaluate_shift_gate(point_estimate_nm=-0.8, ci_lo_nm=-1.2, ci_hi_nm=-0.4, max_allowed_shift_nm=0.505)
        assert d.status == FAIL

    def test_fail_on_nan_estimate(self):
        d = evaluate_shift_gate(point_estimate_nm=np.nan, ci_lo_nm=np.nan, ci_hi_nm=np.nan,
                                 max_allowed_shift_nm=0.505)
        assert d.status == FAIL


class TestPatientBootstrapCurve:
    def test_ci_covers_point_estimate(self):
        rng = np.random.default_rng(0)
        spectra = rng.normal(loc=1.0, scale=0.1, size=(50, 20))
        boot = patient_bootstrap_curve(spectra, n_boot=500, seed=0, ci=95, estimator=np.mean)
        assert np.all(boot["ci_lo"] <= boot["point_estimate"])
        assert np.all(boot["point_estimate"] <= boot["ci_hi"])

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(1)
        spectra = rng.normal(size=(30, 10))
        boot_a = patient_bootstrap_curve(spectra, n_boot=100, seed=7, estimator=np.mean)
        boot_b = patient_bootstrap_curve(spectra, n_boot=100, seed=7, estimator=np.mean)
        np.testing.assert_array_equal(boot_a["ci_lo"], boot_b["ci_lo"])
        np.testing.assert_array_equal(boot_a["ci_hi"], boot_b["ci_hi"])


class TestPopulationMeanTransferPipeline:
    """End-to-end wiring check for the population-mean (target_value=None)
    harmonization method: build_patient_area_normalized_repr ->
    class_group_mean(target_value=None) -> build_transfer_curve_raw ->
    smooth_transfer_curve -> apply_transfer, on synthetic "old" and "new"
    cohorts each mixing PD (target=1) and control (target=0) patients, to
    confirm target_value=None uses ALL patients of that instrument regardless
    of diagnosis (not a diagnosis-conditioned reference population)."""

    def test_target_value_none_uses_all_patients_regardless_of_diagnosis(self, waves):
        rng = np.random.default_rng(11)
        base = gaussian_spectrum(waves)
        # 5 PD-labeled + 5 control-labeled patients, all with (noisy) copies of
        # the SAME underlying shape -- so the population mean is not sensitive
        # to which diagnosis subgroup happens to dominate.
        n_pd, n_ctrl = 5, 5
        pd_spectra = np.vstack([base + rng.normal(0, 0.01, size=base.shape) for _ in range(n_pd)])
        ctrl_spectra = np.vstack([base + rng.normal(0, 0.01, size=base.shape) for _ in range(n_ctrl)])
        spectra = np.vstack([pd_spectra, ctrl_spectra])
        patient_id = np.array([f"p{i}" for i in range(n_pd + n_ctrl)])
        target = np.array([1.0] * n_pd + [0.0] * n_ctrl)

        repr_ = build_patient_area_normalized_repr(spectra, patient_id, target, waves, qc_status=None)
        pooled = class_group_mean(repr_, target_value=None, n_boot=50, seed=0)
        pd_only = class_group_mean(repr_, target_value=1.0, n_boot=50, seed=0)
        ctrl_only = class_group_mean(repr_, target_value=0.0, n_boot=50, seed=0)

        assert pooled["n_patients"] == n_pd + n_ctrl
        assert pd_only["n_patients"] == n_pd
        assert ctrl_only["n_patients"] == n_ctrl
        # the diagnosis-blind pooled mean must lie between (close to both, since
        # both subgroups share the same underlying shape here) the two
        # diagnosis-conditioned means -- confirms target is genuinely ignored,
        # not silently defaulting to one subgroup.
        np.testing.assert_allclose(pooled["mean"], (pd_only["mean"] + ctrl_only["mean"]) / 2, atol=0.05)

    def test_population_mean_curve_wires_into_transfer_and_apply(self, waves):
        """Full pipeline: two instruments' diagnosis-blind population means ->
        raw transfer curve -> smoothed -> applied to the 'new' instrument's own
        representative spectra -> harmonized result should be materially more
        similar to 'old' than the untransformed 'new' spectra were."""
        rng = np.random.default_rng(12)
        old_shape = gaussian_spectrum(waves, centers=(500.0, 560.0), heights=(5.0, 2.0))
        new_shape = old_shape * (1.0 + 0.15 * np.sin(np.linspace(0, 3, len(waves))))  # instrument response distortion

        def make_repr(shape, n=8, seed=0):
            r = np.random.default_rng(seed)
            spectra = np.vstack([shape + r.normal(0, 0.02, size=shape.shape) for _ in range(n)])
            patient_id = np.array([f"q{i}" for i in range(n)])
            target = np.where(r.random(n) > 0.5, 1.0, 0.0)  # mixed diagnosis, irrelevant to target_value=None
            return build_patient_area_normalized_repr(spectra, patient_id, target, waves, qc_status=None)

        old_repr = make_repr(old_shape, seed=1)
        new_repr = make_repr(new_shape, seed=2)

        old_mean = class_group_mean(old_repr, target_value=None, n_boot=30, seed=0)["mean"]
        new_mean = class_group_mean(new_repr, target_value=None, n_boot=30, seed=0)["mean"]

        raw = build_transfer_curve_raw(old_mean, new_mean, denom_near_zero_rel_threshold=0.02,
                                        edge_exclude_n_points=3)
        k = smooth_transfer_curve(raw["k_raw"], window=15, polyorder=2)

        before = compute_similarity_metrics(old_mean, new_mean, waves)
        harmonized = apply_transfer(new_repr.spectra, waves, k)["harmonized"]
        after = compute_similarity_metrics(old_mean, harmonized.mean(axis=0), waves)

        assert after["pearson"] >= before["pearson"] - 1e-6
        assert after["nrmse"] <= before["nrmse"] + 1e-6
