import numpy as np
import pytest

from spectra_ml.interpretability import (aggregate_wavelength_regions, back_project_through_scaler,
                                          fold_stability, group_permutation_importance,
                                          linear_coefficients_in_feature_space, rank_biserial_like_rank)
from spectra_ml.preprocessing import SpectralStandardScaler, build_preprocessing_pipeline


class TestBackProjectThroughScaler:
    def test_variant_a_rescales_by_inverse_std(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5, scale=3, size=(50, 10))
        pipe = build_preprocessing_pipeline("A")
        pipe.fit(X)
        coef = rng.normal(size=10)
        backprojected, was_backprojected = back_project_through_scaler(coef, pipe)
        assert was_backprojected
        scaler: SpectralStandardScaler = pipe.named_steps["scaler"]
        assert np.allclose(backprojected, coef / (scaler.std_ + 1e-12))

    def test_variant_b_not_backprojected(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5, scale=3, size=(50, 20)) + np.linspace(0, 5, 20)
        pipe = build_preprocessing_pipeline("B")
        pipe.fit(X)
        coef = rng.normal(size=20)
        out, was_backprojected = back_project_through_scaler(coef, pipe)
        assert not was_backprojected
        assert np.array_equal(out, coef)  # returned unchanged, not silently wrong


class TestRankByAbsCoef:
    def test_largest_abs_gets_rank_1(self):
        coef = np.array([0.1, -5.0, 0.3, 2.0])
        ranks = rank_biserial_like_rank(coef)
        assert ranks[1] == 1  # -5.0 has largest |coef|
        assert ranks[0] == 4  # 0.1 has smallest


class TestGroupPermutationImportance:
    def test_higher_importance_in_true_signal_region(self):
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        n_wave, n_patients = 30, 80
        waves = np.linspace(400, 680, n_wave)
        signal_region = slice(10, 15)

        patient_id, target, spectra = [], [], []
        for cls in (0, 1):
            for p in range(n_patients // 2):
                base = rng.normal(size=n_wave) * 0.2
                base[signal_region] += 2.0 * cls
                for _ in range(3):  # a few spectra per patient
                    spectra.append(base + rng.normal(scale=0.1, size=n_wave))
                    patient_id.append(f"{cls}_{p}")
                    target.append(float(cls))
        spectra, patient_id, target = np.array(spectra), np.array(patient_id), np.array(target)

        pipe = build_preprocessing_pipeline("A")
        Xt = pipe.fit_transform(spectra)
        model = LogisticRegression(max_iter=1000).fit(Xt, target)

        imp = group_permutation_importance(pipe, model, spectra, target, patient_id, waves,
                                            group_size=5, n_repeats=10, seed=1)
        signal_row = imp[(imp["wavelength_start"] <= waves[12]) & (imp["wavelength_end"] >= waves[12])]
        noise_row = imp.iloc[[-1]]  # last group, far from the signal region
        assert signal_row["importance_mean"].iloc[0] > noise_row["importance_mean"].iloc[0]


class TestAggregateWavelengthRegions:
    def test_groups_contiguous_points(self):
        waves = np.arange(10, dtype=float)
        values = np.array([1, 1, 1, 5, 5, 1, 1, 1, 1, 1], dtype=float)
        out = aggregate_wavelength_regions(values, waves, region_size=5)
        assert len(out) == 2
        assert out.iloc[0]["sum_abs_importance"] == pytest.approx(1 + 1 + 1 + 5 + 5)


class TestFoldStability:
    def test_topk_fraction(self):
        waves = np.arange(5, dtype=float)
        per_fold = np.array([
            [5.0, 1, 1, 1, 1],  # wavelength 0 is top-1
            [1.0, 5, 1, 1, 1],  # wavelength 1 is top-1
        ])
        out = fold_stability(per_fold, waves, top_k=1)
        assert out.loc[0, "fraction_folds_in_topK"] == 0.5
        assert out.loc[1, "fraction_folds_in_topK"] == 0.5
        assert out.loc[2, "fraction_folds_in_topK"] == 0.0
