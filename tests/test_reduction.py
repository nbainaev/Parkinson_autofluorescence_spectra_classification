import numpy as np

from spectra_ml.reduction import PCAFeatureExtractor


class TestPCABackProjection:
    def test_back_projected_coefficient_reproduces_pca_space_score(self):
        """coef_orig @ (X - mean_) should equal coef_pca @ transform(X) for
        every row -- i.e. the back-projected coefficient computes the exact
        same linear score in original feature space as the original
        coefficient computed in PCA-component space (up to the mean-centering
        offset PCA applies internally, which is constant across rows and does
        not affect back-projection correctness)."""
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(30, 12))
        X_test = rng.normal(size=(8, 12))

        pca = PCAFeatureExtractor(n_components=4).fit(X_train)
        coef_pca = rng.normal(size=4)

        Z_test = pca.transform(X_test)
        score_in_pca_space = Z_test @ coef_pca

        coef_orig = pca.back_project_coefficients(coef_pca)
        mean_ = pca.model_.mean_
        score_in_orig_space = (X_test - mean_) @ coef_orig

        np.testing.assert_allclose(score_in_pca_space, score_in_orig_space, atol=1e-8)

    def test_back_projected_coefficient_shape_matches_original_features(self):
        rng = np.random.default_rng(1)
        X_train = rng.normal(size=(20, 15))
        pca = PCAFeatureExtractor(n_components=5).fit(X_train)
        coef_pca = rng.normal(size=5)
        coef_orig = pca.back_project_coefficients(coef_pca)
        assert coef_orig.shape == (15,)
