import numpy as np
import pandas as pd
import pytest

from spectra_ml.bootstrap import bootstrap_patient_metrics, paired_bootstrap_compare


def make_patient_df(n_park=40, n_control=40, seed=0, separation=2.0):
    rng = np.random.default_rng(seed)
    target = np.array([1.0] * n_park + [0.0] * n_control)
    logit = np.where(target == 1, rng.normal(separation, 1, n_park + n_control),
                      rng.normal(-separation, 1, n_park + n_control))
    prob = 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n_park + n_control)],
        "target": target, "probability": prob,
    })


class TestBootstrapPatientMetrics:
    def test_stratified_resample_keeps_class_counts(self):
        from spectra_ml.bootstrap import _stratified_resample_indices
        target = np.array([0.0] * 30 + [1.0] * 70)
        rng = np.random.default_rng(0)
        idx = _stratified_resample_indices(target, rng)
        resampled = target[idx]
        assert (resampled == 0).sum() == 30
        assert (resampled == 1).sum() == 70

    def test_ci_covers_point_estimate(self):
        df = make_patient_df(seed=1)
        replicates, summary = bootstrap_patient_metrics(df, prob_col="probability", n_boot=500, seed=1)
        from spectra_ml.evaluation import compute_patient_metrics
        point = compute_patient_metrics(df["target"].to_numpy(), df["probability"].to_numpy())
        auc_row = summary[summary["metric"] == "roc_auc"].iloc[0]
        assert auc_row["ci95_lo"] <= point["roc_auc"] <= auc_row["ci95_hi"]

    def test_replicates_shape(self):
        df = make_patient_df(seed=2)
        replicates, summary = bootstrap_patient_metrics(df, prob_col="probability", n_boot=100, seed=2)
        assert len(replicates) == 100
        assert set(["roc_auc", "brier"]).issubset(replicates.columns)

    def test_reproducible_with_seed(self):
        df = make_patient_df(seed=3)
        _, summary_a = bootstrap_patient_metrics(df, prob_col="probability", n_boot=200, seed=5)
        _, summary_b = bootstrap_patient_metrics(df, prob_col="probability", n_boot=200, seed=5)
        pd.testing.assert_frame_equal(summary_a, summary_b)

    def test_weak_model_has_ci_spanning_chance(self):
        """A near-random classifier's ROC-AUC 95% CI should plausibly include 0.5."""
        df = make_patient_df(seed=4, separation=0.0)
        _, summary = bootstrap_patient_metrics(df, prob_col="probability", n_boot=1000, seed=4)
        auc_row = summary[summary["metric"] == "roc_auc"].iloc[0]
        assert auc_row["ci95_lo"] < 0.5 < auc_row["ci95_hi"]


class TestPairedBootstrapCompare:
    def test_identical_models_not_significant(self):
        df = make_patient_df(seed=6)
        result = paired_bootstrap_compare(df, df, prob_col="probability", metric="roc_auc", n_boot=500, seed=6)
        assert result["mean_delta"] == pytest.approx(0.0, abs=1e-9)
        assert result["ci95_lo"] <= 0 <= result["ci95_hi"]
        assert not result["significant_at_0.05"]

    def test_clearly_better_model_is_significant(self):
        df_good = make_patient_df(seed=7, separation=3.0)
        df_bad = df_good.copy()
        rng = np.random.default_rng(8)
        df_bad["probability"] = rng.uniform(0, 1, len(df_bad))  # pure noise, same patients
        result = paired_bootstrap_compare(df_good, df_bad, prob_col="probability", metric="roc_auc",
                                           n_boot=500, seed=7)
        assert result["mean_delta"] > 0
        assert result["significant_at_0.05"]

    def test_requires_matching_patients(self):
        df_a = make_patient_df(seed=9)
        df_b = make_patient_df(seed=9).iloc[:-1]  # one fewer patient
        with pytest.raises(Exception):
            paired_bootstrap_compare(df_a, df_b, prob_col="probability", n_boot=50, seed=9)
