import numpy as np
import pandas as pd
import pytest

from spectra_ml.united_stage import _stage_series_from_canonical_ids


class TestStageSeriesFromCanonicalIds:
    def test_consistent_stage_across_merged_pair_joins_correctly(self):
        """Synthetic same-session two-hand pair (raw GUIDs g1/g2 merged onto
        canonical id 'g1', as spectra_ml.identity would do for a same-session
        pair with matching demographics): both underlying rows report the SAME
        Hoehn-Yahr stage value -> the canonical-id-keyed series must expose a
        single, correct row for 'g1' with that stage."""
        canonical_id = pd.Series(["g1", "g1", "g3"])
        stage_values = pd.Series([2.0, 2.0, -1.0])
        series = _stage_series_from_canonical_ids(canonical_id, stage_values)
        assert series.loc["g1"] == 2.0
        assert series.loc["g3"] == -1.0
        assert len(series) == 2

    def test_one_missing_one_present_stage_resolves_to_present_value(self):
        """One of the two merged rows has a missing (NaN) stage value (e.g. the
        clinical form was only filled in for one hand's row) -- this is not a
        conflict (missing is compatible with anything), the present value wins."""
        canonical_id = pd.Series(["g1", "g1"])
        stage_values = pd.Series([np.nan, 3.0])
        series = _stage_series_from_canonical_ids(canonical_id, stage_values)
        assert series.loc["g1"] == 3.0

    def test_conflicting_stage_across_merged_pair_raises(self):
        """A canonical id whose constituent (identity-merged) rows disagree on
        a non-null Hoehn-Yahr stage value must raise, not silently pick one --
        spectra_ml.identity's merge rule does not itself check stage
        consistency (only age/sex/target/timestamp), so this check must
        happen here."""
        canonical_id = pd.Series(["g1", "g1"])
        stage_values = pd.Series([2.0, 3.0])
        with pytest.raises(ValueError, match="conflicting Hoehn-Yahr stage"):
            _stage_series_from_canonical_ids(canonical_id, stage_values)

    def test_all_missing_stage_for_a_canonical_id_is_nan_not_a_conflict(self):
        canonical_id = pd.Series(["g1", "g1"])
        stage_values = pd.Series([np.nan, np.nan])
        series = _stage_series_from_canonical_ids(canonical_id, stage_values)
        assert np.isnan(series.loc["g1"])

    def test_unmerged_singleton_ids_pass_through_unchanged(self):
        canonical_id = pd.Series(["g1", "g2", "g3"])
        stage_values = pd.Series([1.0, 2.0, 3.0])
        series = _stage_series_from_canonical_ids(canonical_id, stage_values)
        assert series.loc["g1"] == 1.0
        assert series.loc["g2"] == 2.0
        assert series.loc["g3"] == 3.0
