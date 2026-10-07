import pandas as pd
import pytest

from spectra_ml.identity import compute_merge_audit, resolve_canonical_patient_ids
from spectra_ml.schema import DataConfig, InstrumentConfig

# Column names used by the synthetic DataConfig below -- deliberately different
# strings from the real 'ID исследования' etc. to make sure identity.py only
# relies on config-driven column names, never hardcoded Russian headers.
INSTR, MG, GUID, AGE, SEX, TARGET, TS = "instrument", "mg_id", "guid", "age", "sex", "target", "ts"


def make_config() -> DataConfig:
    return DataConfig(
        data_path="unused.xlsx",
        sheet_name="unused",
        patient_id_col=GUID,
        legacy_mg_id_col=MG,
        instrument_col=INSTR,
        target_col=TARGET,
        target_map={"yes": 1, "no": 0},
        wavelength_header_row_hint="unused",
        wavelength_min_nm=0.0,
        wavelength_max_nm=1.0,
        random_seed=0,
        age_col=AGE,
        sex_col=SEX,
        timestamp_col=TS,
    )


def make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestSameSessionMerge:
    def test_matching_demographics_no_timestamps_merged(self):
        df = make_df([
            {INSTR: "A", MG: "MG-1", GUID: "g2", AGE: 60, SEX: "f", TARGET: "no", TS: None},
            {INSTR: "A", MG: "MG-1", GUID: "g1", AGE: 60, SEX: "f", TARGET: "no", TS: None},
        ])
        canonical = resolve_canonical_patient_ids(df, make_config())
        assert canonical.nunique() == 1
        # deterministic: lexicographically smallest GUID wins
        assert set(canonical.unique()) == {"g1"}

    def test_matching_date_timestamps_merged(self):
        df = make_df([
            {INSTR: "A", MG: "MG-1", GUID: "g1", AGE: 60, SEX: "f", TARGET: "no", TS: "27.01.2026 12:22:16"},
            {INSTR: "A", MG: "MG-1", GUID: "g2", AGE: 60, SEX: "f", TARGET: "no", TS: "27.01.2026 12:24:03"},
        ])
        canonical = resolve_canonical_patient_ids(df, make_config())
        assert canonical.nunique() == 1

    def test_conflicting_dates_not_merged(self):
        """Synthetic (no such case exists in the real data): a genuine repeat
        visit on a different calendar day must stay separate, even with
        identical demographics and the same legacy mg-id."""
        df = make_df([
            {INSTR: "A", MG: "MG-1", GUID: "g1", AGE: 60, SEX: "f", TARGET: "no", TS: "27.01.2026 12:22:16"},
            {INSTR: "A", MG: "MG-1", GUID: "g2", AGE: 60, SEX: "f", TARGET: "no", TS: "03.03.2026 09:00:00"},
        ])
        canonical = resolve_canonical_patient_ids(df, make_config())
        assert canonical.nunique() == 2
        assert set(canonical.unique()) == {"g1", "g2"}

    def test_conflicting_age_not_merged(self):
        df = make_df([
            {INSTR: "A", MG: "MG-1", GUID: "g1", AGE: 60, SEX: "f", TARGET: "no", TS: None},
            {INSTR: "A", MG: "MG-1", GUID: "g2", AGE: 45, SEX: "f", TARGET: "no", TS: None},
        ])
        canonical = resolve_canonical_patient_ids(df, make_config())
        assert canonical.nunique() == 2

    def test_conflicting_sex_not_merged(self):
        df = make_df([
            {INSTR: "A", MG: "MG-1", GUID: "g1", AGE: 60, SEX: "f", TARGET: "no", TS: None},
            {INSTR: "A", MG: "MG-1", GUID: "g2", AGE: 60, SEX: "m", TARGET: "no", TS: None},
        ])
        canonical = resolve_canonical_patient_ids(df, make_config())
        assert canonical.nunique() == 2

    def test_conflicting_target_not_merged(self):
        df = make_df([
            {INSTR: "A", MG: "MG-1", GUID: "g1", AGE: 60, SEX: "f", TARGET: "no", TS: None},
            {INSTR: "A", MG: "MG-1", GUID: "g2", AGE: 60, SEX: "f", TARGET: "yes", TS: None},
        ])
        canonical = resolve_canonical_patient_ids(df, make_config())
        assert canonical.nunique() == 2

    def test_mg_id_present_once_unchanged(self):
        df = make_df([
            {INSTR: "A", MG: "MG-1", GUID: "g1", AGE: 60, SEX: "f", TARGET: "no", TS: None},
            {INSTR: "A", MG: "MG-2", GUID: "g2", AGE: 70, SEX: "m", TARGET: "yes", TS: None},
        ])
        canonical = resolve_canonical_patient_ids(df, make_config())
        assert list(canonical) == ["g1", "g2"]

    def test_missing_mg_id_unchanged(self):
        df = make_df([
            {INSTR: "A", MG: None, GUID: "g1", AGE: 60, SEX: "f", TARGET: "no", TS: None},
            {INSTR: "A", MG: None, GUID: "g2", AGE: 60, SEX: "f", TARGET: "no", TS: None},
        ])
        canonical = resolve_canonical_patient_ids(df, make_config())
        # both have a missing mg_id -> never grouped together -> unchanged
        assert list(canonical) == ["g1", "g2"]

    def test_cross_instrument_same_mg_id_not_merged(self):
        """MG-ids are independently assigned per instrument (configs/data.yaml);
        the same mg_id under two different instruments must never be merged."""
        df = make_df([
            {INSTR: "A", MG: "MG-1", GUID: "g1", AGE: 60, SEX: "f", TARGET: "no", TS: None},
            {INSTR: "B", MG: "MG-1", GUID: "g2", AGE: 60, SEX: "f", TARGET: "no", TS: None},
        ])
        canonical = resolve_canonical_patient_ids(df, make_config())
        assert list(canonical) == ["g1", "g2"]

    def test_missing_fields_treated_as_compatible(self):
        df = make_df([
            {INSTR: "A", MG: "MG-1", GUID: "g1", AGE: None, SEX: "f", TARGET: "no", TS: None},
            {INSTR: "A", MG: "MG-1", GUID: "g2", AGE: 60, SEX: None, TARGET: "no", TS: None},
        ])
        canonical = resolve_canonical_patient_ids(df, make_config())
        assert canonical.nunique() == 1


class TestEndToEndRealData:
    """Reproduces the bug's documented scope directly against
    data/patients_database.xlsx. NOTE: our independent reproduction of the
    'new_control' (new2) instrument DIFFERS from the number originally
    reported when this fix was scoped (0 merge-groups expected there): the
    real data contains 14 same-mg-id, exactly-2-GUID groups on new_control
    with consistent age/sex/target and same-day timestamps on all 14 pairs
    (byte-for-byte the same evidence pattern as new_pd's 14) -- i.e. these are
    same-session two-hand pairs there too, and are correctly merged by this
    algorithm. Total: 60 (old) + 14 (new_pd) + 14 (new_control) = 88 merge
    groups, not 74. Collision (rejected) groups: 1 on new_pd (MG-1-7, target
    conflict да/нет) + 1 on new_control (MG-2-13, age 51 vs 52) = 2 single-
    instrument collisions -- consistent with the brief's separately-documented
    '1 single-instrument' collision count (the brief's '~20' also includes 19
    cross-instrument mg-id collisions, which this per-instrument-grouped
    algorithm correctly never even considers merging)."""

    def _load(self, path):
        return InstrumentConfig.from_yaml(path)

    def test_old_instrument_merge_counts(self):
        inst = self._load("configs/instrument_old.yaml")
        from spectra_ml.data import load_raw_table
        raw = load_raw_table(inst.data)
        audit = compute_merge_audit(raw, inst.data)
        by_instr = audit["before_after_patient_counts_by_instrument"][inst.raw_device_string]
        assert by_instr["n_merge_groups"] == 60
        assert by_instr["n_rejected_collision_groups"] == 0
        assert by_instr["n_patients_before_raw_guid"] == 226
        assert by_instr["n_patients_after_canonical_id"] == 166

    def test_new_pd_instrument_merge_counts(self):
        inst = self._load("configs/instrument_new1.yaml")
        from spectra_ml.data import load_raw_table
        raw = load_raw_table(inst.data)
        audit = compute_merge_audit(raw, inst.data)
        by_instr = audit["before_after_patient_counts_by_instrument"][inst.raw_device_string]
        assert by_instr["n_merge_groups"] == 14
        assert by_instr["n_rejected_collision_groups"] == 1
        assert by_instr["n_patients_before_raw_guid"] == 34
        assert by_instr["n_patients_after_canonical_id"] == 20

    def test_new_control_instrument_merge_counts(self):
        """See class docstring: this is 14, not 0 as originally scoped --
        confirmed discrepancy vs the task brief, reported rather than adjusted."""
        inst = self._load("configs/instrument_new2_controls_only.yaml")
        from spectra_ml.data import load_raw_table
        raw = load_raw_table(inst.data)
        audit = compute_merge_audit(raw, inst.data)
        by_instr = audit["before_after_patient_counts_by_instrument"][inst.raw_device_string]
        assert by_instr["n_merge_groups"] == 14
        assert by_instr["n_rejected_collision_groups"] == 1
        assert by_instr["n_patients_before_raw_guid"] == 36
        assert by_instr["n_patients_after_canonical_id"] == 22

    def test_build_dataset_patient_id_is_canonical_and_session_id_is_raw_guid(self):
        from spectra_ml.data import build_dataset
        inst = self._load("configs/instrument_old.yaml")
        ds = build_dataset(inst)
        assert len(set(ds.session_id)) == 226
        assert len(set(ds.patient_id)) == 166
        # every merged pair collapses to the lexicographically smaller GUID,
        # which must itself be one of the session ids in that pair
        assert set(ds.patient_id) <= set(ds.session_id)

    def test_patient_table_n_spectra_sums_across_merged_sessions(self):
        from spectra_ml.data import build_dataset
        inst = self._load("configs/instrument_old.yaml")
        ds = build_dataset(inst)
        pat = ds.patient_table()
        assert pat["n_spectra"].sum() == ds.n_spectra
        assert len(pat) == 166
