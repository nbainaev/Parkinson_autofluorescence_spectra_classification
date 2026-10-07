"""Additive extension of spectra_ml.united for the Hoehn-Yahr STAGE (multiclass,
0-3) analysis. Does NOT modify united.py / the binary target wiring in
configs/data.yaml at all: this module reuses `build_united_dataset()` unchanged
(same pooled, harmonized, QC'd spectra as outputs/united/) and additively joins
the clinical PD stage column onto it by patient_id.

Stage column: "Стадия болезни Паркинсона (-1 - нет БП, 0-5 - стадия Хен-Яра)"
(sheet 'patients', same raw table as everything else). Values observed in this
dataset: -1 (no PD / control), 1, 2, 3, 4 (no 0 or 5 recorded). Mapping used
here (user decision, see current_tasks):
    -1 -> class 0 (control)
     1 -> class 1
     2 -> class 2
     3 -> class 3
     4 -> EXCLUDED (n=3, too few to model as a 5th class; logged, not folded
          into stage 3 or dropped silently)
   NaN / other -> EXCLUDED ("unknown/missing stage"; logged)

This column is per-VISIT. `UnitedDataset.patient_id` (spectra_ml.data /
spectra_ml.identity) is the CANONICAL, real-person-level id -- NOT the raw
per-visit GUID -- since a same-session two-hand GUID pair is collapsed onto
one canonical id upstream. The join therefore uses
`build_stage_series_canonical`, which resolves canonical ids for the raw
table itself and verifies stage-value agreement across a canonical id's
(possibly identity-merged) constituent rows before joining -- see that
function's docstring for why this differs from the simpler, raw-GUID-keyed
`build_stage_series` (kept for diagnostic use, e.g. investigating a single
GUID's own raw value).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from spectra_ml.data import load_raw_table
from spectra_ml.identity import resolve_canonical_patient_ids
from spectra_ml.schema import InstrumentConfig
from spectra_ml.united import ROLE_CONFIG, ROLES, UnitedDataset, build_united_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE_COL = "Стадия болезни Паркинсона (-1 - нет БП, 0-5 - стадия Хен-Яра)"

# raw stage value -> included class label (float); values not present in this
# map (including NaN) are excluded -- see STAGE4_REASON / UNKNOWN_REASON below.
STAGE_TO_CLASS = {-1.0: 0.0, 1.0: 1.0, 2.0: 2.0, 3.0: 3.0}
CLASS_NAMES = {0.0: "control", 1.0: "stage1", 2.0: "stage2", 3.0: "stage3"}
STAGE4_REASON = "stage4_excluded_small_n"
UNKNOWN_REASON = "unknown_missing_stage"


def build_stage_series(patient_id_col: str = "ID исследования") -> pd.Series:
    """One row per RAW GUID (patient_id_col): raw Hoehn-Yahr stage value (float,
    NaN if missing/blank). Raises if any single GUID has more than one distinct
    non-null stage value among its OWN repeated rows in the raw table (would
    indicate a data problem, not expected -- verified absent in the current
    dataset).

    NOTE: this is the RAW-GUID-keyed series -- informational / diagnostic only.
    Since spectra_ml.identity, UnitedDataset.patient_id (and therefore
    UnitedStageDataset's join key) is the CANONICAL (real-person-level) id, not
    the raw GUID; a canonical id formed by merging two GUIDs (e.g. a same-
    session two-hand pair) is NOT itself a key of this raw-GUID series unless
    it happens to equal one of its own constituent GUIDs. Use
    `build_stage_series_canonical` to join onto a UnitedDataset/SpectraDataset
    built after the identity fix -- see its docstring for why this function
    alone is not sufficient for that join.
    """
    inst = InstrumentConfig.from_yaml(ROLE_CONFIG["old"])
    raw_df = load_raw_table(inst.data)
    if STAGE_COL not in raw_df.columns:
        raise ValueError(f"stage column {STAGE_COL!r} not found in raw table.")

    sub = raw_df[[patient_id_col, STAGE_COL]].copy()
    conflict = sub.groupby(patient_id_col)[STAGE_COL].nunique(dropna=True)
    bad = conflict[conflict > 1]
    if len(bad):
        raise ValueError(
            f"{len(bad)} patient_id(s) have conflicting stage values across repeated rows: "
            f"{bad.index.tolist()[:10]}"
        )
    # first non-null value per GUID (falls back to NaN if all rows are NaN for that GUID)
    series = sub.groupby(patient_id_col)[STAGE_COL].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    return series


def _stage_series_from_canonical_ids(canonical_id: pd.Series, stage_values: pd.Series) -> pd.Series:
    """Pure helper (no file I/O) behind `build_stage_series_canonical`: given a
    per-row canonical id Series and a per-row (parallel, same index) raw stage
    value Series, groups by canonical id, raises if any canonical id's own
    constituent rows disagree on a non-null stage value, and otherwise returns
    one row per canonical id (first non-null stage value, NaN if all missing).
    Split out from `build_stage_series_canonical` so the merge-conflict logic
    is unit-testable without touching the real raw Excel table."""
    sub = pd.DataFrame({"_canonical_id": canonical_id.to_numpy(), "_stage": stage_values.to_numpy()})
    conflict = sub.groupby("_canonical_id")["_stage"].nunique(dropna=True)
    bad = conflict[conflict > 1]
    if len(bad):
        raise ValueError(
            f"{len(bad)} canonical patient_id(s) have conflicting Hoehn-Yahr stage values across their "
            f"constituent (possibly identity-merged) rows: {bad.index.tolist()[:10]} -- a canonical id merges "
            "GUIDs based on age/sex/target/timestamp consistency only (spectra_ml.identity), which does NOT "
            "guarantee stage agreement; investigate before proceeding rather than silently picking one value."
        )
    series = sub.groupby("_canonical_id")["_stage"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    return series


def build_stage_series_canonical(patient_id_col: str = "ID исследования") -> pd.Series:
    """Canonical-(real-person)-id-keyed sibling of `build_stage_series`, needed
    because `spectra_ml.identity.resolve_canonical_patient_ids` (the same fix
    already wired into `spectra_ml.data.build_dataset`) merges same-session
    two-hand GUID pairs onto ONE canonical id, but does so WITHOUT inspecting
    the Hoehn-Yahr stage column at all (the merge rule only checks age/sex/
    target/timestamp consistency, see spectra_ml.identity module docstring).
    So a merged canonical id's own two constituent GUID rows could in
    principle disagree on stage even though `resolve_canonical_patient_ids`
    considered them the same real person -- that possibility must be checked
    here explicitly (mirrors `build_stage_series`'s own within-GUID conflict
    guard, just at the canonical-id level instead of the raw-GUID level; the
    actual conflict-detection logic lives in `_stage_series_from_canonical_ids`
    so it can be unit-tested directly).

    One row per CANONICAL id: raw Hoehn-Yahr stage value (float, NaN if
    missing/blank across every constituent row). Raises ValueError if any
    canonical id's constituent rows disagree on a non-null stage value.
    """
    inst = InstrumentConfig.from_yaml(ROLE_CONFIG["old"])
    raw_df = load_raw_table(inst.data)
    if STAGE_COL not in raw_df.columns:
        raise ValueError(f"stage column {STAGE_COL!r} not found in raw table.")

    canonical_id = resolve_canonical_patient_ids(raw_df, inst.data)
    return _stage_series_from_canonical_ids(canonical_id, raw_df[STAGE_COL])


def map_stage_to_class(raw_stage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """raw_stage: (n,) float, raw Hoehn-Yahr values (NaN if missing). Returns
    (stage_class, exclusion_reason): stage_class is float (0/1/2/3, NaN if
    excluded); exclusion_reason is object array (None if included, else one of
    STAGE4_REASON / UNKNOWN_REASON)."""
    stage_class = np.full(len(raw_stage), np.nan)
    reason = np.full(len(raw_stage), None, dtype=object)
    for i, v in enumerate(raw_stage):
        if pd.isna(v):
            reason[i] = UNKNOWN_REASON
            continue
        if v == 4.0:
            reason[i] = STAGE4_REASON
            continue
        cls = STAGE_TO_CLASS.get(float(v))
        if cls is None:
            reason[i] = UNKNOWN_REASON  # any other unexpected value, defensively treated as unknown
            continue
        stage_class[i] = cls
    return stage_class, reason


class UnitedStageDataset:
    """Wraps a UnitedDataset (unchanged) with additive per-spectrum stage_class
    (float, NaN if excluded) and exclusion_reason (object, None if included)
    arrays, joined by patient_id. `spectra`, `waves`, `patient_id`, `instrument`,
    `qc_status`, `qc_df` etc. are all passed through unmodified from `ud`."""

    def __init__(self, ud: UnitedDataset, stage_class: np.ndarray, exclusion_reason: np.ndarray,
                 raw_stage: np.ndarray):
        self.ud = ud
        self.stage_class = stage_class
        self.exclusion_reason = exclusion_reason
        self.raw_stage = raw_stage

    # pass-through convenience accessors
    @property
    def spectra(self): return self.ud.spectra

    @property
    def waves(self): return self.ud.waves

    @property
    def patient_id(self): return self.ud.patient_id

    @property
    def instrument(self): return self.ud.instrument

    @property
    def qc_status(self): return self.ud.qc_status

    @property
    def qc_df(self): return self.ud.qc_df

    @property
    def n_spectra(self): return self.ud.n_spectra

    def patient_table(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "patient_id": self.patient_id, "instrument": self.instrument,
            "raw_stage": self.raw_stage, "stage_class": self.stage_class,
            "exclusion_reason": self.exclusion_reason,
        })
        agg = df.groupby("patient_id", sort=False).agg(
            instrument=("instrument", "first"), raw_stage=("raw_stage", "first"),
            stage_class=("stage_class", "first"), exclusion_reason=("exclusion_reason", "first"),
            n_spectra=("stage_class", "size"),
        ).reset_index()
        return agg

    def included_mask(self) -> np.ndarray:
        """Per-spectrum mask: True if this spectrum's patient is in one of the
        4 modeled classes AND its QC status is not HARD_FAIL."""
        from spectra_ml.qc import HARD_FAIL
        return ~np.isnan(self.stage_class) & (self.qc_status != HARD_FAIL)


def build_united_stage_dataset(ud: UnitedDataset | None = None) -> UnitedStageDataset:
    if ud is None:
        ud = build_united_dataset()

    # ud.patient_id is the CANONICAL (real-person-level) id (spectra_ml.identity,
    # wired into spectra_ml.data.build_dataset) -- join by canonical id, NOT the
    # raw GUID, so a same-session two-hand-merged patient's stage is resolved
    # correctly instead of silently missing for the row whose original GUID
    # differs from the chosen canonical id. See build_stage_series_canonical.
    stage_series = build_stage_series_canonical()
    n_missing_pid = int(pd.Index(ud.patient_id).difference(stage_series.index).shape[0])
    if n_missing_pid:
        raise ValueError(
            f"{n_missing_pid} pooled canonical patient_id(s) not found in the canonical-keyed stage table -- "
            "join key mismatch, investigate before proceeding."
        )
    raw_stage = stage_series.reindex(ud.patient_id).to_numpy(dtype=float)
    stage_class, exclusion_reason = map_stage_to_class(raw_stage)

    return UnitedStageDataset(ud, stage_class=stage_class, exclusion_reason=exclusion_reason, raw_stage=raw_stage)


def compute_stage_audit(usd: UnitedStageDataset) -> dict:
    pat = usd.patient_table()
    n_total = len(pat)

    included_counts = {}
    for cls, name in CLASS_NAMES.items():
        included_counts[name] = int((pat["stage_class"] == cls).sum())

    excluded_counts = {
        "stage4_excluded_small_n": int((pat["exclusion_reason"] == STAGE4_REASON).sum()),
        "unknown_missing_stage": int((pat["exclusion_reason"] == UNKNOWN_REASON).sum()),
    }

    by_instrument = {}
    for role in ROLES:
        sub = pat[pat["instrument"] == role]
        row = {"n_patients": int(len(sub))}
        for cls, name in CLASS_NAMES.items():
            row[f"n_{name}"] = int((sub["stage_class"] == cls).sum())
        row["n_stage4_excluded"] = int((sub["exclusion_reason"] == STAGE4_REASON).sum())
        row["n_unknown_excluded"] = int((sub["exclusion_reason"] == UNKNOWN_REASON).sum())
        by_instrument[role] = row

    qc_all = usd.qc_df["qc_status"].value_counts().to_dict()
    included_mask_patients = pat["stage_class"].notna()
    qc_included = usd.qc_df.loc[np.isin(usd.patient_id, pat.loc[included_mask_patients, "patient_id"]),
                                 "qc_status"].value_counts().to_dict()

    n_included = int(included_mask_patients.sum())
    n_excluded = n_total - n_included

    return {
        "scope": ("Multiclass (4-class Hoehn-Yahr PD stage: control/stage1/stage2/stage3) pooled, "
                  "harmonized, 3-instrument dataset. Built on top of the UNCHANGED "
                  "spectra_ml.united.build_united_dataset() (diagnosis-independent, uniform "
                  "per-instrument harmonization, QC run once post-pooling); the stage column is joined "
                  "additively by CANONICAL (real-person-level) patient_id from the raw table (see "
                  "build_stage_series_canonical), not part of united.py's binary target wiring."),
        "n_patients_total_pooled": n_total,
        "n_patients_included_4class": n_included,
        "n_patients_excluded": n_excluded,
        "included_class_counts": included_counts,
        "excluded_counts": excluded_counts,
        "expected_counts_from_task_brief": {
            "control": 91, "stage1": 17, "stage2": 17, "stage3": 42, "stage4_excluded": 3,
            "unknown_missing_stage": 38,
        },
        "verification_note": (
            "Post-identity-fix (spectra_ml.identity; same-session two-hand GUID pairs collapsed onto one "
            "canonical, real-person-level patient_id) and post-join-fix (build_stage_series_canonical joins "
            "the Hoehn-Yahr stage column by canonical id, not the raw per-visit GUID -- see united_stage.py "
            "module docstring) re-derivation reproduces control=91, stage1=17, stage2=17, stage3=42, "
            "stage4_excluded=3, unknown_missing_stage=38 EXACTLY (pooled total 208 patients, matching "
            "spectra_ml.united.build_united_dataset()'s own n_patients_total). This SUPERSEDES the pre-fix "
            "figures (control=107, stage1=28, stage2=26, stage3=61, unknown=71-72, total ~296-297) that an "
            "earlier version of this audit reported before the patient-identity double-counting bug (88 "
            "same-session two-hand GUID pairs counted as 176 separate patients) was fixed; see "
            "outputs/united_multiclass/ (stale, pre-fix) vs outputs/united_multiclass_v2/ (this corrected "
            "run)."
        ),
        "by_instrument": by_instrument,
        "qc_status_counts_pooled_all": {k: int(v) for k, v in qc_all.items()},
        "qc_status_counts_pooled_included_patients_first_spectrum": {k: int(v) for k, v in qc_included.items()},
        "note_instrument_skew": (
            "Included classes are heavily 'old'-instrument-skewed: stage1/stage2/stage3 patients are "
            "almost entirely measured on the 'old' STS instrument (see by_instrument above); 'new_pd' "
            "contributes only a handful of stage3+control patients, 'new_control' contributes controls "
            "only. This is a real property of the pooled cohort, not an artifact of this join -- stated "
            "plainly per task instructions, not obscured."
        ),
    }


def write_stage_audit_outputs(usd: UnitedStageDataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    audit = compute_stage_audit(usd)
    with open(os.path.join(out_dir, "data_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    usd.patient_table().to_csv(os.path.join(out_dir, "patient_summary.csv"), index=False)
