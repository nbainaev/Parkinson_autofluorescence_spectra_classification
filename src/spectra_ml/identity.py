"""Real-person identity resolution for patients_database.xlsx.

Bug this fixes: `configs/data.yaml` uses `patient_id_col` = 'ID исследования'
(a per-visit GUID) as the patient/grouping key, instead of the legacy
'ID пациента' (MG-1-N) id, because the legacy id has confirmed collisions
where two DIFFERENT real people share one MG-id (see configs/data.yaml). That
fix was correct for the collision problem, but it introduced the opposite
bug: when the same real person had both hands measured in one session, each
hand got its own GUID row, and every patient-level operation downstream
(splitting, grouped CV, bootstrap, representative-spectrum averaging) has
been counting each hand as a separate "patient".

This module assigns a canonical, real-person-level id to every row, WITHOUT
discarding the raw GUID (kept as `SpectraDataset.session_id` -- see data.py)
and without touching the genuine MG-id collision cases (two GUIDs sharing an
MG-id but with differing demographics/target: these are two different real
people incorrectly assigned the same legacy id upstream, and must stay
separate; GUID already does that correctly).

Merge rule (only within one instrument's own legacy_mg_id_col value -- MG-ids
are independently assigned per instrument/department, see configs/data.yaml,
so cross-instrument matches on legacy_mg_id_col are NEVER inspected here):
two or more GUIDs sharing one (instrument, legacy_mg_id) are merged into a
single canonical id iff age, sex, and target are pairwise consistent (missing
values are treated as compatible with anything) AND the recorded timestamps
either are all missing, or all agree on calendar date. If either check fails,
the group is a genuine collision (or a genuine repeat visit on a different
day) and every GUID in it keeps its own canonical id (= itself), unchanged.

The canonical id chosen for a merged group is the lexicographically smallest
GUID in the group, so the mapping is deterministic and reproducible.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pandas as pd

from spectra_ml.schema import DataConfig

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v)


def _blank_to_none(v):
    """Treat NaN/None and blank/whitespace-only strings as 'missing'."""
    if _is_missing(v):
        return None
    s = str(v).strip()
    return s if s else None


def _extract_date_part(ts) -> str | None:
    """Extract the calendar-date portion of a `.Время`-style timestamp, e.g.
    '27.01.2026 12:22:16' -> '27.01.2026'. No locale/date library is assumed
    available or reliable for this column's exact format; this is a plain
    string split on the first whitespace/'T', which is robust to the observed
    'DD.MM.YYYY HH:MM:SS' format without assuming a fixed timezone or locale.
    Returns None for missing/blank values."""
    s = _blank_to_none(ts)
    if s is None:
        return None
    return s.split(" ")[0].split("T")[0]


def _field_consistent(values: list) -> tuple[bool, list]:
    """`values`: raw column values for one (instrument, mg_id) group. Missing
    values are compatible with anything (they simply don't constrain the
    check). Returns (is_consistent, distinct_non_missing_values_seen)."""
    distinct = []
    for v in values:
        if _is_missing(v):
            continue
        if v not in distinct:
            distinct.append(v)
    return (len(distinct) <= 1), distinct


@dataclass
class MergeDecision:
    instrument: str
    legacy_mg_id: str
    guids: list
    merged: bool
    canonical_id: str | None
    reason: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "legacy_mg_id": self.legacy_mg_id,
            "guids": self.guids,
            "merged": self.merged,
            "canonical_id": self.canonical_id,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def _require_identity_columns(config: DataConfig) -> None:
    missing = [name for name in ("age_col", "sex_col", "timestamp_col") if getattr(config, name) is None]
    if missing:
        raise ValueError(
            f"DataConfig is missing identity-merge column(s) {missing}: cannot run "
            "spectra_ml.identity.resolve_canonical_patient_ids without them (set them in "
            "configs/data.yaml)."
        )


def _decide_groups(df_full: pd.DataFrame, config: DataConfig) -> list[MergeDecision]:
    """One MergeDecision per (instrument, legacy_mg_id) group that has >1
    distinct GUID. Groups with a missing legacy_mg_id, or with only one
    distinct GUID, are not decisions (there is nothing to merge or reject) --
    those rows keep their own GUID as their canonical id implicitly."""
    _require_identity_columns(config)
    guid_col, mg_col, instr_col = config.patient_id_col, config.legacy_mg_id_col, config.instrument_col
    age_col, sex_col, target_col, ts_col = config.age_col, config.sex_col, config.target_col, config.timestamp_col

    work = df_full[[instr_col, mg_col, guid_col, age_col, sex_col, target_col, ts_col]].copy()
    work["_mg_norm"] = work[mg_col].map(_blank_to_none)
    work = work[work["_mg_norm"].notna() & work[guid_col].map(lambda g: not _is_missing(g))]

    decisions: list[MergeDecision] = []
    for (instrument, mg_id), g in work.groupby([instr_col, "_mg_norm"], sort=False):
        guids = sorted(set(g[guid_col].tolist()))
        if len(guids) <= 1:
            continue

        age_ok, age_vals = _field_consistent(g[age_col].tolist())
        sex_ok, sex_vals = _field_consistent(g[sex_col].tolist())
        target_ok, target_vals = _field_consistent(g[target_col].tolist())

        dates = [_extract_date_part(t) for t in g[ts_col].tolist()]
        dates_present = sorted({d for d in dates if d is not None})
        date_ok = len(dates_present) <= 1

        demo_ok = age_ok and sex_ok and target_ok
        evidence = {"age": age_vals, "sex": sex_vals, "target": target_vals, "dates": dates_present}

        if demo_ok and date_ok:
            decisions.append(MergeDecision(
                instrument=str(instrument), legacy_mg_id=str(mg_id), guids=guids,
                merged=True, canonical_id=guids[0],
                reason=("same legacy_mg_id, same instrument, consistent age/sex/target, and dates "
                        "compatible (both missing or same calendar day) -> same-session merge"),
                evidence=evidence,
            ))
        else:
            reasons = []
            if not age_ok:
                reasons.append(f"age conflict {age_vals}")
            if not sex_ok:
                reasons.append(f"sex conflict {sex_vals}")
            if not target_ok:
                reasons.append(f"target conflict {target_vals}")
            if not date_ok:
                reasons.append(f"date conflict {dates_present}")
            decisions.append(MergeDecision(
                instrument=str(instrument), legacy_mg_id=str(mg_id), guids=guids,
                merged=False, canonical_id=None,
                reason="; ".join(reasons),
                evidence=evidence,
            ))
    return decisions


def resolve_canonical_patient_ids(df_full: pd.DataFrame, data_config: DataConfig) -> pd.Series:
    """Returns a pd.Series indexed like `df_full`, values = canonical
    (real-person-level) patient id per row. Rows whose legacy_mg_id is
    missing, or whose legacy_mg_id has only one distinct GUID, or whose GUID
    is itself missing, keep their raw GUID unchanged as their canonical id."""
    guid_col = data_config.patient_id_col
    decisions = _decide_groups(df_full, data_config)

    guid_to_canonical = {}
    for d in decisions:
        if d.merged:
            for g in d.guids:
                guid_to_canonical[g] = d.canonical_id

    raw = df_full[guid_col]
    if guid_to_canonical:
        canonical = raw.map(lambda g: guid_to_canonical.get(g, g))
    else:
        canonical = raw.copy()
    canonical.index = df_full.index
    canonical.name = "canonical_patient_id"
    return canonical


def compute_merge_audit(df_full: pd.DataFrame, data_config: DataConfig) -> dict:
    guid_col, instr_col = data_config.patient_id_col, data_config.instrument_col
    decisions = _decide_groups(df_full, data_config)
    merged = [d for d in decisions if d.merged]
    rejected = [d for d in decisions if not d.merged]

    canonical = resolve_canonical_patient_ids(df_full, data_config)

    by_instrument = {}
    for instrument, g in df_full.groupby(instr_col, dropna=True, sort=False):
        idx = g.index
        n_before = int(df_full.loc[idx, guid_col].nunique(dropna=True))
        n_after = int(canonical.loc[idx].nunique(dropna=True))
        by_instrument[str(instrument)] = {
            "n_patients_before_raw_guid": n_before,
            "n_patients_after_canonical_id": n_after,
            "n_merge_groups": sum(1 for d in merged if d.instrument == str(instrument)),
            "n_rejected_collision_groups": sum(1 for d in rejected if d.instrument == str(instrument)),
        }

    return {
        "n_rows_examined": int(len(df_full)),
        "n_merge_groups": len(merged),
        "n_guids_collapsed": sum(len(d.guids) - 1 for d in merged),
        "n_rejected_collision_groups": len(rejected),
        "before_after_patient_counts_by_instrument": by_instrument,
        "merged_groups": [d.to_dict() for d in merged],
        "rejected_collision_groups": [d.to_dict() for d in rejected],
    }


def write_merge_audit(df_full: pd.DataFrame, data_config: DataConfig, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    audit = compute_merge_audit(df_full, data_config)
    path = os.path.join(out_dir, "merge_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2, default=str)
    return audit
