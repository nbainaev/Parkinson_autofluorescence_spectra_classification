"""Stage 1: data loading + automatic audit (build_project_and_exp.md section 2).

No QC, no preprocessing, no modeling here -- only: load, validate schema,
filter to one instrument, expose a clean (patient_id, target, spectra) table,
and describe what was loaded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from spectra_ml.identity import resolve_canonical_patient_ids
from spectra_ml.schema import DataConfig, InstrumentConfig, SchemaIssue, validate_schema


def load_raw_table(config: DataConfig) -> pd.DataFrame:
    df = pd.read_excel(config.resolved_data_path(), sheet_name=config.sheet_name, header=0)
    return df


def get_wavelength_columns(df: pd.DataFrame, config: DataConfig) -> tuple[list, np.ndarray]:
    """Locate wavelength columns by header value (nm), restricted to
    [wavelength_min_nm, wavelength_max_nm]. Columns are identified by parsing
    every header that looks like a comma/dot decimal number and sits after the
    marker column (config.wavelength_header_row_hint) -- not by fixed index,
    so an upstream column insertion/removal doesn't silently misalign it."""
    columns = list(df.columns)
    try:
        marker_idx = columns.index(config.wavelength_header_row_hint)
    except ValueError as e:
        raise ValueError(
            f"wavelength marker column {config.wavelength_header_row_hint!r} not found in header."
        ) from e

    wave_cols, waves = [], []
    for col in columns[marker_idx + 1:]:
        try:
            w = float(str(col).replace(",", "."))
        except ValueError:
            continue
        if config.wavelength_min_nm <= w <= config.wavelength_max_nm:
            wave_cols.append(col)
            waves.append(w)
    if not wave_cols:
        raise ValueError("no wavelength columns found in the configured [min_nm, max_nm] window.")

    order = np.argsort(waves)
    waves = np.asarray(waves)[order]
    wave_cols = [wave_cols[i] for i in order]
    if not np.all(np.diff(waves) > 0):
        raise ValueError("wavelength grid is not strictly increasing after sorting -- check for duplicate headers.")
    return wave_cols, waves


def filter_instrument(df: pd.DataFrame, instrument: InstrumentConfig) -> pd.DataFrame:
    mask = df[instrument.data.instrument_col] == instrument.raw_device_string
    sub = df.loc[mask].copy()
    if sub.empty:
        raise ValueError(f"no rows match instrument raw_device_string={instrument.raw_device_string!r}")
    return sub


@dataclass
class SpectraDataset:
    """`patient_id` is the CANONICAL, real-person-level id (spectra_ml.identity):
    two GUIDs recorded in the same session (same instrument, same legacy_mg_id,
    consistent age/sex/target, same or missing date -- typically a same-visit
    left-/right-hand pair) are collapsed onto one `patient_id` so patient-level
    operations (splitting, grouped CV, bootstrap, representative-spectrum
    averaging) count them as one patient, not two. It is NOT the raw per-visit
    GUID -- use `session_id` for per-session/per-hand traceability. See
    spectra_ml/identity.py and outputs/identity_fix/merge_audit.json."""
    instrument: InstrumentConfig
    patient_id: np.ndarray          # (n_spectra,) str, canonical real-person id (see docstring)
    session_id: np.ndarray          # (n_spectra,) str, raw per-visit GUID ('ID исследования'), unchanged
    legacy_mg_id: np.ndarray        # (n_spectra,) str, informational only
    target: np.ndarray              # (n_spectra,) float, nan if unknown
    spectra: np.ndarray             # (n_spectra, n_wavelengths) float
    waves: np.ndarray               # (n_wavelengths,) float, nm
    schema_issues: list[SchemaIssue]

    @property
    def n_spectra(self) -> int:
        return len(self.patient_id)

    def patient_table(self) -> pd.DataFrame:
        """One row per unique patient_id: target (must be unique, enforced upstream
        for FATAL conflicts), n_spectra, legacy_mg_id (first occurrence, may be
        many-to-one if the same GUID never recurs under different mg ids -- checked
        in schema validation)."""
        df = pd.DataFrame({
            "patient_id": self.patient_id,
            "legacy_mg_id": self.legacy_mg_id,
            "target": self.target,
        })
        agg = df.groupby("patient_id", sort=False).agg(
            legacy_mg_id=("legacy_mg_id", "first"),
            target=("target", "first"),
            n_spectra=("target", "size"),
        ).reset_index()
        return agg


def build_dataset(instrument: InstrumentConfig, df_full: pd.DataFrame | None = None) -> SpectraDataset:
    config = instrument.data
    if df_full is None:
        df_full = load_raw_table(config)

    issues = validate_schema(df_full, config)
    fatal = [i for i in issues if i.severity == "FATAL"]

    df = filter_instrument(df_full, instrument)
    wave_cols, waves = get_wavelength_columns(df_full, config)

    target = df[config.target_col].map(config.target_map).astype(float).to_numpy()
    spectra = df[wave_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    session_id = df[config.patient_id_col].to_numpy()
    # canonical (real-person-level) id: collapses same-session two-hand GUID
    # pairs onto one id -- see spectra_ml.identity and the SpectraDataset docstring.
    canonical_id = resolve_canonical_patient_ids(df, config).to_numpy()

    return SpectraDataset(
        instrument=instrument,
        patient_id=canonical_id,
        session_id=session_id,
        legacy_mg_id=df[config.legacy_mg_id_col].to_numpy(),
        target=target,
        spectra=spectra,
        waves=waves,
        schema_issues=issues if not fatal else issues,  # keep FATAL issues visible in the audit too
    )


def compute_audit(dataset: SpectraDataset) -> dict:
    pat = dataset.patient_table()
    n_park = int((pat["target"] == 1).sum())
    n_ctrl = int((pat["target"] == 0).sum())
    n_unknown_target = int(pat["target"].isna().sum())

    valid_spectrum_mask = ~np.isnan(dataset.spectra).all(axis=1)
    n_missing_values = int(np.isnan(dataset.spectra).sum())
    n_total_values = int(dataset.spectra.size)

    return {
        "instrument": {
            "id": dataset.instrument.id,
            "model": dataset.instrument.model,
            "serial_number": dataset.instrument.serial_number,
            "has_both_classes": dataset.instrument.has_both_classes,
        },
        "n_patients": int(pat["patient_id"].nunique()),
        "n_spectra": int(dataset.n_spectra),
        "n_spectra_all_nan": int((~valid_spectrum_mask).sum()),
        "n_parkinson_patients": n_park,
        "n_control_patients": n_ctrl,
        "n_patients_unknown_target": n_unknown_target,
        "spectra_per_patient": {
            "mean": float(pat["n_spectra"].mean()),
            "std": float(pat["n_spectra"].std()),
            "min": int(pat["n_spectra"].min()),
            "max": int(pat["n_spectra"].max()),
            "median": float(pat["n_spectra"].median()),
        },
        "n_wavelength_features": int(len(dataset.waves)),
        "wavelength_range_nm": [float(dataset.waves.min()), float(dataset.waves.max())],
        "missing_values": {
            "n_missing_cells": n_missing_values,
            "fraction_missing_cells": n_missing_values / n_total_values if n_total_values else None,
        },
        "class_balance_patients": {
            "parkinson": n_park,
            "control": n_ctrl,
            "parkinson_fraction": n_park / (n_park + n_ctrl) if (n_park + n_ctrl) else None,
        },
        "schema_issues": [i.to_dict() for i in dataset.schema_issues],
        "patient_id_definition": (
            f"canonical real-person id derived from {dataset.instrument.data.patient_id_col!r} (per-visit "
            f"GUID) via spectra_ml.identity.resolve_canonical_patient_ids -- same-session two-hand GUID "
            f"pairs sharing one {dataset.instrument.data.legacy_mg_id_col!r} are merged; raw GUID is kept "
            "as SpectraDataset.session_id. See configs/data.yaml and outputs/identity_fix/merge_audit.json."
        ),
    }


def write_audit_outputs(dataset: SpectraDataset, out_dir: str) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    audit = compute_audit(dataset)
    with open(os.path.join(out_dir, "data_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    dataset.patient_table().to_csv(os.path.join(out_dir, "patient_summary.csv"), index=False)
