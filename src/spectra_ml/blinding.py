"""Shared helpers for the retrospective blinded study manifests
(current_tasks/double_blind_exp.md Parts II-III, XXII-XXIV).

Deliberately tiny and dumb: this module only reads/filters the CSV manifests
produced by scripts/prepare_blinded_study.py. It never computes a split,
never fits anything, and never decides what may or may not be done with the
data it returns -- that responsibility stays in each calling script, so the
one-directional flow (prepare -> transfer -> freeze -> predict -> unblind)
stays easy to audit against Part XXXIV's forbidden-actions list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_development_manifest(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"patient_id": str})


def development_patient_ids_by_role(manifest_df: pd.DataFrame) -> dict[str, set]:
    return {role: set(g["patient_id"]) for role, g in manifest_df.groupby("instrument_role")}


def load_blinded_test_manifest(path: str) -> pd.DataFrame:
    """study_id, instrument_role, instrument_id, cohort -- NEVER target/real_patient_id
    (those columns do not exist in this file at all, by construction)."""
    df = pd.read_csv(path, dtype={"study_id": str})
    forbidden = {"target", "real_patient_id"}
    leaked = forbidden & set(df.columns)
    if leaked:
        raise ValueError(f"blinded_test_manifest.csv unexpectedly contains {leaked} -- refusing to proceed blind-safe.")
    return df


def load_blinding_key(path: str) -> pd.DataFrame:
    """study_id, real_patient_id, target, instrument_role, instrument_id, cohort.
    Caller: ONLY unblind_and_evaluate.py, after predictions are frozen. The
    prediction process uses a physically separate target-free identity lookup.
    No training/prediction/hyperparameter-selection/threshold-selection code
    may import this function."""
    return pd.read_csv(path, dtype={"study_id": str, "real_patient_id": str})


def restrict_dataset_to_patients(ds, patient_ids: set):
    """In-place-ish filter of a spectra_ml.data.SpectraDataset to only rows
    whose patient_id is in `patient_ids`. Returns the same object for chaining."""
    keep = np.isin(ds.patient_id, list(patient_ids))
    if not keep.any():
        raise ValueError(f"no patients from the given id set found for instrument {ds.instrument.id!r}.")
    ds.patient_id = ds.patient_id[keep]
    ds.legacy_mg_id = ds.legacy_mg_id[keep]
    ds.target = ds.target[keep]
    ds.spectra = ds.spectra[keep]
    return ds
