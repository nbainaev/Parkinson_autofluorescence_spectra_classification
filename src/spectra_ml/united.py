"""United (pooled, harmonized, all-3-instrument) dataset construction.

This module implements the "united harmonized dataset" requested for the
broader, non-blinded, all-data descriptive + classification study (see
current_tasks -- this is explicitly NOT part of the Phase-2 blinded
validation protocol in outputs/final_study_unified/).

Pipeline (all target-blind, all deterministic, no leakage concern since this
is not a blind holdout study):
  1. Load old/new1(new_pd)/new2(new_control) via spectra_ml.data.build_dataset,
     sharing one raw-table read.
  2. Confirm patient_id (GUID) is globally unique across instruments (raise if
     not -- see configs/data.yaml's note on MG-id collisions; GUIDs are
     expected to already be collision-free).
  3. Confirm the new1/new2 wavelength grids already match OLD's grid (Phase 1
     found wavelength-shift PASS for both); only interpolate if an actual
     mismatch is detected, never double-interpolate already-aligned data.
  4. Pool the RAW (pre-QC-correction) spectra of all 3 instruments into one
     array and run spectra_ml.qc.build_qc_table + hot-pixel correction ONCE,
     post-pooling (QC rules are per-spectrum, relative-scale statistics, and
     instrument-agnostic -- see qc.py docstring).
  5. Area-normalize every corrected spectrum (row-wise, stateless).
  6. Apply the FROZEN diagnosis-independent transfer curves from
     outputs/final_study_unified/transfer/ to new1 (new_pd) and new2
     (new_control) rows only; OLD rows pass through with no K-multiplication
     (they define the reference space) -- this reuses
     spectra_ml.harmonization.apply_transfer, no reimplementation.

The resulting `UnitedDataset.spectra` is the common-scale, cross-instrument-
comparable "harmonized" representation that downstream statistics/
classification scripts treat as their base ("raw" in outputs/united/statistics
naming) input -- exactly analogous to how outputs/old's "raw" was QC-corrected,
un-normalized OLD intensity (there was no cross-instrument scale problem
there, since only one instrument was in play).

`instrument` is metadata ONLY (values "old"/"new_pd"/"new_control"): it must
never be concatenated into a model's feature matrix -- callers must keep it
out of `spectra`.
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from spectra_ml.data import SpectraDataset, build_dataset, load_raw_table
from spectra_ml.harmonization import apply_transfer, area_normalize, reindex_to_reference_grid
from spectra_ml.qc import HARD_FAIL, SOFT_WARNING, QCConfig, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRANSFER_DIR = os.path.join(ROOT, "outputs", "final_study_unified", "transfer")

# instrument role order used everywhere in this module -- fixed so pooled-row
# order is reproducible.
ROLES = ("old", "new_pd", "new_control")
ROLE_CONFIG = {
    "old": os.path.join(ROOT, "configs", "instrument_old.yaml"),
    "new_pd": os.path.join(ROOT, "configs", "instrument_new1.yaml"),
    "new_control": os.path.join(ROOT, "configs", "instrument_new2_controls_only.yaml"),
}


@dataclass
class UnitedDataset:
    spectra: np.ndarray          # (n_spectra, n_wave) harmonized (area-normalized [+K for new1/new2]), hot-pixel corrected
    waves: np.ndarray            # (n_wave,) OLD's reference wavelength grid, nm
    patient_id: np.ndarray       # (n_spectra,) GUIDs, globally unique across instruments
    legacy_mg_id: np.ndarray     # (n_spectra,) informational only
    target: np.ndarray           # (n_spectra,) float, nan = unknown diagnosis
    instrument: np.ndarray       # (n_spectra,) str in {"old","new_pd","new_control"} -- METADATA ONLY, never a model feature
    qc_status: np.ndarray        # (n_spectra,) str
    qc_df: pd.DataFrame
    hot_pixel_correction_log: list
    n_interpolated: dict         # per-role count of spectra that needed grid interpolation
    transfer_dir: str = ""       # provenance: directory the applied K(wavelength) curves were loaded from

    @property
    def n_spectra(self) -> int:
        return len(self.patient_id)

    def patient_table(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "patient_id": self.patient_id, "legacy_mg_id": self.legacy_mg_id,
            "target": self.target, "instrument": self.instrument,
        })
        agg = df.groupby("patient_id", sort=False).agg(
            legacy_mg_id=("legacy_mg_id", "first"), target=("target", "first"),
            instrument=("instrument", "first"), n_spectra=("target", "size"),
        ).reset_index()
        return agg


def _check_no_guid_collisions(ds_map: dict[str, SpectraDataset]) -> None:
    sets = {role: set(ds.patient_id) for role, ds in ds_map.items()}
    for a, b in itertools.combinations(ROLES, 2):
        inter = sets[a] & sets[b]
        if inter:
            raise ValueError(
                f"patient_id (GUID) collision between instruments {a!r} and {b!r}: "
                f"{sorted(inter)[:10]}{'...' if len(inter) > 10 else ''} -- GUIDs are assumed globally unique "
                "per configs/data.yaml; refusing to pool without resolving this."
            )


def build_united_dataset(qc_config: QCConfig | None = None, transfer_dir: str | None = None) -> UnitedDataset:
    """`transfer_dir`: directory containing k_new_pd_to_old.npy /
    k_new_control_to_old.npy / wavelength_grid.npy to apply for cross-
    instrument harmonization. Defaults to TRANSFER_DIR (the frozen Phase-2
    diagnosis-independent curves, outputs/final_study_unified/transfer/) for
    backward compatibility with every existing caller (outputs/united,
    outputs/united_multiclass, outputs/final_study_unified). Pass e.g.
    outputs/harmonization_v2/ to use the corrected, population-mean
    (target_value=None) curves built by scripts/build_transfer_v2.py instead
    -- see spectra_ml.united_stage / scripts/build_united_stage_v2_dataset.py.
    """
    if qc_config is None:
        qc_config = QCConfig.from_yaml(os.path.join(ROOT, "configs", "qc.yaml"))
    if transfer_dir is None:
        transfer_dir = TRANSFER_DIR

    inst = {role: InstrumentConfig.from_yaml(path) for role, path in ROLE_CONFIG.items()}
    raw_df = load_raw_table(inst["old"].data)  # shared read, reused for all 3 instrument filters
    ds_map = {role: build_dataset(cfg, df_full=raw_df) for role, cfg in inst.items()}

    _check_no_guid_collisions(ds_map)

    waves = ds_map["old"].waves
    n_interp = {}
    spectra_by_role = {"old": ds_map["old"].spectra}
    n_interp["old"] = 0
    for role in ("new_pd", "new_control"):
        ds = ds_map[role]
        already_aligned = ds.waves.shape == waves.shape and np.allclose(ds.waves, waves, atol=1e-4)
        if already_aligned:
            spectra_by_role[role] = ds.spectra
            n_interp[role] = 0
        else:
            spectra_by_role[role] = reindex_to_reference_grid(ds.spectra, ds.waves, waves)
            n_interp[role] = int(ds.spectra.shape[0])

    pooled_spectra_raw = np.concatenate([spectra_by_role[r] for r in ROLES], axis=0)
    pooled_patient_id = np.concatenate([ds_map[r].patient_id for r in ROLES])
    pooled_legacy_mg_id = np.concatenate([ds_map[r].legacy_mg_id for r in ROLES])
    pooled_target = np.concatenate([ds_map[r].target for r in ROLES])
    pooled_instrument = np.concatenate([np.full(ds_map[r].n_spectra, r, dtype=object) for r in ROLES])
    spectrum_id = np.array(
        [f"{r}:{pid}_{i}" for r in ROLES for i, pid in enumerate(ds_map[r].patient_id)]
    )

    # -- Stage 2 QC, run ONCE on the pooled RAW array (per-spectrum, relative-scale
    # statistics -> instrument-agnostic; see qc.py docstring and module docstring above) --
    qc_df, hp, correction_log = build_qc_table(pooled_spectra_raw, waves, pooled_patient_id, spectrum_id, qc_config)
    corrected, _ = correct_hot_pixels(pooled_spectra_raw, hp, qc_config)

    normalized = np.full_like(corrected, np.nan)
    valid = ~np.isnan(corrected).any(axis=1)
    normalized[valid] = area_normalize(corrected[valid], waves)

    k_new_pd = np.load(os.path.join(transfer_dir, "k_new_pd_to_old.npy"))
    k_new_control = np.load(os.path.join(transfer_dir, "k_new_control_to_old.npy"))
    ref_grid = np.load(os.path.join(transfer_dir, "wavelength_grid.npy"))
    if not (ref_grid.shape == waves.shape and np.allclose(ref_grid, waves, atol=1e-4)):
        raise ValueError(
            f"{transfer_dir}/wavelength_grid.npy does not match OLD's wavelength grid "
            "-- refusing to apply the transfer curves without a matching grid."
        )

    harmonized = normalized.copy()
    for role, k in (("new_pd", k_new_pd), ("new_control", k_new_control)):
        mask = (pooled_instrument == role) & valid
        if mask.any():
            harmonized[mask] = apply_transfer(normalized[mask], waves, k)["harmonized"]
    # old rows: already area-normalized above, no K applied -- OLD defines the reference space.

    return UnitedDataset(
        spectra=harmonized, waves=waves, patient_id=pooled_patient_id, legacy_mg_id=pooled_legacy_mg_id,
        target=pooled_target, instrument=pooled_instrument, qc_status=qc_df["qc_status"].to_numpy(),
        qc_df=qc_df, hot_pixel_correction_log=correction_log, n_interpolated=n_interp,
        transfer_dir=transfer_dir,
    )


def compute_audit(ud: UnitedDataset) -> dict:
    pat = ud.patient_table()
    n_park = int((pat["target"] == 1).sum())
    n_ctrl = int((pat["target"] == 0).sum())
    n_unknown = int(pat["target"].isna().sum())

    by_instrument = {}
    for role in ROLES:
        sub = pat[pat["instrument"] == role]
        sub_spec = ud.instrument == role
        by_instrument[role] = {
            "n_patients": int(len(sub)),
            "n_spectra": int(sub_spec.sum()),
            "n_parkinson_patients": int((sub["target"] == 1).sum()),
            "n_control_patients": int((sub["target"] == 0).sum()),
            "n_patients_unknown_target": int(sub["target"].isna().sum()),
            "n_wavelengths_interpolated_spectra": ud.n_interpolated[role],
            "qc_status_counts": {k: int(v) for k, v in ud.qc_df.loc[ud.instrument == role, "qc_status"]
                                  .value_counts().to_dict().items()},
        }

    return {
        "scope": ("United harmonized 3-instrument pooled dataset (old + new_pd(new1) + new_control(new2)), "
                  "target-blind harmonization using the FROZEN diagnosis-independent transfer curves from "
                  "outputs/final_study_unified/transfer/. Non-blinded, all-data descriptive/classification "
                  "study -- NOT the Phase-2 blinded validation protocol."),
        "n_patients_total": int(pat["patient_id"].nunique()),
        "n_spectra_total": int(ud.n_spectra),
        "n_parkinson_patients": n_park,
        "n_control_patients": n_ctrl,
        "n_patients_unknown_target": n_unknown,
        "class_balance_patients": {
            "parkinson": n_park, "control": n_ctrl,
            "parkinson_fraction": n_park / (n_park + n_ctrl) if (n_park + n_ctrl) else None,
        },
        "n_wavelength_features": int(len(ud.waves)),
        "wavelength_range_nm": [float(ud.waves.min()), float(ud.waves.max())],
        "by_instrument": by_instrument,
        "qc_status_counts_pooled": {k: int(v) for k, v in ud.qc_df["qc_status"].value_counts().to_dict().items()},
        "harmonization": {
            "method": "area-normalize -> multiply by frozen diagnosis-independent K(wavelength) -> re-area-normalize"
                      " for new_pd/new_control; OLD area-normalized only (no K, reference space).",
            "transfer_curve_source": f"{ud.transfer_dir} (see spectra_ml.united.build_united_dataset's "
                                     "transfer_dir parameter for provenance)",
        },
        "patient_id_definition": "'ID исследования' (per-visit GUID) -- confirmed globally unique across all 3 instruments.",
    }


def write_audit_outputs(ud: UnitedDataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    audit = compute_audit(ud)
    with open(os.path.join(out_dir, "data_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    ud.patient_table().to_csv(os.path.join(out_dir, "patient_summary.csv"), index=False)
