"""Build ONE diagnosis-blind (population-mean) empirical transfer curve per new
instrument (new_pd->old, new_control->old), on the CORRECTED (post-identity-
fix) patient cohort.

Replaces the earlier diagnosis-conditioned PD-derived-primary / control-
derived-diagnostic curve pair (scripts/build_transfer_functions.py,
outputs/final_study/transfer/) and the pre-identity-fix diagnosis-independent
curves (scripts/build_unified_transfer.py, outputs/final_study_unified/
transfer/) with a single, simpler, per-new-instrument curve estimated from the
AREA-NORMALIZED MEAN SPECTRUM OVER *ALL* PATIENTS of that instrument (not
split by PD/control class) -- i.e. spectra_ml.harmonization.class_group_mean's
target_value=None population, exactly as requested for this task.

Pipeline (reuses spectra_ml.harmonization directly, no reimplementation):
  1. build_dataset() (now fixed: canonical, real-person-level patient_id) for
     old / new_pd / new_control.
  2. QC (build_qc_table + hot-pixel correction), per instrument (same QC
     convention as scripts/build_unified_transfer.py).
  3. build_patient_area_normalized_repr(...) -> one area-normalized
     representative (median) spectrum per patient.
  4. class_group_mean(repr_, target_value=None, n_boot=..., seed=...) -> the
     diagnosis-blind population mean curve (+ patient-bootstrap CI) for each
     instrument.
  5. build_transfer_curve_raw(old_mean, new_mean, ...) -> smooth_transfer_curve
     -> K(wavelength) per new instrument.
  6. bootstrap_transfer_curve(...) for the curve's own patient-bootstrap CI.
  7. compute_similarity_metrics before/after apply_transfer, as a cheap sanity
     check (NOT a full Phase-1-style diagnostic report).

Wavelength-shift diagnostics (estimate_wavelength_shift/evaluate_shift_gate)
are DELIBERATELY NOT re-derived here: the wavelength-shift PASS finding from
the prior (pre-identity-fix) harmonization work is a property of the
instruments' physical wavelength grids, independent of exact patient identity/
count, and is inherited unchanged -- noted in the output metadata, not
recomputed.

Usage: python scripts/build_transfer_v2.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from spectra_ml.data import build_dataset, load_raw_table
from spectra_ml.harmonization import (
    apply_transfer,
    bootstrap_transfer_curve,
    build_patient_area_normalized_repr,
    build_transfer_curve_raw,
    class_group_mean,
    compute_similarity_metrics,
    smooth_transfer_curve,
)
from spectra_ml.qc import HARD_FAIL, SOFT_WARNING, QCConfig, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig
from spectra_ml.united import ROLE_CONFIG, ROLES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "harmonization_v2")
RANDOM_SEED = 42
N_BOOT = 500  # runtime discipline: modest bootstrap count, matching the 500-750 precedent elsewhere


def build_repr_for_role(ds, qc_cfg: QCConfig):
    spectrum_id = np.array([f"{ds.instrument.id}:{i}" for i in range(ds.n_spectra)])
    qc_df, hp, _ = build_qc_table(ds.spectra, ds.waves, ds.patient_id, spectrum_id, qc_cfg)
    corrected, _ = correct_hot_pixels(ds.spectra, hp, qc_cfg)
    return build_patient_area_normalized_repr(
        corrected, ds.patient_id, ds.target, ds.waves, qc_status=qc_df["qc_status"].to_numpy(),
        exclude_statuses=(HARD_FAIL, SOFT_WARNING), method="median",
    ), qc_df


def main():
    os.makedirs(OUT, exist_ok=True)
    qc_cfg = QCConfig.from_yaml(os.path.join(ROOT, "configs", "qc.yaml"))
    inst = {role: InstrumentConfig.from_yaml(path) for role, path in ROLE_CONFIG.items()}
    raw_df = load_raw_table(inst["old"].data)
    ds = {role: build_dataset(cfg, df_full=raw_df) for role, cfg in inst.items()}
    waves = ds["old"].waves

    repr_by_role, qc_by_role = {}, {}
    for role in ROLES:
        repr_by_role[role], qc_by_role[role] = build_repr_for_role(ds[role], qc_cfg)

    pop_mean = {}
    for role in ROLES:
        pop_mean[role] = class_group_mean(repr_by_role[role], target_value=None, n_boot=N_BOOT, seed=RANDOM_SEED)
        print(f"[harmonization_v2] {role}: population-mean curve from {pop_mean[role]['n_patients']} patients "
              "(diagnosis-blind, target_value=None)")

    np.save(os.path.join(OUT, "wavelength_grid.npy"), waves)
    np.save(os.path.join(OUT, "old_population_mean.npy"), pop_mean["old"]["mean"])

    meta = {
        "method": "diagnosis-blind (target_value=None) area-normalized population-mean instrument harmonization",
        "reference_population": "ALL patients of each instrument, NOT split by PD/control class (per task spec)",
        "old_reference_n_patients": pop_mean["old"]["n_patients"],
        "wavelength_shift_diagnostic": (
            "NOT re-derived in this script -- inherited from the prior (pre-identity-fix) Phase 1/2 finding "
            "that new1/new2 wavelength grids PASS the alignment gate against OLD's grid (a property of the "
            "instruments' physical wavelength axes, not of exact patient identity/count); see "
            "outputs/final_study_unified/transfer/transfer_functions_meta.json for that original gate result."
        ),
        "curves": {},
    }

    rows = []
    for role in ("new_pd", "new_control"):
        old_mean = pop_mean["old"]["mean"]
        new_mean = pop_mean[role]["mean"]

        raw = build_transfer_curve_raw(old_mean, new_mean, denom_near_zero_rel_threshold=0.02,
                                        edge_exclude_n_points=3)
        k = smooth_transfer_curve(raw["k_raw"], window=15, polyorder=3)
        ci = bootstrap_transfer_curve(
            repr_by_role["old"].spectra, repr_by_role[role].spectra,
            denom_near_zero_rel_threshold=0.02, edge_exclude_n_points=3,
            smoothing_window=15, smoothing_polyorder=3, n_boot=N_BOOT, seed=RANDOM_SEED,
        )

        npy = os.path.join(OUT, f"k_{role}_to_old.npy")
        np.save(npy, k)

        harmonized = apply_transfer(repr_by_role[role].spectra, waves, k)["harmonized"]
        before = compute_similarity_metrics(old_mean, new_mean, waves)
        after = compute_similarity_metrics(old_mean, harmonized.mean(axis=0), waves)

        meta["curves"][role] = {
            "file": os.path.basename(npy),
            "n_old_patients": pop_mean["old"]["n_patients"],
            "n_new_patients": pop_mean[role]["n_patients"],
            "diagnosis_blind": True,
            "similarity_before": before,
            "similarity_after": after,
        }
        print(f"  {role}->old: pearson {before['pearson']:.4f} -> {after['pearson']:.4f}   "
              f"nrmse {before['nrmse']:.4f} -> {after['nrmse']:.4f}")

        for i, w in enumerate(waves):
            rows.append({"instrument_role": role, "wavelength": float(w), "k_raw": float(raw["k_raw"][i]),
                         "k_smoothed": float(k[i]), "bootstrap_median": float(ci.bootstrap_median.iloc[i]),
                         "ci95_lo": float(ci.ci_lo.iloc[i]), "ci95_hi": float(ci.ci_hi.iloc[i])})

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_curves.csv"), index=False)
    with open(os.path.join(OUT, "transfer_functions_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  -> {OUT}/k_new_pd_to_old.npy, k_new_control_to_old.npy, wavelength_grid.npy")
    print(f"  -> {OUT}/transfer_curves.csv")
    print(f"  -> {OUT}/transfer_functions_meta.json")


if __name__ == "__main__":
    main()
