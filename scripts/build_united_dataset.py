"""United-dataset entry point: build the pooled, harmonized, all-3-instrument
dataset (old + new1/new_pd + new2/new_control) and write its audit + QC
artifacts under outputs/united/.

Scope: a single non-blinded, all-data pooled analysis -- NOT the Phase-2
blinded validation protocol (does not touch outputs/final_study_unified/,
does not rebuild transfer curves, does not create a blind split).

Usage: python scripts/build_united_dataset.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.qc import HARD_FAIL, SOFT_WARNING
from spectra_ml.united import ROLES, build_united_dataset, compute_audit, write_audit_outputs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united")


def main():
    ud = build_united_dataset()

    audit_dir = os.path.join(OUT, "audit")
    qc_dir = os.path.join(OUT, "qc")
    os.makedirs(qc_dir, exist_ok=True)
    write_audit_outputs(ud, audit_dir)

    ud.qc_df.to_csv(os.path.join(qc_dir, "qc_report.csv"), index=False)
    with open(os.path.join(qc_dir, "hot_pixel_correction_log.json"), "w", encoding="utf-8") as f:
        json.dump(ud.hot_pixel_correction_log, f, indent=2)

    audit = compute_audit(ud)
    print(f"United dataset: {audit['n_patients_total']} patients, {audit['n_spectra_total']} spectra "
          f"(park={audit['n_parkinson_patients']}, control={audit['n_control_patients']}, "
          f"unknown={audit['n_patients_unknown_target']})")
    for role in ROLES:
        bi = audit["by_instrument"][role]
        print(f"  [{role:>12s}] patients={bi['n_patients']:>4d} spectra={bi['n_spectra']:>4d} "
              f"park={bi['n_parkinson_patients']:>3d} control={bi['n_control_patients']:>3d} "
              f"unknown={bi['n_patients_unknown_target']:>3d} qc={bi['qc_status_counts']} "
              f"interpolated={bi['n_wavelengths_interpolated_spectra']}")
    print(f"  pooled QC status counts: {audit['qc_status_counts_pooled']}")

    keep_labeled = (ud.qc_status != HARD_FAIL) & ~np.isnan(ud.target)
    print(f"\n  labeled modeling-eligible spectra (QC != HARD_FAIL, target known): {int(keep_labeled.sum())}"
          f" / {ud.n_spectra}")

    print(f"\n  -> {audit_dir}/data_audit.json")
    print(f"  -> {audit_dir}/patient_summary.csv")
    print(f"  -> {qc_dir}/qc_report.csv")


if __name__ == "__main__":
    main()
