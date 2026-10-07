"""Task 1: build the pooled, harmonized, 4-class (Hoehn-Yahr PD stage)
patient dataset and write its audit.

Reuses spectra_ml.united.build_united_dataset() UNCHANGED (same harmonization/
QC as outputs/united/), then additively joins the clinical PD-stage column
(spectra_ml.united_stage) by patient_id. Writes
outputs/united_multiclass/audit/{data_audit.json,patient_summary.csv}.

Usage: python scripts/build_united_stage_dataset.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectra_ml.united import build_united_dataset
from spectra_ml.united_stage import build_united_stage_dataset, compute_stage_audit, write_stage_audit_outputs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united_multiclass", "audit")


def main():
    ud = build_united_dataset()
    usd = build_united_stage_dataset(ud)
    audit = compute_stage_audit(usd)

    print(f"[united_multiclass] pooled patients total: {audit['n_patients_total_pooled']}")
    print(f"  included (4-class): {audit['n_patients_included_4class']}  {audit['included_class_counts']}")
    print(f"  excluded: {audit['n_patients_excluded']}  {audit['excluded_counts']}")
    print(f"  verification: {audit['verification_note']}")
    for role, row in audit["by_instrument"].items():
        print(f"    {role}: {row}")

    write_stage_audit_outputs(usd, OUT)
    print(f"\n  -> {OUT}/data_audit.json")
    print(f"  -> {OUT}/patient_summary.csv")


if __name__ == "__main__":
    main()
