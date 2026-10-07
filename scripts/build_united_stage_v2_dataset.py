"""Task 1 (v2): build the pooled, harmonized, 4-class (Hoehn-Yahr PD stage)
patient dataset on top of the CORRECTED patient-identity fix
(spectra_ml.data.build_dataset's canonical patient_id) and the NEW
population-mean (diagnosis-blind) transfer curves from
outputs/harmonization_v2/ (scripts/build_transfer_v2.py) -- instead of the
stale pre-identity-fix outputs/united_multiclass/ (which used the OLD
diagnosis-independent-but-pre-identity-fix curves from
outputs/final_study_unified/transfer/).

Also exercises the fixed canonical-id stage join (spectra_ml.united_stage.
build_stage_series_canonical) -- see that module's docstring for why the raw-
GUID-keyed join broke after the identity fix.

Writes outputs/united_multiclass_v2/audit/{data_audit.json,patient_summary.csv}.

Usage: python scripts/build_united_stage_v2_dataset.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectra_ml.united import build_united_dataset
from spectra_ml.united_stage import build_united_stage_dataset, compute_stage_audit, write_stage_audit_outputs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSFER_DIR_V2 = os.path.join(ROOT, "outputs", "harmonization_v2")
OUT = os.path.join(ROOT, "outputs", "united_multiclass_v2", "audit")


def load_v2_stage_dataset():
    ud = build_united_dataset(transfer_dir=TRANSFER_DIR_V2)
    return build_united_stage_dataset(ud)


def main():
    usd = load_v2_stage_dataset()
    audit = compute_stage_audit(usd)

    print(f"[united_multiclass_v2] pooled patients total: {audit['n_patients_total_pooled']}")
    print(f"  included (4-class): {audit['n_patients_included_4class']}  {audit['included_class_counts']}")
    print(f"  excluded: {audit['n_patients_excluded']}  {audit['excluded_counts']}")
    for role, row in audit["by_instrument"].items():
        print(f"    {role}: {row}")

    write_stage_audit_outputs(usd, OUT)
    print(f"\n  -> {OUT}/data_audit.json")
    print(f"  -> {OUT}/patient_summary.csv")


if __name__ == "__main__":
    main()
