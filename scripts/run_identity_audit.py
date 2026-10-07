"""Identity-fix entry point: run spectra_ml.identity's same-session GUID-merge
resolution against the full patients_database.xlsx table (all instruments)
and write outputs/identity_fix/merge_audit.json.

Usage: python scripts/run_identity_audit.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectra_ml.data import load_raw_table
from spectra_ml.identity import write_merge_audit
from spectra_ml.schema import DataConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    config = DataConfig.from_yaml(os.path.join(ROOT, "configs", "data.yaml"))
    raw = load_raw_table(config)
    out_dir = os.path.join(ROOT, "outputs", "identity_fix")
    audit = write_merge_audit(raw, config, out_dir)

    print(f"rows examined: {audit['n_rows_examined']}")
    print(f"merge groups (same-session two-hand pairs collapsed): {audit['n_merge_groups']}")
    print(f"GUIDs collapsed: {audit['n_guids_collapsed']}")
    print(f"rejected collision groups (kept separate, real conflict): {audit['n_rejected_collision_groups']}")
    for instrument, stats in audit["before_after_patient_counts_by_instrument"].items():
        print(f"  [{instrument[:60]}...] before={stats['n_patients_before_raw_guid']} "
              f"after={stats['n_patients_after_canonical_id']} "
              f"merges={stats['n_merge_groups']} rejected={stats['n_rejected_collision_groups']}")
    print(f"-> {out_dir}/merge_audit.json")


if __name__ == "__main__":
    main()
