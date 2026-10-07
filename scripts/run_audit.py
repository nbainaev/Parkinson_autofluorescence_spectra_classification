"""Stage 1 entry point: load one instrument's data, validate schema, write
outputs/<instrument_id>/audit/{data_audit.json,patient_summary.csv}.

Usage: python scripts/run_audit.py configs/instrument_old.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectra_ml.data import build_dataset, write_audit_outputs
from spectra_ml.schema import InstrumentConfig, SchemaValidationError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instrument_config")
    args = parser.parse_args()

    instrument = InstrumentConfig.from_yaml(args.instrument_config)
    try:
        dataset = build_dataset(instrument)
    except SchemaValidationError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(ROOT, "outputs", instrument.id, "audit")
    write_audit_outputs(dataset, out_dir)

    fatal_issues = [i for i in dataset.schema_issues if i.severity == "FATAL"]
    print(f"[{instrument.id}] patients={dataset.patient_table()['patient_id'].nunique()} "
          f"spectra={dataset.n_spectra} fatal_issues={len(fatal_issues)}")
    print(f"  -> {out_dir}/data_audit.json")
    print(f"  -> {out_dir}/patient_summary.csv")
    if fatal_issues:
        print("  FATAL schema issues found (see data_audit.json['schema_issues']):", file=sys.stderr)
        for i in fatal_issues:
            print(f"    - {i.code}: {i.message}", file=sys.stderr)


if __name__ == "__main__":
    main()
