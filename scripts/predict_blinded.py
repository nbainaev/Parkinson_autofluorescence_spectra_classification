"""Generate frozen-model predictions without using blind targets."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd

from spectra_ml.blinding import load_blinded_test_manifest
from spectra_ml.data import load_raw_table
from spectra_ml.final_study import (load_instruments, load_transfer_artifacts, prepare_role_dataset,
                                    raw_model_probability)
from spectra_ml.integrity import sha256_file, verify_manifest, write_manifest
from spectra_ml.qc import QCConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.environ.get("SPECTRA_STUDY_DIR", os.path.join(ROOT, "outputs", "final_study"))


def main():
    frozen = os.path.join(STUDY, "frozen")
    verify_manifest(os.path.join(frozen, "frozen_manifest.json"), base_dir=STUDY)
    blind = load_blinded_test_manifest(os.path.join(STUDY, "blinded_test_manifest.csv"))
    # Operational identity lookup is physically separate from the label key.
    identity_path = os.path.join(STUDY, "private", "prediction_identity_lookup.csv")
    identity = pd.read_csv(identity_path, dtype={"study_id": str, "real_patient_id": str})
    if "target" in identity.columns:
        raise RuntimeError("Prediction identity lookup contains target; refusing to predict.")
    identity = identity[["study_id", "real_patient_id"]]
    mapping = blind.merge(identity, on="study_id", validate="one_to_one")
    assert "target" not in mapping.columns

    _, instruments = load_instruments(ROOT, os.path.join(ROOT, "configs", "harmonization.yaml"))
    # Prediction consumes the exact transfer arrays included in the frozen
    # manifest, never mutable working copies from the transfer-build directory.
    waves, curves, _ = load_transfer_artifacts(frozen)
    qc = QCConfig.from_yaml(os.path.join(ROOT, "configs", "qc.yaml"))
    df_full = load_raw_table(instruments["old"].data)
    branches = [
        ("harmonized", frozen, True, ["linear_svm", "pls_logreg", "logreg"]),
        ("no_transfer", os.path.join(STUDY, "frozen_no_transfer"), False, ["linear_svm"]),
        ("old_only", os.path.join(STUDY, "frozen_old_only"), True, ["linear_svm"]),
    ]
    rows = []
    for role, group in mapping.groupby("instrument_role", sort=False):
        ids = set(group.real_patient_id.astype(str))
        for branch, model_dir, use_transfer, families in branches:
            x, _, pid, qc_df = prepare_role_dataset(role, ids, instruments, df_full, waves, curves, qc,
                                                     use_transfer=use_transfer)
            sid_for_pid = dict(zip(group.real_patient_id.astype(str), group.study_id))
            for family in families:
                bundle = joblib.load(os.path.join(model_dir, f"final_model_{family}.joblib"))
                probability = raw_model_probability(bundle, x)
                tmp = pd.DataFrame({"real_patient_id": pid, "probability": probability})
                agg = tmp.groupby("real_patient_id", sort=False).probability.agg(["mean", "size"]).reset_index()
                for rec in agg.itertuples(index=False):
                    rows.append({"study_id": sid_for_pid[rec.real_patient_id], "instrument": role,
                                 "model": family, "analysis_branch": branch, "n_spectra": int(rec.size),
                                 "predicted_probability_PD": float(rec.mean),
                                 "predicted_class": int(rec.mean >= bundle["threshold"]),
                                 "frozen_threshold": float(bundle["threshold"])})
    predictions = pd.DataFrame(rows).sort_values(["analysis_branch", "model", "study_id"])
    out = os.path.join(STUDY, "blinded_predictions.csv")
    predictions.to_csv(out, index=False)
    write_manifest([out], os.path.join(STUDY, "blinded_predictions_manifest.json"), base_dir=STUDY,
                   extra={"state": "PREDICTIONS_FROZEN_BEFORE_UNBLINDING",
                          "frozen_manifest_sha256": sha256_file(os.path.join(frozen, "frozen_manifest.json"))})
    print(f"Wrote {len(predictions)} blind prediction rows and SHA256 manifest: {out}")


if __name__ == "__main__": main()
