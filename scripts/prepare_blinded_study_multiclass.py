"""Part II/III of current_tasks/double_blind_exp.md, adapted to the small,
4-class Hoehn-Yahr STAGE multiclass problem (outputs/united_multiclass_v2).

Scope decision (documented, not hidden): this is a much smaller blinding
exercise than the full 39-part binary protocol -- only the core mechanics
(patient-level split, private key file, target-free blind manifest, frozen
predict-then-unblind ordering, SHA256 checks) are reproduced, on top of the
ALREADY-FROZEN, diagnosis-blind harmonization curves in
outputs/harmonization_v2/ (built from the full patient population of each
instrument, regardless of diagnosis -- reused as-is here, NOT rebuilt
development-only; this is a stated, defensible scope simplification because
those curves never touch labels).

Split design (documented judgement calls):
  * Stratify by the 4-class stage label ONLY (not jointly by instrument x
    class): new_pd contributes only 7 stage-labeled patients (1 control + 6
    stage3) and new_control contributes only controls (20) -- joint
    instrument x class strata would leave singleton/near-singleton cells
    (e.g. new_pd x control has n=1), so pure class-stratification is judged
    more important for this task than instrument-stratification. Instrument
    composition of dev vs blind is still reported for transparency.
  * test_size=0.20 (not the original binary study's ~0.30): the smallest
    class (stage2, n=14) would leave very few patients per side at 30%;
    80/20 keeps every class's dev fold usable for nested CV while still
    giving a handful of blind-test patients per class (stage2 -> 3 of 14).
  * Fixed seed 42, consistent with the rest of the project.

Reuses `spectra_ml.splitting.patient_level_holdout_split` UNCHANGED -- that
function is already class-count-agnostic (loops over
`sorted(set(patient_target))`), so no new splitting code was needed for the
4-class case.

Writes (mirrors scripts/prepare_blinded_study.py's file layout):
  outputs/blinded_multiclass/development_manifest.csv       (patient_id, target, instrument, n_spectra)
  outputs/blinded_multiclass/blinded_test_manifest.csv       (study_id, instrument, n_spectra -- NO target, NO real id)
  outputs/blinded_multiclass/private/blinding_key.csv        (study_id, real_patient_id, target, instrument, n_spectra)
  outputs/blinded_multiclass/private/prediction_identity_lookup.csv (study_id, real_patient_id, instrument -- NO target;
                                                                       the ONLY file predict_blinded_multiclass.py may read)
  outputs/blinded_multiclass/split_manifest.json             (seed, counts, SHA256 of every file above)

Usage: python scripts/prepare_blinded_study_multiclass.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from run_experiment_united_stage_v2 import CLASS_LABELS, load_modeling_set  # noqa: E402
from spectra_ml.integrity import write_manifest
from spectra_ml.splitting import assert_all_classes_present, assert_no_patient_leakage, patient_level_holdout_split
from spectra_ml.united_stage import CLASS_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "outputs", "blinded_multiclass")
PRIVATE = os.path.join(STUDY, "private")
TEST_SIZE = 0.30
SEED = 42


def _patient_table(patient_id, target, instrument):
    df = pd.DataFrame({"patient_id": patient_id, "target": target, "instrument": instrument})
    agg = df.groupby("patient_id", sort=False).agg(
        target=("target", "first"), instrument=("instrument", "first"), n_spectra=("target", "size"),
    ).reset_index()
    agg["stage_name"] = agg["target"].map(CLASS_NAMES)
    return agg


def main():
    os.makedirs(PRIVATE, exist_ok=True)
    usd, spectra, target, patient_id, instrument = load_modeling_set()

    dev_idx, blind_idx = patient_level_holdout_split(patient_id, target, TEST_SIZE, SEED,
                                                       singleton_class_destination="dev")
    assert_no_patient_leakage(patient_id, dev_idx, blind_idx, "development", "blind_test")
    assert_all_classes_present(target, CLASS_LABELS, dev_idx, "development")
    assert_all_classes_present(target, CLASS_LABELS, blind_idx, "blind_test")

    dev_patients = _patient_table(patient_id[dev_idx], target[dev_idx], instrument[dev_idx])
    blind_patients = _patient_table(patient_id[blind_idx], target[blind_idx], instrument[blind_idx])
    assert set(dev_patients.patient_id).isdisjoint(set(blind_patients.patient_id))

    dev_patients.to_csv(os.path.join(STUDY, "development_manifest.csv"), index=False)

    rng = np.random.default_rng(SEED)
    study_ids = [f"MC-{rng.bytes(16).hex()}" for _ in range(len(blind_patients))]
    blind_patients = blind_patients.assign(study_id=study_ids)

    blinding_key = blind_patients.rename(columns={"patient_id": "real_patient_id"})[
        ["study_id", "real_patient_id", "target", "stage_name", "instrument", "n_spectra"]
    ]
    identity_lookup = blinding_key[["study_id", "real_patient_id", "instrument", "n_spectra"]].copy()
    blinded_test_manifest = blinding_key[["study_id", "instrument", "n_spectra"]].copy()

    blinding_key.to_csv(os.path.join(PRIVATE, "blinding_key.csv"), index=False)
    identity_lookup.to_csv(os.path.join(PRIVATE, "prediction_identity_lookup.csv"), index=False)
    blinded_test_manifest.to_csv(os.path.join(STUDY, "blinded_test_manifest.csv"), index=False)

    def counts(df):
        return df.groupby(["instrument", "stage_name"]).size().rename("n").reset_index().to_dict("records")

    files = [
        os.path.join(STUDY, "development_manifest.csv"),
        os.path.join(STUDY, "blinded_test_manifest.csv"),
        os.path.join(PRIVATE, "blinding_key.csv"),
        os.path.join(PRIVATE, "prediction_identity_lookup.csv"),
    ]
    write_manifest(files, os.path.join(STUDY, "split_manifest.json"), base_dir=STUDY, extra={
        "random_seed": SEED,
        "test_size": TEST_SIZE,
        "stratification": "4-class stage label only (see script docstring for why instrument x class joint "
                            "stratification was rejected: near-singleton cells for new_pd/new_control)",
        "harmonization_curves": "outputs/harmonization_v2/ (FROZEN, diagnosis-blind, reused as-is -- not "
                                 "rebuilt development-only)",
        "n_development_patients": len(dev_patients), "n_blind_patients": len(blind_patients),
        "development_counts_by_instrument_class": counts(dev_patients),
        "blind_counts_by_instrument_class": counts(blind_patients),
        "development_class_totals": dev_patients.groupby("stage_name").size().to_dict(),
        "blind_class_totals": blind_patients.groupby("stage_name").size().to_dict(),
    })

    print(f"development: {len(dev_patients)} patients {dev_patients.groupby('stage_name').size().to_dict()}")
    print(f"blind_test:  {len(blind_patients)} patients {blind_patients.groupby('stage_name').size().to_dict()}")
    print(f"  -> {STUDY}/development_manifest.csv")
    print(f"  -> {STUDY}/blinded_test_manifest.csv")
    print(f"  -> {PRIVATE}/blinding_key.csv (PRIVATE -- never read by training/prediction code)")
    print(f"  -> {PRIVATE}/prediction_identity_lookup.csv (target-free, used ONLY by predict_blinded_multiclass.py)")
    print(f"  -> {STUDY}/split_manifest.json")


if __name__ == "__main__":
    main()
