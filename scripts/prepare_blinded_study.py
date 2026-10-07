"""Phase 2 step 1 (current_tasks/double_blind_exp.md Parts II, III): patient-level
development / blind-test split, executed BEFORE any transfer-curve estimation
or model training touches these patients.

Judgment calls made here (documented per the spec's explicit delegation of
"exact fraction is configurable" to the implementer):

  * OLD locked hold-out (secondary validation): 20% patient-level stratified
    holdout (Part II: "например около 20%"). OLD has 152 PD / 74 control
    patients, so both classes are easily split at 20% (~30 PD / ~15 control
    held out) -- no small-sample edge case here.

  * NEW_PD_DEPARTMENT primary blind test: the spec explicitly says "if control
    patients are few, use ~20% rather than shrinking development further"
    (Part II). new_pd has only 29 PD / 1 control (labeled) patients, so we use
    the LOW end of the suggested 20-30% range (test_size=0.20) applied to the
    PD class: round(29*0.20) = 6 PD patients -> blind test, 23 PD -> dev.

    The single control patient is a genuine edge case the spec does not fully
    resolve: a fractional split of n=1 cannot leave a nonzero count on both
    sides. We assign it to the BLIND TEST side (not development), specifically
    so the primary endpoint (Part XXV: ROC-AUC on blinded new_pd patients) is
    at least *computable* -- ROC-AUC is undefined with only one class present.
    The cost of this choice: the dev-only transfer-curve rebuild (step 2) has
    ZERO new_pd control patients available, so Part XI's PD-derived-vs-
    Control-derived new_pd diagnostic curve comparison cannot be built for the
    real study (documented as a known gap, not silently dropped). This
    trade-off (a computable, if underpowered, primary endpoint vs. a Part XI
    diagnostic) is judged preferable given Part XXV is the PRIMARY endpoint
    and Part XI is an internal diagnostic already covered by Phase 1's
    full-cohort exploratory report.

  * NEW_CONTROL_DEPARTMENT: per Part XVIII's explicit allowance ("NEW_CONTROL
    можно добавить в training controls, если это заранее определено в
    protocol"), ALL labeled new_control patients (2 PD / 32 control) go to
    DEVELOPMENT -- no blind-test carve-out. A blind-test split of 2 PD
    patients would be statistically meaningless and is not attempted. This is
    stated here, in advance, as the predetermined protocol choice.

  * Patients with unknown/missing target (new_pd: 4, new_control: 2) are
    EXCLUDED entirely from both development and blind test (cannot be scored
    against a label they don't have, and cannot be used unsupervised anywhere
    in a supervised pipeline per the existing project convention).

Random seed: reuses configs/harmonization.yaml's `random_seed: 42` (same value
as configs/data.yaml's, kept consistent project-wide).

Usage: python scripts/prepare_blinded_study.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from spectra_ml.data import build_dataset, load_raw_table
from spectra_ml.harmonization import HarmonizationConfig
from spectra_ml.integrity import write_manifest
from spectra_ml.schema import InstrumentConfig
from spectra_ml.splitting import assert_no_patient_leakage, patient_level_holdout_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "final_study")
PRIVATE_DIR = os.path.join(OUT_DIR, "private")

OLD_HOLDOUT_FRACTION = 0.20
NEW_PD_BLIND_FRACTION = 0.20
NEW_PD_SINGLETON_CONTROL_DESTINATION = "test"  # see module docstring


def _patient_table_labeled(ds):
    """ds.patient_table() restricted to patients with a known target (0/1)."""
    pat = ds.patient_table()
    labeled = pat[pat["target"].notna()].copy()
    labeled["target"] = labeled["target"].astype(int)
    return labeled, pat[pat["target"].isna()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmonization-config", default=os.path.join(ROOT, "configs", "harmonization.yaml"))
    args = parser.parse_args()

    os.makedirs(PRIVATE_DIR, exist_ok=True)

    cfg = HarmonizationConfig.from_yaml(args.harmonization_config)
    seed = cfg.random_seed  # = 42, same as configs/data.yaml's random_seed -- documented single source of truth

    old_inst = InstrumentConfig.from_yaml(os.path.join(ROOT, cfg.instruments["old"].config))
    new_pd_inst = InstrumentConfig.from_yaml(os.path.join(ROOT, cfg.instruments["new_pd"].config))
    new_control_inst = InstrumentConfig.from_yaml(os.path.join(ROOT, cfg.instruments["new_control"].config))

    df_full = load_raw_table(old_inst.data)
    ds_old = build_dataset(old_inst, df_full=df_full)
    ds_new_pd = build_dataset(new_pd_inst, df_full=df_full)
    ds_new_control = build_dataset(new_control_inst, df_full=df_full)

    pat_old, unk_old = _patient_table_labeled(ds_old)
    pat_new_pd, unk_new_pd = _patient_table_labeled(ds_new_pd)
    pat_new_control, unk_new_control = _patient_table_labeled(ds_new_control)

    print(f"[data] old: {len(pat_old)} labeled patients ({int((pat_old.target==1).sum())} PD / "
          f"{int((pat_old.target==0).sum())} control), {len(unk_old)} unknown-target excluded")
    print(f"[data] new_pd: {len(pat_new_pd)} labeled patients ({int((pat_new_pd.target==1).sum())} PD / "
          f"{int((pat_new_pd.target==0).sum())} control), {len(unk_new_pd)} unknown-target excluded")
    print(f"[data] new_control: {len(pat_new_control)} labeled patients ({int((pat_new_control.target==1).sum())} "
          f"PD / {int((pat_new_control.target==0).sum())} control), {len(unk_new_control)} unknown-target excluded")

    # -- OLD: secondary locked hold-out (Part II) -----------------------------
    old_dev_idx, old_test_idx = patient_level_holdout_split(
        pat_old["patient_id"].to_numpy(), pat_old["target"].to_numpy().astype(float),
        test_size=OLD_HOLDOUT_FRACTION, seed=seed, singleton_class_destination="dev",
    )
    old_dev_patients = pat_old.iloc[old_dev_idx].copy()
    old_holdout_patients = pat_old.iloc[old_test_idx].copy()

    # -- new_pd: primary blind test (Part II) ---------------------------------
    new_pd_dev_idx, new_pd_test_idx = patient_level_holdout_split(
        pat_new_pd["patient_id"].to_numpy(), pat_new_pd["target"].to_numpy().astype(float),
        test_size=NEW_PD_BLIND_FRACTION, seed=seed, singleton_class_destination=NEW_PD_SINGLETON_CONTROL_DESTINATION,
    )
    new_pd_dev_patients = pat_new_pd.iloc[new_pd_dev_idx].copy()
    new_pd_blind_patients = pat_new_pd.iloc[new_pd_test_idx].copy()

    # -- new_control: entirely DEVELOPMENT (Part XVIII predetermined choice) --
    new_control_dev_patients = pat_new_control.copy()

    # -- sanity: zero leakage between dev and blind-test patient sets --------
    dev_ids_all = pd.concat([
        old_dev_patients[["patient_id"]].assign(instrument_id="old"),
        new_pd_dev_patients[["patient_id"]].assign(instrument_id="new1"),
        new_control_dev_patients[["patient_id"]].assign(instrument_id="new2"),
    ])
    blind_ids_all = pd.concat([
        old_holdout_patients[["patient_id"]].assign(instrument_id="old"),
        new_pd_blind_patients[["patient_id"]].assign(instrument_id="new1"),
    ])
    combined_pid = np.concatenate([dev_ids_all["patient_id"].to_numpy(), blind_ids_all["patient_id"].to_numpy()])
    dev_row_idx = np.arange(len(dev_ids_all))
    blind_row_idx = np.arange(len(dev_ids_all), len(dev_ids_all) + len(blind_ids_all))
    assert_no_patient_leakage(combined_pid, dev_row_idx, blind_row_idx, "development", "blind_test")
    print("[check] zero patient leakage between development and blind-test sets: OK")

    # -- development_manifest.csv (real IDs + target -- used by training) ----
    def _dev_rows(patients_df, instrument_role, instrument_id):
        return pd.DataFrame({
            "patient_id": patients_df["patient_id"].to_numpy(),
            "instrument_role": instrument_role,
            "instrument_id": instrument_id,
            "target": patients_df["target"].to_numpy().astype(int),
            "n_spectra": patients_df["n_spectra"].to_numpy(),
        })

    development_manifest = pd.concat([
        _dev_rows(old_dev_patients, "old", "old"),
        _dev_rows(new_pd_dev_patients, "new_pd", "new1"),
        _dev_rows(new_control_dev_patients, "new_control", "new2"),
    ], ignore_index=True)
    development_manifest.to_csv(os.path.join(OUT_DIR, "development_manifest.csv"), index=False)
    print(f"  -> development_manifest.csv ({len(development_manifest)} patients)")

    # -- blinded_test_manifest.csv (study_id only, NO target, NO real id) ----
    rng = np.random.default_rng(seed + 1)  # separate stream from the split RNG

    def _make_study_ids(n):
        return [f"S-{rng.bytes(16).hex()}" for _ in range(n)]

    blind_rows = []
    key_rows = []
    for patients_df, instrument_role, instrument_id, cohort in [
        (old_holdout_patients, "old", "old", "secondary_old_holdout"),
        (new_pd_blind_patients, "new_pd", "new1", "primary_new_pd_blind_test"),
    ]:
        study_ids = _make_study_ids(len(patients_df))
        for sid, (_, row) in zip(study_ids, patients_df.iterrows()):
            blind_rows.append({"study_id": sid, "instrument_role": instrument_role, "instrument_id": instrument_id,
                                "cohort": cohort})
            key_rows.append({"study_id": sid, "real_patient_id": row["patient_id"], "target": int(row["target"]),
                              "instrument_role": instrument_role, "instrument_id": instrument_id, "cohort": cohort})

    blinded_test_manifest = pd.DataFrame(blind_rows)
    blinded_test_manifest.to_csv(os.path.join(OUT_DIR, "blinded_test_manifest.csv"), index=False)
    print(f"  -> blinded_test_manifest.csv ({len(blinded_test_manifest)} study_ids, NO target/real id)")

    blinding_key = pd.DataFrame(key_rows)
    blinding_key.to_csv(os.path.join(PRIVATE_DIR, "blinding_key.csv"), index=False)
    print(f"  -> private/blinding_key.csv ({len(blinding_key)} rows) -- NOT to be read by training/prediction code")

    # Prediction needs an operational mapping from randomized study IDs to the
    # raw patient records, but must not open the label-bearing key.  Keep a
    # separate target-free lookup for that purpose.
    identity_lookup = blinding_key[["study_id", "real_patient_id", "instrument_role", "instrument_id",
                                    "cohort"]].copy()
    identity_lookup.to_csv(os.path.join(PRIVATE_DIR, "prediction_identity_lookup.csv"), index=False)
    print(f"  -> private/prediction_identity_lookup.csv ({len(identity_lookup)} rows, NO target)")

    # -- split_manifest.json ---------------------------------------------------
    def _class_counts(df):
        return {"n_patients": int(len(df)), "n_pd": int((df["target"] == 1).sum()),
                "n_control": int((df["target"] == 0).sum())}

    split_manifest_body = {
        "random_seed": seed,
        "seed_source": "configs/harmonization.yaml random_seed (same value as configs/data.yaml random_seed=42)",
        "fractions": {
            "old_secondary_holdout_fraction": OLD_HOLDOUT_FRACTION,
            "new_pd_primary_blind_fraction": NEW_PD_BLIND_FRACTION,
            "new_pd_singleton_control_destination": NEW_PD_SINGLETON_CONTROL_DESTINATION,
            "new_control_protocol": "all labeled new_control patients -> DEVELOPMENT (Part XVIII), no blind carve-out",
        },
        "development": {
            "old": _class_counts(old_dev_patients),
            "new_pd": _class_counts(new_pd_dev_patients),
            "new_control": _class_counts(new_control_dev_patients),
            "total_patients": int(len(development_manifest)),
        },
        "blind_test": {
            "secondary_old_holdout": _class_counts(old_holdout_patients),
            "primary_new_pd_blind_test": _class_counts(new_pd_blind_patients),
            "total_patients": int(len(blinded_test_manifest)),
        },
        "excluded_unknown_target": {
            "old": int(len(unk_old)), "new_pd": int(len(unk_new_pd)), "new_control": int(len(unk_new_control)),
        },
        "notes": [
            "OLD locked hold-out = secondary validation (Part II): OLD was already used for prior exploratory "
            "model selection, so its held-out performance is corroborating evidence, not the primary endpoint.",
            "new_pd's single labeled control patient was assigned to the BLIND TEST side (not development) so "
            "that the primary endpoint (Part XXV, ROC-AUC on blinded new_pd patients) is computable at all; the "
            "consequence is that the dev-only transfer-curve rebuild (step 2) has zero new_pd control patients, "
            "so the Part XI PD-derived-vs-Control-derived diagnostic curve comparison is NOT available for the "
            "real study (Phase 1's full-cohort exploratory report already covers that diagnostic).",
            "new_control patients all go to DEVELOPMENT per Part XVIII's explicit allowance -- predetermined "
            "protocol choice, stated here before any modeling.",
        ],
    }

    manifest_files = [
        os.path.join(OUT_DIR, "development_manifest.csv"),
        os.path.join(OUT_DIR, "blinded_test_manifest.csv"),
        os.path.join(PRIVATE_DIR, "blinding_key.csv"),
        os.path.join(PRIVATE_DIR, "prediction_identity_lookup.csv"),
    ]
    write_manifest(manifest_files, os.path.join(OUT_DIR, "split_manifest.json"), base_dir=OUT_DIR,
                    extra=split_manifest_body)
    print(f"  -> split_manifest.json (incl. SHA256 of the 3 manifest files above)")

    print("\nDone. Development-side scripts (transfer rebuild, freeze_final_model.py) may read "
          "development_manifest.csv freely. blinded_test_manifest.csv carries NO target/real id. "
          "private/prediction_identity_lookup.csv is the target-free operational mapping used for prediction. "
          "private/blinding_key.csv must ONLY be read by unblind_and_evaluate.py after predictions are frozen.")


if __name__ == "__main__":
    main()
