"""Split the single pooled cohort after global diagnosis-independent harmonization."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from spectra_ml.data import build_dataset, load_raw_table
from spectra_ml.harmonization import HarmonizationConfig
from spectra_ml.integrity import write_manifest
from spectra_ml.schema import InstrumentConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "outputs", "final_study_unified")
PRIVATE = os.path.join(STUDY, "private")
TEST_FRACTION, SEED = .20, 42


def main():
    os.makedirs(PRIVATE, exist_ok=True)
    transfer_meta = os.path.join(STUDY, "transfer", "transfer_functions_meta.json")
    if not os.path.exists(transfer_meta): raise FileNotFoundError("Run build_unified_transfer.py first")
    cfg = HarmonizationConfig.from_yaml(os.path.join(ROOT, "configs", "harmonization.yaml"))
    inst = {r: InstrumentConfig.from_yaml(os.path.join(ROOT, x.config)) for r, x in cfg.instruments.items()}
    raw = load_raw_table(inst["old"].data)
    parts = []
    for role in ("old", "new_pd", "new_control"):
        p = build_dataset(inst[role], df_full=raw).patient_table()
        p = p[p.target.notna()].copy(); p["target"] = p.target.astype(int); p["instrument_role"] = role
        parts.append(p[["patient_id", "target", "n_spectra", "instrument_role"]])
    allp = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(SEED)
    test_idx = []
    # Stratify on instrument x diagnosis. Singleton strata go to test so each
    # source/diagnosis can be represented in the pooled primary endpoint.
    for _, g in allp.groupby(["instrument_role", "target"], sort=True):
        idx = g.index.to_numpy(); rng.shuffle(idx)
        n = 1 if len(idx) == 1 else max(1, int(round(TEST_FRACTION * len(idx))))
        if len(idx) > 1: n = min(n, len(idx) - 1)
        test_idx.extend(idx[:n])
    is_test = allp.index.isin(test_idx)
    train, test = allp.loc[~is_test].copy(), allp.loc[is_test].copy()
    assert set(train.patient_id).isdisjoint(test.patient_id)
    train.to_csv(os.path.join(STUDY, "development_manifest.csv"), index=False)
    ids = [f"U-{rng.bytes(16).hex()}" for _ in range(len(test))]
    key = test.assign(study_id=ids)[["study_id", "patient_id", "target", "instrument_role"]].rename(
        columns={"patient_id": "real_patient_id"})
    blind = key[["study_id", "instrument_role"]].copy()
    identity = key[["study_id", "real_patient_id", "instrument_role"]].copy()
    blind.to_csv(os.path.join(STUDY, "blinded_test_manifest.csv"), index=False)
    identity.to_csv(os.path.join(PRIVATE, "prediction_identity_lookup.csv"), index=False)
    key.to_csv(os.path.join(PRIVATE, "blinding_key.csv"), index=False)
    counts = lambda d: d.groupby(["instrument_role", "target"]).size().rename("n").reset_index().to_dict("records")
    files = [os.path.join(STUDY, "development_manifest.csv"), os.path.join(STUDY, "blinded_test_manifest.csv"),
             os.path.join(PRIVATE, "prediction_identity_lookup.csv"), os.path.join(PRIVATE, "blinding_key.csv")]
    write_manifest(files, os.path.join(STUDY, "split_manifest.json"), base_dir=STUDY,
        extra={"random_seed": SEED, "test_fraction": TEST_FRACTION, "split_after_global_harmonization": True,
               "stratification": "instrument_role x target", "train_n": len(train), "test_n": len(test),
               "train_counts": counts(train), "test_counts": counts(test)})
    print("train", len(train), counts(train)); print("test", len(test), counts(test))


if __name__ == "__main__": main()
