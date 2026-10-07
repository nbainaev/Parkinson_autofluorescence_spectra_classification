"""Part XXIII of current_tasks/double_blind_exp.md: generate frozen-model
predictions for the blind-test patients WITHOUT ever reading target/real
patient identity. Only file this script may read from `private/` is
`prediction_identity_lookup.csv` (study_id, real_patient_id, instrument --
structurally cannot contain target, see spectra_ml.blinding.load_blinded_test_
manifest's analogous check, reproduced here for the identity lookup too).

Pipeline (mirrors Part XXIII exactly): rebuild the pooled, QC'd, harmonized
spectra for ALL patients (deterministic, target-blind at the feature-
construction level -- harmonization curves are the FROZEN copies inside
frozen/, not the mutable outputs/harmonization_v2/ working directory) ->
restrict rows to the blind-test patients' real_patient_ids -> per-family
frozen preprocessing + model -> per-spectrum probabilities -> mean-aggregate
to patient -> map real_patient_id back to study_id -> write predictions
keyed ONLY by study_id (no target, no real patient id).

Usage: python scripts/predict_blinded_multiclass.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd

from run_experiment_united_stage_v2 import CLASS_LABELS  # noqa: E402
from spectra_ml.blinding import load_blinded_test_manifest
from spectra_ml.evaluation import aggregate_to_patient_multiclass
from spectra_ml.integrity import sha256_file, verify_manifest, write_manifest
from spectra_ml.qc import HARD_FAIL
from spectra_ml.united import build_united_dataset
from spectra_ml.united_stage import CLASS_NAMES, build_united_stage_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "outputs", "blinded_multiclass")
FROZEN = os.path.join(STUDY, "frozen")
FAMILIES = ("dummy", "pca_mnlogreg", "pls_mnlogreg", "linear_svm_mc_reduced",
            "random_forest_reduced_pca", "hgb_reduced_pca", "rbf_svm", "knn",
            "mlp_multiclass", "cnn1d_multiclass")


def _assert_identity_lookup_is_target_free(df: pd.DataFrame) -> pd.DataFrame:
    """Pure guard (no file I/O, directly unit-testable): the identity lookup
    used by this script must NEVER carry the label -- structurally the same
    invariant spectra_ml.blinding.load_blinded_test_manifest already enforces
    for blinded_test_manifest.csv, reproduced here for the separate identity
    lookup file this script reads instead."""
    forbidden = {"target"} & set(df.columns)
    if forbidden:
        raise RuntimeError(f"prediction_identity_lookup.csv unexpectedly contains {forbidden}; refusing to predict.")
    return df


def _load_identity_lookup():
    path = os.path.join(STUDY, "private", "prediction_identity_lookup.csv")
    df = pd.read_csv(path, dtype={"study_id": str, "real_patient_id": str})
    return _assert_identity_lookup_is_target_free(df)


def main():
    verify_manifest(os.path.join(FROZEN, "frozen_manifest.json"), base_dir=STUDY)

    blind = load_blinded_test_manifest(os.path.join(STUDY, "blinded_test_manifest.csv"))
    identity = _load_identity_lookup()
    mapping = blind.merge(identity[["study_id", "real_patient_id"]], on="study_id", validate="one_to_one")
    assert "target" not in mapping.columns

    # Point at the FROZEN copies (frozen/harmonization/), never the mutable
    # outputs/harmonization_v2/ working directory (Part XXII).
    ud = build_united_dataset(transfer_dir=os.path.join(FROZEN, "harmonization"))
    usd = build_united_stage_dataset(ud)
    keep = (usd.qc_status != HARD_FAIL) & ~np.isnan(usd.stage_class)
    spectra, patient_id, instrument = usd.spectra[keep], usd.patient_id[keep], usd.instrument[keep]
    waves = usd.waves

    frozen_waves = np.load(os.path.join(FROZEN, "wavelength_grid.npy"))
    if not (frozen_waves.shape == waves.shape and np.allclose(frozen_waves, waves, atol=1e-4)):
        raise ValueError("frozen wavelength_grid.npy does not match the rebuilt dataset's grid.")

    blind_ids = set(mapping["real_patient_id"])
    row_mask = np.isin(patient_id, list(blind_ids))
    if len(np.unique(patient_id[row_mask])) != len(blind_ids):
        raise ValueError("some blind-test real_patient_id(s) from the identity lookup were not found in the "
                          "rebuilt modeling set -- investigate before predicting.")
    X_blind = spectra[row_mask]
    pid_blind = patient_id[row_mask]
    inst_blind = instrument[row_mask]

    sid_for_pid = dict(zip(mapping["real_patient_id"], mapping["study_id"]))
    inst_for_pid = dict(zip(pid_blind, inst_blind))

    rows = []
    for family in FAMILIES:
        bundle = joblib.load(os.path.join(FROZEN, f"final_model_{family}.joblib"))
        prep, model, class_labels = bundle["preprocessing"], bundle["model"], bundle["class_labels"]
        Xt = prep.transform(X_blind)
        # rbf_svm has no predict_proba (has_decision_function=True convention, see
        # spectra_ml.multiclass._fit_predict_proba_multiclass's own docstring/branch --
        # mirrored here exactly): fall back to a per-class sigmoid-squashed
        # decision_function for ranking-shaped "probabilities" (not calibrated,
        # columns need not sum to 1). hasattr() correctly returns False here even
        # though Pipeline/SVC expose predict_proba as a conditional (available_if)
        # attribute that raises AttributeError on access when probability=False.
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xt)
        else:
            raw = np.asarray(model.decision_function(Xt))
            if raw.ndim == 1:
                raw = np.column_stack([-raw, raw])
            probs = 1.0 / (1.0 + np.exp(-raw))
        classes_ = list(model.classes_)
        col_order = [classes_.index(c) for c in class_labels]
        probs = probs[:, col_order]
        dummy_target = np.zeros(len(pid_blind))  # placeholder only -- aggregate_to_patient_multiclass requires
        # a `target` column but this script never reads or uses it (see docstring); it is dropped below
        # before writing predictions, so no label information reaches this "blind" step.
        patient_df = aggregate_to_patient_multiclass(probs, pid_blind, dummy_target, class_labels)
        for _, rec in patient_df.iterrows():
            row = {"study_id": sid_for_pid[rec["patient_id"]], "instrument": inst_for_pid[rec["patient_id"]],
                   "model": family, "n_spectra": int((pid_blind == rec["patient_id"]).sum())}
            probs_rec = {f"prob_{CLASS_NAMES[c]}": float(rec[f"prob_{c}"]) for c in class_labels}
            row.update(probs_rec)
            pred_class = class_labels[int(np.argmax([probs_rec[f"prob_{CLASS_NAMES[c]}"] for c in class_labels]))]
            row["predicted_class"] = CLASS_NAMES[pred_class]
            rows.append(row)

    predictions = pd.DataFrame(rows).sort_values(["model", "study_id"]).reset_index(drop=True)
    assert "target" not in predictions.columns and "real_patient_id" not in predictions.columns
    out = os.path.join(STUDY, "blinded_predictions.csv")
    predictions.to_csv(out, index=False)
    write_manifest([out], os.path.join(STUDY, "blinded_predictions_manifest.json"), base_dir=STUDY, extra={
        "state": "PREDICTIONS_FROZEN_BEFORE_UNBLINDING",
        "frozen_manifest_sha256": sha256_file(os.path.join(FROZEN, "frozen_manifest.json")),
    })
    print(f"Wrote {len(predictions)} blind prediction rows ({predictions.study_id.nunique()} patients x "
          f"{len(FAMILIES)} models) -> {out}")


if __name__ == "__main__":
    main()
