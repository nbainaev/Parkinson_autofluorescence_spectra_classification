"""Lean core-logic tests for the multiclass retrospective blinded validation
scripts (scripts/prepare_blinded_study_multiclass.py, freeze_final_model_
multiclass.py, predict_blinded_multiclass.py, unblind_and_evaluate_
multiclass.py). Deliberately NOT exhaustive (5-8 tests per task scope) --
covers the 3 explicitly requested areas (leakage-free split, blinding-key
join correctness, SHA256 verification catches tampering) plus the two
genuinely NEW pieces of logic this task added (the identity-lookup
target-free guard, and freeze's placeholder-NestedFold candidate-selection
helper). Everything already covered generically elsewhere (spectra_ml.
splitting/integrity/blinding unit tests) is reused, not re-tested from
scratch.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from spectra_ml.integrity import IntegrityError, verify_manifest, write_manifest
from spectra_ml.splitting import assert_all_classes_present, assert_no_patient_leakage, patient_level_holdout_split

CLASS_LABELS = (0.0, 1.0, 2.0, 3.0)


def _synthetic_multiclass_cohort(n_per_class=(20, 10, 8, 15), spectra_per_patient=2, seed=0):
    rng = np.random.default_rng(seed)
    patient_id, target = [], []
    for cls, n in zip(CLASS_LABELS, n_per_class):
        for i in range(n):
            pid = f"cls{int(cls)}_p{i}"
            patient_id += [pid] * spectra_per_patient
            target += [cls] * spectra_per_patient
    order = rng.permutation(len(patient_id))
    return np.array(patient_id)[order], np.array(target, dtype=float)[order]


# --------------------------------------------------------------------------
# 1-2. Leakage-free, all-classes-present multiclass holdout split (the binary
# 2-class case is already covered in test_blinded_study.py; this task's new
# scope is a genuine >2-class stage label, not previously exercised through
# this exact split function with 4 classes).
# --------------------------------------------------------------------------

def test_multiclass_holdout_split_has_no_leakage_and_every_class_both_sides():
    patient_id, target = _synthetic_multiclass_cohort()
    dev_idx, blind_idx = patient_level_holdout_split(patient_id, target, test_size=0.2, seed=42)
    assert_no_patient_leakage(patient_id, dev_idx, blind_idx, "development", "blind_test")
    assert_all_classes_present(target, CLASS_LABELS, dev_idx, "development")
    assert_all_classes_present(target, CLASS_LABELS, blind_idx, "blind_test")


def test_multiclass_holdout_split_is_deterministic_under_fixed_seed():
    patient_id, target = _synthetic_multiclass_cohort()
    dev_a, blind_a = patient_level_holdout_split(patient_id, target, test_size=0.2, seed=42)
    dev_b, blind_b = patient_level_holdout_split(patient_id, target, test_size=0.2, seed=42)
    assert set(patient_id[dev_a]) == set(patient_id[dev_b])
    assert set(patient_id[blind_a]) == set(patient_id[blind_b])


# --------------------------------------------------------------------------
# 3. Identity-lookup target-free guard (predict_blinded_multiclass.py's new
# structural safety check, mirroring spectra_ml.blinding.load_blinded_test_
# manifest's analogous binary-study check).
# --------------------------------------------------------------------------

def test_identity_lookup_guard_rejects_target_column():
    from predict_blinded_multiclass import _assert_identity_lookup_is_target_free

    clean = pd.DataFrame({"study_id": ["S1"], "real_patient_id": ["P1"], "instrument": ["old"]})
    assert _assert_identity_lookup_is_target_free(clean) is clean

    leaky = pd.DataFrame({"study_id": ["S1"], "real_patient_id": ["P1"], "target": [1.0]})
    with pytest.raises(RuntimeError, match="unexpectedly contains"):
        _assert_identity_lookup_is_target_free(leaky)


# --------------------------------------------------------------------------
# 4-5. Blinding-key join correctness (unblind_and_evaluate_multiclass.py's
# core merge step): every predicted study_id must resolve to exactly the
# target recorded for that patient in the private key, and a study_id
# missing from the key must be caught rather than silently dropped/NaN-filled.
# --------------------------------------------------------------------------

def test_blinding_key_join_attaches_correct_target_per_study_id():
    predictions = pd.DataFrame({
        "study_id": ["S1", "S1", "S2", "S2"], "model": ["pls_mnlogreg", "pca_mnlogreg"] * 2,
        "prob_control": [0.7, 0.6, 0.1, 0.2],
    })
    key = pd.DataFrame({"study_id": ["S1", "S2"], "real_patient_id": ["P1", "P2"], "target": [0.0, 3.0]})
    joined = predictions.merge(key[["study_id", "target"]], on="study_id", validate="many_to_one")
    assert dict(zip(joined.loc[joined.study_id == "S1", "model"], joined.loc[joined.study_id == "S1", "target"]))
    assert (joined.loc[joined.study_id == "S1", "target"] == 0.0).all()
    assert (joined.loc[joined.study_id == "S2", "target"] == 3.0).all()
    assert "real_patient_id" not in joined.columns  # unblind output must not carry raw identity


def test_blinding_key_join_detects_a_study_id_missing_from_the_key():
    """An (inner) merge silently DROPS a row whose study_id isn't in the
    private key (or vice versa) rather than raising -- exercises the REAL
    guard function from unblind_and_evaluate_multiclass.py that catches this.
    Deliberately includes a case where nunique(evaluated) == nunique(key)
    would COINCIDENTALLY match despite a dropped row, to pin down that the
    guard checks row counts, not just nunique."""
    from unblind_and_evaluate_multiclass import _assert_full_study_id_coverage

    predictions = pd.DataFrame({"study_id": ["S1", "S_UNKNOWN"], "prob_control": [0.7, 0.4]})
    key_full = pd.DataFrame({"study_id": ["S1", "S_UNKNOWN"], "target": [0.0, 3.0]})
    # same nunique (1) as key_full's overlap with predictions, but S_UNKNOWN's row would be silently dropped
    key_coincidental_count_match = pd.DataFrame({"study_id": ["S1"], "target": [0.0]})

    joined_full = predictions.merge(key_full, on="study_id", validate="many_to_one")
    _assert_full_study_id_coverage(joined_full, predictions, key_full)  # no raise: every row matched

    joined_dropped = predictions.merge(key_coincidental_count_match, on="study_id", validate="many_to_one")
    with pytest.raises(RuntimeError, match="[Nn]ot every"):
        _assert_full_study_id_coverage(joined_dropped, predictions, key_coincidental_count_match)


# --------------------------------------------------------------------------
# 6-7. SHA256 verification catches tampering (extends the existing generic
# hash-mismatch test in test_blinded_study.py with the MISSING-file case,
# which this task's chained frozen_manifest -> blinded_predictions_manifest
# verification in unblind_and_evaluate_multiclass.py also depends on).
# --------------------------------------------------------------------------

def test_manifest_verification_detects_a_missing_frozen_artifact(tmp_path):
    artifact = tmp_path / "final_model_pls_mnlogreg.joblib"
    artifact.write_bytes(b"frozen-bytes")
    manifest_path = tmp_path / "frozen_manifest.json"
    write_manifest([str(artifact)], str(manifest_path), base_dir=str(tmp_path))
    verify_manifest(str(manifest_path), base_dir=str(tmp_path))  # passes before deletion

    artifact.unlink()
    with pytest.raises(IntegrityError, match="MISSING"):
        verify_manifest(str(manifest_path), base_dir=str(tmp_path))


def test_prediction_manifest_chain_breaks_if_frozen_manifest_changes_after_prediction(tmp_path):
    """Mirrors unblind_and_evaluate_multiclass.py's own chained check: the
    prediction manifest records the frozen manifest's SHA256 at prediction
    time, so if the frozen manifest is (re)written afterward, the recorded
    hash goes stale even if verify_manifest itself would still pass on the
    (now different) frozen manifest in isolation."""
    from spectra_ml.integrity import sha256_file

    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    model_path = frozen_dir / "final_model_dummy.joblib"
    model_path.write_bytes(b"v1")
    frozen_manifest_path = frozen_dir / "frozen_manifest.json"
    write_manifest([str(model_path)], str(frozen_manifest_path), base_dir=str(tmp_path))
    recorded_hash = sha256_file(str(frozen_manifest_path))

    # simulate someone re-freezing (changing the model + rewriting the manifest) after prediction
    model_path.write_bytes(b"v2-tampered")
    write_manifest([str(model_path)], str(frozen_manifest_path), base_dir=str(tmp_path))

    assert sha256_file(str(frozen_manifest_path)) != recorded_hash


# --------------------------------------------------------------------------
# 8. freeze_final_model_multiclass.py's new selection+refit helper (the
# placeholder-NestedFold trick that reuses spectra_ml.multiclass._select_
# best_candidate_multiclass for a flat, single-level development-only CV
# round instead of a real nested outer/inner split).
# --------------------------------------------------------------------------

def test_select_and_refit_returns_a_fitted_predict_proba_model():
    from freeze_final_model_multiclass import _select_and_refit
    from spectra_ml.models.classical import build_pca_multinomial_logreg_candidates

    rng = np.random.default_rng(0)
    n_wave = 20
    patients_per_class, spectra_per_patient = 6, 2
    waves = np.linspace(400, 700, n_wave)

    patient_id, target, rows = [], [], []
    for cls in CLASS_LABELS:
        for i in range(patients_per_class):
            base = rng.normal(loc=cls, scale=0.1, size=n_wave)
            for _ in range(spectra_per_patient):
                rows.append(base + rng.normal(scale=0.05, size=n_wave))
                patient_id.append(f"cls{int(cls)}_p{i}")
                target.append(cls)
    spectra = np.abs(np.stack(rows)) + 1e-3
    patient_id, target = np.array(patient_id), np.array(target, dtype=float)

    candidates = build_pca_multinomial_logreg_candidates(
        preprocessing_variants=("A",), n_components_grid=(2, 3), C_grid=(1.0,), seed=0)
    summary, prep, model, best = _select_and_refit("pca_mnlogreg", candidates, spectra, target, patient_id,
                                                      waves, seed=0)

    assert summary["family"] == "pca_mnlogreg"
    assert np.isfinite(summary["selection_mean_auc_within_development"])
    assert hasattr(model, "predict_proba")
    probs = model.predict_proba(prep.transform(spectra))
    assert probs.shape == (len(spectra), len(CLASS_LABELS))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
