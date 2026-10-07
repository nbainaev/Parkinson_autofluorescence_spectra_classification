"""Lean tests for the multiclass torch training additions (task 2 of the
united_multiclass_v2 nonlinear/neural extension): group_aware_val_split_
multiclass, train_torch_multiclass_classifier, MLPMulticlassClassifier,
CNN1DMulticlassClassifier. Mirrors tests/test_neural.py's binary-sibling
style; kept small per runtime discipline, not exhaustive.
"""
import numpy as np

from spectra_ml.models.cnn1d import CNN1DMulticlassClassifier, build_cnn1d_multiclass_candidates
from spectra_ml.models.mlp import MLPMulticlassClassifier, build_mlp_multiclass_candidates
from spectra_ml.models.torch_common import group_aware_val_split_multiclass


def make_synthetic_multiclass(n_per_class=20, spectra_per_patient=3, n_wave=278, n_classes=4, seed=0, signal=3.0):
    rng = np.random.default_rng(seed)
    patient_id, target, spectra = [], [], []
    for cls in range(n_classes):
        for p in range(n_per_class):
            base = rng.normal(size=n_wave) * 0.2
            band = n_wave // n_classes
            base[cls * band:(cls + 1) * band] += signal
            for _ in range(spectra_per_patient):
                spectra.append(base + rng.normal(scale=0.15, size=n_wave))
                patient_id.append(f"C{cls}P{p}")
                target.append(float(cls))
    return np.array(spectra), np.array(patient_id), np.array(target)


class TestGroupAwareValSplitMulticlass:
    def test_no_patient_split_across_train_val(self):
        _, pid, target = make_synthetic_multiclass()
        train_idx, val_idx = group_aware_val_split_multiclass(pid, target, [0.0, 1.0, 2.0, 3.0],
                                                                val_fraction=0.2, seed=0)
        assert set(pid[train_idx]).isdisjoint(set(pid[val_idx]))

    def test_val_nonempty_and_multiclass_when_patient_count_allows(self):
        _, pid, target = make_synthetic_multiclass()
        train_idx, val_idx = group_aware_val_split_multiclass(pid, target, [0.0, 1.0, 2.0, 3.0],
                                                                val_fraction=0.2, seed=0)
        assert len(val_idx) > 0 and len(train_idx) > 0
        assert len(np.unique(target[val_idx])) == 4

    def test_degenerate_case_falls_back_to_no_val(self):
        # Only 1 patient per class -- cannot form >=2 group-stratified folds.
        _, pid, target = make_synthetic_multiclass(n_per_class=1, spectra_per_patient=2)
        train_idx, val_idx = group_aware_val_split_multiclass(pid, target, [0.0, 1.0, 2.0, 3.0],
                                                                val_fraction=0.2, seed=0)
        assert len(val_idx) == 0
        assert len(train_idx) == len(target)


class TestMLPMulticlassClassifier:
    def test_fit_predict_proba_shape_and_simplex(self):
        spectra, pid, target = make_synthetic_multiclass(seed=1)
        model = MLPMulticlassClassifier(hidden_dims=[32, 16], max_epochs=40, patience=8, random_state=1,
                                         n_components=10)
        model.fit(spectra, target, sample_weight=np.ones(len(target)), patient_id=pid)
        proba = model.predict_proba(spectra)
        assert proba.shape == (len(target), 4)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
        assert model.classes_ == [0.0, 1.0, 2.0, 3.0]

    def test_learns_separable_signal(self):
        spectra, pid, target = make_synthetic_multiclass(seed=2, signal=4.0)
        model = MLPMulticlassClassifier(hidden_dims=[32, 16], max_epochs=80, patience=10, random_state=2,
                                         n_components=10)
        model.fit(spectra, target, sample_weight=np.ones(len(target)), patient_id=pid)
        pred = model.predict(spectra)
        accuracy = float(np.mean(pred == target))
        assert accuracy > 0.7


class TestCNN1DMulticlassClassifier:
    def test_fit_predict_proba_shape_and_simplex(self):
        spectra, pid, target = make_synthetic_multiclass(seed=3)
        model = CNN1DMulticlassClassifier(channels=(8, 16), kernel_sizes=(7, 5), max_epochs=40, patience=8,
                                           random_state=3)
        model.fit(spectra, target, sample_weight=np.ones(len(target)), patient_id=pid)
        proba = model.predict_proba(spectra)
        assert proba.shape == (len(target), 4)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


class TestMulticlassCandidateBuilders:
    def test_mlp_multiclass_candidates_single_config_requires_patient_groups(self):
        cands = build_mlp_multiclass_candidates()
        assert len(cands) == 1
        assert cands[0].requires_patient_groups
        assert cands[0].family == "mlp_multiclass"

    def test_cnn1d_multiclass_candidates_single_config_requires_patient_groups(self):
        cands = build_cnn1d_multiclass_candidates()
        assert len(cands) == 1
        assert cands[0].requires_patient_groups
        assert cands[0].family == "cnn1d_multiclass"
