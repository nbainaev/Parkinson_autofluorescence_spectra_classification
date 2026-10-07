import numpy as np
import pytest

from spectra_ml.models.cnn1d import CNN1DBinaryClassifier, build_cnn1d_candidates
from spectra_ml.models.mlp import MLPBinaryClassifier, build_mlp_candidates
from spectra_ml.models.torch_common import group_aware_val_split


def make_synthetic(n_park=40, n_control=40, spectra_per_patient=4, n_wave=30, seed=0, signal=2.5):
    rng = np.random.default_rng(seed)
    patient_id, target, spectra = [], [], []
    for cls, n in [(1, n_park), (0, n_control)]:
        for p in range(n):
            base = rng.normal(size=n_wave) * 0.2
            base[5:10] += signal * cls
            for _ in range(spectra_per_patient):
                spectra.append(base + rng.normal(scale=0.15, size=n_wave))
                patient_id.append(f"{'P' if cls else 'C'}{p}")
                target.append(float(cls))
    return np.array(spectra), np.array(patient_id), np.array(target)


class TestGroupAwareValSplit:
    def test_no_patient_split_across_train_val(self):
        spectra, pid, target = make_synthetic()
        train_idx, val_idx = group_aware_val_split(pid, target, val_fraction=0.2, seed=0)
        assert set(pid[train_idx]).isdisjoint(set(pid[val_idx]))

    def test_val_nonempty_for_reasonable_fraction(self):
        spectra, pid, target = make_synthetic()
        train_idx, val_idx = group_aware_val_split(pid, target, val_fraction=0.2, seed=0)
        assert len(val_idx) > 0
        assert len(train_idx) > 0


class TestMLPBinaryClassifier:
    def test_fit_predict_roundtrip(self):
        spectra, pid, target = make_synthetic(seed=1)
        model = MLPBinaryClassifier(hidden_dims=(16,), max_epochs=30, patience=5, random_state=1)
        model.fit(spectra, target, sample_weight=np.ones(len(target)), patient_id=pid)
        proba = model.predict_proba(spectra)
        assert proba.shape == (len(target), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_learns_separable_signal(self):
        spectra, pid, target = make_synthetic(seed=2, signal=3.0)
        model = MLPBinaryClassifier(hidden_dims=(16,), max_epochs=100, patience=15, random_state=2)
        model.fit(spectra, target, sample_weight=np.ones(len(target)), patient_id=pid)
        proba = model.predict_proba(spectra)[:, 1]
        from sklearn.metrics import roc_auc_score
        assert roc_auc_score(target, proba) > 0.8

    def test_save_load_roundtrip_matches_predictions(self, tmp_path):
        spectra, pid, target = make_synthetic(seed=3)
        model = MLPBinaryClassifier(hidden_dims=(16,), max_epochs=20, patience=5, random_state=3)
        model.fit(spectra, target, sample_weight=np.ones(len(target)), patient_id=pid)
        pred_before = model.predict_proba(spectra)

        path = str(tmp_path / "mlp.pt")
        model.save(path)
        loaded = MLPBinaryClassifier.load(path)
        pred_after = loaded.predict_proba(spectra)
        assert np.allclose(pred_before, pred_after)

    def test_get_set_params(self):
        model = MLPBinaryClassifier(hidden_dims=(32,))
        params = model.get_params()
        assert params["hidden_dims"] == [32]
        model.set_params(hidden_dims=[64, 32])
        assert model.hidden_dims == [64, 32]

    def test_early_stopping_triggers_on_overfitting_trajectory(self):
        """Directly exercises the early-stopping branch: an unregularized model
        with a high learning rate on a tiny, noisy dataset should start
        overfitting (val loss rising) well before max_epochs."""
        spectra, pid, target = make_synthetic(seed=4, n_park=10, n_control=10, spectra_per_patient=2,
                                               n_wave=30, signal=0.3)
        model = MLPBinaryClassifier(hidden_dims=(64, 32), dropout=0.0, weight_decay=0.0, learning_rate=0.05,
                                     max_epochs=2000, patience=5, random_state=4)
        model.fit(spectra, target, sample_weight=np.ones(len(target)), patient_id=pid)
        assert len(model.history_["train_loss"]) < 2000
        assert "stopped_epoch" in model.history_


class TestCNN1DBinaryClassifier:
    def test_fit_predict_roundtrip(self):
        spectra, pid, target = make_synthetic(seed=5, n_wave=40)
        model = CNN1DBinaryClassifier(channels=(8, 16), kernel_sizes=(5, 5), max_epochs=30, patience=5,
                                       random_state=5)
        model.fit(spectra, target, sample_weight=np.ones(len(target)), patient_id=pid)
        proba = model.predict_proba(spectra)
        assert proba.shape == (len(target), 2)

    def test_learns_separable_signal(self):
        spectra, pid, target = make_synthetic(seed=6, n_wave=40, signal=3.0)
        model = CNN1DBinaryClassifier(channels=(8, 16), kernel_sizes=(5, 5), max_epochs=100, patience=15,
                                       random_state=6)
        model.fit(spectra, target, sample_weight=np.ones(len(target)), patient_id=pid)
        proba = model.predict_proba(spectra)[:, 1]
        from sklearn.metrics import roc_auc_score
        assert roc_auc_score(target, proba) > 0.75

    def test_save_load_roundtrip(self, tmp_path):
        spectra, pid, target = make_synthetic(seed=7, n_wave=40)
        model = CNN1DBinaryClassifier(channels=(8, 16), kernel_sizes=(5, 5), max_epochs=20, patience=5,
                                       random_state=7)
        model.fit(spectra, target, sample_weight=np.ones(len(target)), patient_id=pid)
        pred_before = model.predict_proba(spectra)
        path = str(tmp_path / "cnn.pt")
        model.save(path)
        loaded = CNN1DBinaryClassifier.load(path)
        assert np.allclose(pred_before, loaded.predict_proba(spectra))


class TestCandidateBuilders:
    def test_mlp_candidates_require_patient_groups(self):
        cands = build_mlp_candidates()
        assert all(c.requires_patient_groups for c in cands)
        assert all(not c.has_decision_function for c in cands)

    def test_cnn_candidates_require_patient_groups(self):
        cands = build_cnn1d_candidates()
        assert all(c.requires_patient_groups for c in cands)
