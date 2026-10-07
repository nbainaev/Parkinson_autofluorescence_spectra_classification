import numpy as np
import pytest

from spectra_ml.splitting import (Fold, LeakageError, assert_both_classes_present, assert_no_patient_leakage,
                                   choose_n_splits, group_stratified_splits, nested_group_stratified_splits)


def make_synthetic(n_park=60, n_control=60, spectra_per_patient=8, seed=0):
    rng = np.random.default_rng(seed)
    patient_id, target = [], []
    for cls, n in [(1, n_park), (0, n_control)]:
        for p in range(n):
            k = rng.integers(1, spectra_per_patient + 1)
            patient_id += [f"{'P' if cls else 'C'}{p}"] * k
            target += [cls] * k
    return np.array(patient_id), np.array(target, dtype=float)


class TestAssertNoPatientLeakage:
    def test_disjoint_ok(self):
        pid = np.array(["a", "a", "b", "b", "c"])
        assert_no_patient_leakage(pid, np.array([0, 1]), np.array([2, 3, 4]))  # no error

    def test_overlap_raises(self):
        pid = np.array(["a", "a", "b", "b", "c"])
        with pytest.raises(LeakageError):
            assert_no_patient_leakage(pid, np.array([0, 1, 2]), np.array([2, 3, 4]))


class TestAssertBothClassesPresent:
    def test_both_present_ok(self):
        target = np.array([0.0, 1.0, 0.0, 1.0])
        assert_both_classes_present(target, np.array([0, 1, 2, 3]))

    def test_missing_class_raises(self):
        target = np.array([0.0, 0.0, 0.0, 1.0])
        with pytest.raises(ValueError):
            assert_both_classes_present(target, np.array([0, 1, 2]))


class TestChooseNSplits:
    def test_balanced_large_uses_max(self):
        y = np.array([0.0] * 50 + [1.0] * 50)
        assert choose_n_splits(y, max_splits=5, min_splits=3) == 5

    def test_small_minority_uses_min_or_less(self):
        y = np.array([0.0] * 4 + [1.0] * 50)
        n = choose_n_splits(y, max_splits=5, min_splits=3)
        assert n <= 4

    def test_zero_class_raises(self):
        y = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            choose_n_splits(y)

    def test_too_small_minority_raises(self):
        y = np.array([0.0] + [1.0] * 50)
        with pytest.raises(ValueError):
            choose_n_splits(y, max_splits=5, min_splits=3)


class TestGroupStratifiedSplits:
    def test_no_leakage_across_folds(self):
        pid, target = make_synthetic()
        folds = group_stratified_splits(pid, target, seed=1)
        for f in folds:
            assert_no_patient_leakage(pid, f.train_idx, f.test_idx)

    def test_every_test_spectrum_used_exactly_once(self):
        pid, target = make_synthetic()
        folds = group_stratified_splits(pid, target, seed=1)
        all_test = np.concatenate([f.test_idx for f in folds])
        assert sorted(all_test.tolist()) == list(range(len(target)))
        assert len(all_test) == len(set(all_test.tolist()))

    def test_every_fold_has_both_classes_train_and_test(self):
        pid, target = make_synthetic()
        folds = group_stratified_splits(pid, target, seed=1)
        for f in folds:
            assert_both_classes_present(target, f.train_idx)
            assert_both_classes_present(target, f.test_idx)

    def test_patient_never_split_across_own_spectra(self):
        """A single patient's spectra must all land on the same side of every fold."""
        pid, target = make_synthetic()
        folds = group_stratified_splits(pid, target, seed=1)
        for f in folds:
            train_patients = set(pid[f.train_idx])
            test_patients = set(pid[f.test_idx])
            assert train_patients.isdisjoint(test_patients)

    def test_reproducible_with_same_seed(self):
        pid, target = make_synthetic()
        folds_a = group_stratified_splits(pid, target, seed=7)
        folds_b = group_stratified_splits(pid, target, seed=7)
        for fa, fb in zip(folds_a, folds_b):
            assert np.array_equal(fa.train_idx, fb.train_idx)
            assert np.array_equal(fa.test_idx, fb.test_idx)

    def test_severe_class_imbalance_like_real_data(self):
        """Mirrors the actual instrument='old' patient-level imbalance (park=140,
        control=43) -- must still produce valid folds with both classes throughout."""
        pid, target = make_synthetic(n_park=140, n_control=43, seed=3)
        folds = group_stratified_splits(pid, target, seed=3)
        assert len(folds) >= 2
        for f in folds:
            assert_no_patient_leakage(pid, f.train_idx, f.test_idx)
            assert_both_classes_present(target, f.train_idx)
            assert_both_classes_present(target, f.test_idx)


class TestNestedGroupStratifiedSplits:
    def test_no_leakage_outer_and_inner(self):
        pid, target = make_synthetic(n_park=80, n_control=80, seed=2)
        nested = nested_group_stratified_splits(pid, target, seed=2)
        for outer in nested:
            assert_no_patient_leakage(pid, outer.outer_train_idx, outer.outer_test_idx,
                                       "outer_train", "outer_test")
            for inner in outer.inner_folds:
                assert_no_patient_leakage(pid, inner.train_idx, inner.test_idx, "inner_train", "inner_val")
                # inner val/train must both be subsets of THIS outer's train set only
                assert set(inner.train_idx).issubset(set(outer.outer_train_idx))
                assert set(inner.test_idx).issubset(set(outer.outer_train_idx))
                # and must never touch this outer fold's held-out test patients
                assert_no_patient_leakage(pid, inner.test_idx, outer.outer_test_idx, "inner_val", "outer_test")
                assert_no_patient_leakage(pid, inner.train_idx, outer.outer_test_idx, "inner_train", "outer_test")

    def test_inner_folds_never_see_outer_test_patients(self):
        pid, target = make_synthetic(n_park=80, n_control=80, seed=5)
        nested = nested_group_stratified_splits(pid, target, seed=5)
        for outer in nested:
            outer_test_patients = set(pid[outer.outer_test_idx])
            for inner in outer.inner_folds:
                inner_patients = set(pid[inner.train_idx]) | set(pid[inner.test_idx])
                assert inner_patients.isdisjoint(outer_test_patients)


# ---------------------------------------------------------------------------
# Multiclass (N>2 classes) siblings
# ---------------------------------------------------------------------------
from spectra_ml.splitting import (assert_all_classes_present, choose_n_splits_multiclass,  # noqa: E402
                                   group_stratified_splits_multiclass, nested_group_stratified_splits_multiclass)


def make_synthetic_multiclass(n_per_class=30, spectra_per_patient=4, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    patient_id, target = [], []
    for cls in range(n_classes):
        for p in range(n_per_class):
            k = rng.integers(1, spectra_per_patient + 1)
            patient_id += [f"C{cls}_{p}"] * k
            target += [float(cls)] * k
    return np.array(patient_id), np.array(target, dtype=float)


class TestAssertAllClassesPresent:
    def test_all_present_ok(self):
        target = np.array([0.0, 1.0, 2.0, 3.0])
        assert_all_classes_present(target, (0, 1, 2, 3), np.array([0, 1, 2, 3]))

    def test_missing_class_raises(self):
        target = np.array([0.0, 1.0, 2.0, 2.0])
        with pytest.raises(ValueError):
            assert_all_classes_present(target, (0, 1, 2, 3), np.array([0, 1, 2, 3]))


class TestChooseNSplitsMulticlass:
    def test_uses_max_splits_when_all_classes_large(self):
        target = np.array([0.0] * 20 + [1.0] * 20 + [2.0] * 20 + [3.0] * 20)
        assert choose_n_splits_multiclass(target, (0, 1, 2, 3), max_splits=5, min_splits=3) == 5

    def test_falls_back_to_min_splits_when_a_class_small(self):
        target = np.array([0.0] * 20 + [1.0] * 20 + [2.0] * 20 + [3.0] * 6)
        assert choose_n_splits_multiclass(target, (0, 1, 2, 3), max_splits=5, min_splits=3) == 3

    def test_zero_patients_in_a_class_raises(self):
        target = np.array([0.0] * 20 + [1.0] * 20 + [2.0] * 20)
        with pytest.raises(ValueError):
            choose_n_splits_multiclass(target, (0, 1, 2, 3), max_splits=5, min_splits=3)


class TestGroupStratifiedSplitsMulticlass:
    def test_no_leakage_and_all_classes_present_every_fold(self):
        pid, target = make_synthetic_multiclass(n_per_class=30, seed=1)
        folds = group_stratified_splits_multiclass(pid, target, (0, 1, 2, 3), seed=1)
        assert len(folds) >= 2
        for f in folds:
            assert_no_patient_leakage(pid, f.train_idx, f.test_idx)
            assert_all_classes_present(target, (0, 1, 2, 3), f.train_idx)
            assert_all_classes_present(target, (0, 1, 2, 3), f.test_idx)

    def test_imbalanced_classes_still_produce_valid_folds(self):
        # mirrors the real stage cohort's imbalance shape (stage2 smallest)
        rng = np.random.default_rng(3)
        patient_id, tgt = [], []
        for cls, n in [(0.0, 76), (1.0, 28), (2.0, 23), (3.0, 57)]:
            for p in range(n):
                k = rng.integers(1, 4)
                patient_id += [f"C{int(cls)}_{p}"] * k
                tgt += [cls] * k
        patient_id = np.array(patient_id)
        tgt = np.array(tgt, dtype=float)
        folds = group_stratified_splits_multiclass(patient_id, tgt, (0.0, 1.0, 2.0, 3.0), seed=3)
        for f in folds:
            assert_no_patient_leakage(patient_id, f.train_idx, f.test_idx)
            assert_all_classes_present(tgt, (0.0, 1.0, 2.0, 3.0), f.train_idx)
            assert_all_classes_present(tgt, (0.0, 1.0, 2.0, 3.0), f.test_idx)


class TestNestedGroupStratifiedSplitsMulticlass:
    def test_no_leakage_across_outer_inner_and_all_classes_present(self):
        pid, target = make_synthetic_multiclass(n_per_class=30, seed=4)
        nested = nested_group_stratified_splits_multiclass(pid, target, (0, 1, 2, 3), seed=4)
        assert len(nested) >= 2
        for outer in nested:
            assert_no_patient_leakage(pid, outer.outer_train_idx, outer.outer_test_idx)
            assert_all_classes_present(target, (0, 1, 2, 3), outer.outer_train_idx)
            assert_all_classes_present(target, (0, 1, 2, 3), outer.outer_test_idx)
            for inner in outer.inner_folds:
                assert_no_patient_leakage(pid, inner.train_idx, inner.test_idx)
                assert_no_patient_leakage(pid, inner.test_idx, outer.outer_test_idx, "inner_val", "outer_test")
                assert_all_classes_present(target, (0, 1, 2, 3), inner.train_idx)
                assert_all_classes_present(target, (0, 1, 2, 3), inner.test_idx)
