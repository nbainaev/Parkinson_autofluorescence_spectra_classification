"""Stage 5: group-aware nested CV infrastructure (build_project_and_exp.md
sections 9-10). Every split is by patient_id; no spectrum of one patient may
appear in two different folds, in either the outer or any inner split.

This module only produces indices/generators. It does not touch preprocessing,
models, or QC -- ExperimentRunner (later stage) wires those together per fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class LeakageError(AssertionError):
    """Raised when a patient_id appears in two sides of a split that must be disjoint."""


def assert_no_patient_leakage(patient_id: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray,
                               label_a: str = "train", label_b: str = "test") -> None:
    ids_a = set(patient_id[idx_a])
    ids_b = set(patient_id[idx_b])
    overlap = ids_a & ids_b
    if overlap:
        raise LeakageError(
            f"{len(overlap)} patient_id(s) appear in both {label_a!r} and {label_b!r}: "
            f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
        )


def assert_both_classes_present(target: np.ndarray, idx: np.ndarray, split_name: str = "fold") -> None:
    y = target[idx]
    classes = set(np.unique(y[~np.isnan(y)]).tolist())
    if classes != {0.0, 1.0}:
        raise ValueError(f"{split_name} is missing a class: found {classes}, expected {{0.0, 1.0}}")


def choose_n_splits(patient_target: np.ndarray, max_splits: int = 5, min_splits: int = 3) -> int:
    """Adaptive fold count (section 10): use `max_splits` if both classes have
    enough patients, fall back to `min_splits`; never exceed the minority
    class's patient count (a fold cannot test on more minority patients than
    exist)."""
    counts = np.array([np.sum(patient_target == c) for c in (0.0, 1.0)])
    if (counts == 0).any():
        raise ValueError(f"one class has zero patients: counts={counts}")
    min_class = int(counts.min())
    n = max_splits if min_class >= max_splits * 2 else min_splits
    n = min(n, min_class)
    if n < 2:
        raise ValueError(f"minority class has only {min_class} patient(s); cannot form >=2 folds.")
    return n


@dataclass
class Fold:
    fold_index: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def group_stratified_splits(patient_id: np.ndarray, target: np.ndarray, n_splits: int | None = None,
                             seed: int = 42, max_splits: int = 5, min_splits: int = 3) -> list[Fold]:
    """Row-level (spectrum-level) indices, grouped and stratified by patient.
    `target` must be constant within each patient_id (enforced upstream by
    schema validation / compute_patient_representative_spectra)."""
    if n_splits is None:
        uniq_patients, first_idx = np.unique(patient_id, return_index=True)
        n_splits = choose_n_splits(target[first_idx])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for i, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(target)), target, groups=patient_id)):
        assert_no_patient_leakage(patient_id, train_idx, test_idx, "outer_train", "outer_test")
        assert_both_classes_present(target, train_idx, f"outer_train[{i}]")
        assert_both_classes_present(target, test_idx, f"outer_test[{i}]")
        folds.append(Fold(i, train_idx, test_idx))
    return folds


@dataclass
class NestedFold:
    outer_fold_index: int
    outer_train_idx: np.ndarray
    outer_test_idx: np.ndarray
    inner_folds: list[Fold]   # indices are into the FULL array, but only ever drawn from outer_train_idx


def nested_group_stratified_splits(patient_id: np.ndarray, target: np.ndarray,
                                    outer_n_splits: int | None = None, inner_n_splits: int | None = None,
                                    seed: int = 42, max_splits: int = 5, min_splits: int = 3) -> list[NestedFold]:
    """Outer StratifiedGroupKFold for unbiased evaluation; inner StratifiedGroupKFold
    fit ONLY on each outer_train (never sees outer_test) for preprocessing/
    reducer/hyperparameter selection (section 10)."""
    outer_folds = group_stratified_splits(patient_id, target, outer_n_splits, seed, max_splits, min_splits)

    nested = []
    for outer in outer_folds:
        sub_patient_id = patient_id[outer.train_idx]
        sub_target = target[outer.train_idx]
        inner_local_folds = group_stratified_splits(sub_patient_id, sub_target, inner_n_splits,
                                                      seed + 1000 + outer.fold_index, max_splits, min_splits)
        # map inner fold indices (local to outer_train) back to global row indices
        inner_global_folds = [
            Fold(f.fold_index, outer.train_idx[f.train_idx], outer.train_idx[f.test_idx])
            for f in inner_local_folds
        ]
        for f in inner_global_folds:
            assert_no_patient_leakage(patient_id, f.train_idx, f.test_idx, "inner_train", "inner_val")
            assert_no_patient_leakage(patient_id, f.test_idx, outer.test_idx, "inner_val", "outer_test")
        nested.append(NestedFold(outer.fold_index, outer.train_idx, outer.test_idx, inner_global_folds))
    return nested


# ---------------------------------------------------------------------------
# double_blind_exp.md Part II: a ONE-TIME patient-level stratified
# development/blind-test holdout split (not k-fold). Built on top of the same
# leakage/class-presence primitives as the nested-CV splitters above.
# ---------------------------------------------------------------------------

def patient_level_holdout_split(patient_id: np.ndarray, target: np.ndarray, test_size: float, seed: int,
                                 singleton_class_destination: str = "dev") -> tuple[np.ndarray, np.ndarray]:
    """Row-level (spectrum-level) indices for a single patient-level stratified
    holdout split: `round(n_class * test_size)` patients of each class go to
    the TEST side, the rest to DEV. `target` must be constant within each
    patient_id (as elsewhere in this module).

    Classes with only ONE patient cannot be split at all (rounding would put
    that single patient's entire class representation on one side no matter
    what) -- `singleton_class_destination` ("dev" or "test") decides which
    side that lone patient goes to; the caller must make and document this
    judgment call explicitly (e.g. double_blind_exp.md Part XXV needs >=1
    patient per class in the primary blind-test cohort to compute ROC-AUC, so
    a singleton class may need to go to "test" there even though the default
    here is the more conservative "dev").

    Returns (dev_row_idx, test_row_idx), row-level indices into the input
    arrays. Raises via assert_no_patient_leakage if construction ever produced
    an overlap (defensive; should be unreachable by construction).
    """
    if singleton_class_destination not in ("dev", "test"):
        raise ValueError(f"singleton_class_destination must be 'dev' or 'test', got {singleton_class_destination!r}")
    if not (0.0 <= test_size <= 1.0):
        raise ValueError(f"test_size must be in [0, 1], got {test_size}")

    uniq_patients, first_idx = np.unique(patient_id, return_index=True)
    patient_target = target[first_idx]
    rng = np.random.default_rng(seed)

    dev_patients: list = []
    test_patients: list = []
    for cls in sorted(set(patient_target[~np.isnan(patient_target)].tolist())):
        cls_patients = uniq_patients[patient_target == cls]
        cls_patients = cls_patients[rng.permutation(len(cls_patients))]
        n_cls = len(cls_patients)
        if n_cls == 1:
            (test_patients if singleton_class_destination == "test" else dev_patients).extend(cls_patients.tolist())
            continue
        n_test = int(round(n_cls * test_size))
        n_test = min(max(n_test, 0), n_cls - 1)  # always leave >=1 patient of this class in dev
        test_patients.extend(cls_patients[:n_test].tolist())
        dev_patients.extend(cls_patients[n_test:].tolist())

    dev_set, test_set = set(dev_patients), set(test_patients)
    dev_idx = np.where(np.isin(patient_id, list(dev_set)))[0]
    test_idx = np.where(np.isin(patient_id, list(test_set)))[0]
    assert_no_patient_leakage(patient_id, dev_idx, test_idx, "development", "blind_test")
    return dev_idx, test_idx


# ---------------------------------------------------------------------------
# Multiclass (N>2 classes, e.g. Hoehn-Yahr PD stage 0-3) group-stratified
# splitting. Additive siblings of the binary helpers above -- NOT used by any
# existing binary pipeline (outputs/old, outputs/united, outputs/final_study*),
# which all keep using the {0.0, 1.0}-only functions unchanged.
# ---------------------------------------------------------------------------

def assert_all_classes_present(target: np.ndarray, class_labels, idx: np.ndarray,
                                split_name: str = "fold") -> None:
    y = target[idx]
    classes = set(np.unique(y[~np.isnan(y)]).tolist())
    expected = set(float(c) for c in class_labels)
    if classes != expected:
        raise ValueError(f"{split_name} is missing a class: found {classes}, expected {expected}")


def choose_n_splits_multiclass(patient_target: np.ndarray, class_labels, max_splits: int = 5,
                                min_splits: int = 3) -> int:
    """Multiclass generalization of choose_n_splits (section 10): use
    `max_splits` only if EVERY class has enough patients (>= 2*max_splits),
    else fall back to `min_splits`; never exceed the smallest class's patient
    count (a fold cannot test on more of a class's patients than exist)."""
    class_labels = list(class_labels)
    counts = np.array([np.sum(patient_target == c) for c in class_labels])
    if (counts == 0).any():
        raise ValueError(f"a class has zero patients: labels={class_labels} counts={counts}")
    min_class = int(counts.min())
    n = max_splits if min_class >= max_splits * 2 else min_splits
    n = min(n, min_class)
    if n < 2:
        raise ValueError(f"smallest class has only {min_class} patient(s); cannot form >=2 folds.")
    return n


def group_stratified_splits_multiclass(patient_id: np.ndarray, target: np.ndarray, class_labels,
                                        n_splits: int | None = None, seed: int = 42, max_splits: int = 5,
                                        min_splits: int = 3) -> list[Fold]:
    """Row-level (spectrum-level) indices, grouped and stratified by patient,
    for an arbitrary set of classes (`class_labels`). `target` must be
    constant within each patient_id, as elsewhere in this module.
    StratifiedGroupKFold natively supports multiclass `y` (sklearn docs), so
    no algorithmic change is needed beyond the class-presence/fold-count
    checks being multiclass-aware."""
    class_labels = list(class_labels)
    if n_splits is None:
        uniq_patients, first_idx = np.unique(patient_id, return_index=True)
        n_splits = choose_n_splits_multiclass(target[first_idx], class_labels, max_splits, min_splits)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for i, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(target)), target, groups=patient_id)):
        assert_no_patient_leakage(patient_id, train_idx, test_idx, "outer_train", "outer_test")
        assert_all_classes_present(target, class_labels, train_idx, f"outer_train[{i}]")
        assert_all_classes_present(target, class_labels, test_idx, f"outer_test[{i}]")
        folds.append(Fold(i, train_idx, test_idx))
    return folds


def nested_group_stratified_splits_multiclass(patient_id: np.ndarray, target: np.ndarray, class_labels,
                                               outer_n_splits: int | None = None,
                                               inner_n_splits: int | None = None, seed: int = 42,
                                               max_splits: int = 5, min_splits: int = 3) -> list[NestedFold]:
    """Multiclass sibling of nested_group_stratified_splits."""
    class_labels = list(class_labels)
    outer_folds = group_stratified_splits_multiclass(patient_id, target, class_labels, outer_n_splits, seed,
                                                        max_splits, min_splits)

    nested = []
    for outer in outer_folds:
        sub_patient_id = patient_id[outer.train_idx]
        sub_target = target[outer.train_idx]
        inner_local_folds = group_stratified_splits_multiclass(
            sub_patient_id, sub_target, class_labels, inner_n_splits, seed + 1000 + outer.fold_index,
            max_splits, min_splits,
        )
        inner_global_folds = [
            Fold(f.fold_index, outer.train_idx[f.train_idx], outer.train_idx[f.test_idx])
            for f in inner_local_folds
        ]
        for f in inner_global_folds:
            assert_no_patient_leakage(patient_id, f.train_idx, f.test_idx, "inner_train", "inner_val")
            assert_no_patient_leakage(patient_id, f.test_idx, outer.test_idx, "inner_val", "outer_test")
        nested.append(NestedFold(outer.fold_index, outer.train_idx, outer.test_idx, inner_global_folds))
    return nested
