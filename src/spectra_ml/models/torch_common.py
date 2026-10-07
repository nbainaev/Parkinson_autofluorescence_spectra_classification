"""Shared PyTorch training loop for MLPBinaryClassifier and CNN1DBinaryClassifier
(Stage 11/12, sections 14-16). Both wrap this exact same fit/predict machinery
so ExperimentRunner-style code can treat them identically to sklearn models.

Leakage-relevant design (section 14: "early stopping decisions принимаются
только по inner validation data; не использовать outer test для выбора
epochs"): `fit()` internally carves out a small GROUP-AWARE validation split
from whatever (X, y, patient_id) it receives -- never from data the caller
didn't give it. Since the caller in this pipeline is always an outer_train (or
inner_train) fold, this validation split never touches outer_test.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def group_aware_val_split(patient_id: np.ndarray, target: np.ndarray, val_fraction: float,
                           seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (train_idx, val_idx), split by whole patients, stratified by
    class as best effort (falls back to unstratified if a class has too few
    patients for the requested fraction)."""
    rng = np.random.default_rng(seed)
    uniq_patients, first_idx = np.unique(patient_id, return_index=True)
    patient_target = target[first_idx]

    val_patients = set()
    for cls in (0.0, 1.0):
        cls_patients = uniq_patients[patient_target == cls]
        n_val = max(1, int(round(len(cls_patients) * val_fraction)))
        n_val = min(n_val, len(cls_patients) - 1) if len(cls_patients) > 1 else 0
        if n_val > 0:
            val_patients.update(rng.choice(cls_patients, size=n_val, replace=False).tolist())

    val_mask = np.isin(patient_id, list(val_patients))
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx


def weighted_bce_loss(logits: torch.Tensor, y: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    per_sample = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (per_sample * sample_weight).sum() / sample_weight.sum()


def train_torch_binary_classifier(module: nn.Module, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray,
                                   patient_id: np.ndarray | None, *, lr: float, weight_decay: float,
                                   max_epochs: int, patience: int, val_fraction: float, seed: int,
                                   reshape_for_conv: bool = False) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)

    if patient_id is not None and val_fraction > 0:
        train_idx, val_idx = group_aware_val_split(patient_id, y, val_fraction, seed)
    else:
        train_idx, val_idx = np.arange(len(y)), np.array([], dtype=int)
    has_val = len(val_idx) > 0 and len(np.unique(y[val_idx])) > 1

    def to_tensor(idx):
        x = X[idx]
        if reshape_for_conv:
            x = x[:, None, :]  # (n, 1, n_wave)
        return (torch.tensor(x, dtype=torch.float32), torch.tensor(y[idx], dtype=torch.float32),
                torch.tensor(sample_weight[idx], dtype=torch.float32))

    X_tr, y_tr, w_tr = to_tensor(train_idx)
    if has_val:
        X_val, y_val, w_val = to_tensor(val_idx)

    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_val_loss, epochs_no_improve = None, np.inf, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(max_epochs):
        module.train()
        optimizer.zero_grad()
        logits = module(X_tr).squeeze(-1)
        loss = weighted_bce_loss(logits, y_tr, w_tr)
        loss.backward()
        optimizer.step()
        history["train_loss"].append(float(loss.item()))

        if has_val:
            module.eval()
            with torch.no_grad():
                val_logits = module(X_val).squeeze(-1)
                val_loss = weighted_bce_loss(val_logits, y_val, w_val).item()
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss - 1e-5:
                best_val_loss, epochs_no_improve = val_loss, 0
                best_state = {k: v.clone() for k, v in module.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    if has_val and best_state is not None:
        module.load_state_dict(best_state)
        history["stopped_epoch"] = epoch - epochs_no_improve
    else:
        history["stopped_epoch"] = epoch  # no val signal (e.g. degenerate split) -- ran to max_epochs

    return module, history


def predict_proba_torch(module: nn.Module, X: np.ndarray, reshape_for_conv: bool = False) -> np.ndarray:
    module.eval()
    x = X[:, None, :] if reshape_for_conv else X
    with torch.no_grad():
        logits = module(torch.tensor(x, dtype=torch.float32)).squeeze(-1)
        prob = torch.sigmoid(logits).numpy()
    return np.stack([1 - prob, prob], axis=1)


# ---------------------------------------------------------------------------
# Multiclass siblings (MLPMulticlassClassifier / CNN1DMulticlassClassifier).
# Additive: the binary functions above are untouched (other parts of this
# project depend on them unchanged). CrossEntropyLoss replaces weighted BCE;
# the group-aware validation split is generalized to N classes by reusing
# spectra_ml.splitting's existing multiclass-aware StratifiedGroupKFold helper
# (group_stratified_splits_multiclass) rather than reimplementing the
# per-class patient-sampling loop from scratch.
# ---------------------------------------------------------------------------

def group_aware_val_split_multiclass(patient_id: np.ndarray, target: np.ndarray, class_labels,
                                      val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (train_idx, val_idx), split by whole patients, stratified by
    class via spectra_ml.splitting.group_stratified_splits_multiclass (reused,
    not reimplemented): treats the desired val_fraction as choosing a
    (patient-count-capped) number of StratifiedGroupKFold folds and takes one
    fold's test half as the validation split. Falls back to no validation
    split (train on everything, no early-stopping signal) if the smallest
    class does not have enough patients for >=2 group-stratified folds, or if
    a resulting fold would still be missing a class -- same "degenerate split"
    fallback spirit as the binary group_aware_val_split, generalized to N
    classes instead of silently assuming 2."""
    from spectra_ml.splitting import group_stratified_splits_multiclass

    class_labels = list(class_labels)
    if val_fraction <= 0:
        return np.arange(len(target)), np.array([], dtype=int)
    n_splits = max(2, min(5, int(round(1.0 / val_fraction))))
    try:
        folds = group_stratified_splits_multiclass(patient_id, target, class_labels, n_splits=n_splits, seed=seed)
    except (ValueError, AssertionError):
        return np.arange(len(target)), np.array([], dtype=int)
    return folds[0].train_idx, folds[0].test_idx


def weighted_cross_entropy_loss(logits: torch.Tensor, y_idx: torch.Tensor,
                                 sample_weight: torch.Tensor) -> torch.Tensor:
    per_sample = nn.functional.cross_entropy(logits, y_idx, reduction="none")
    return (per_sample * sample_weight).sum() / sample_weight.sum()


def train_torch_multiclass_classifier(module: nn.Module, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray,
                                       patient_id: np.ndarray | None, class_labels, *, lr: float,
                                       weight_decay: float, max_epochs: int, patience: int, val_fraction: float,
                                       seed: int, reshape_for_conv: bool = False) -> tuple[nn.Module, dict]:
    """Multiclass sibling of train_torch_binary_classifier: CrossEntropyLoss
    over raw logits (shape (n, n_classes)) instead of weighted BCE over a
    single logit; same group-aware-val-split-only-from-caller-supplied-data
    leakage boundary (never touches outer_test, see module docstring)."""
    torch.manual_seed(seed)
    class_labels = list(class_labels)
    class_to_idx = {c: i for i, c in enumerate(class_labels)}
    y_idx_full = np.array([class_to_idx[v] for v in y], dtype=np.int64)

    if patient_id is not None and val_fraction > 0:
        train_idx, val_idx = group_aware_val_split_multiclass(patient_id, y, class_labels, val_fraction, seed)
    else:
        train_idx, val_idx = np.arange(len(y)), np.array([], dtype=int)
    has_val = len(val_idx) > 0 and len(np.unique(y_idx_full[val_idx])) > 1

    def to_tensor(idx):
        x = X[idx]
        if reshape_for_conv:
            x = x[:, None, :]  # (n, 1, n_wave)
        return (torch.tensor(x, dtype=torch.float32), torch.tensor(y_idx_full[idx], dtype=torch.long),
                torch.tensor(sample_weight[idx], dtype=torch.float32))

    X_tr, y_tr, w_tr = to_tensor(train_idx)
    if has_val:
        X_val, y_val, w_val = to_tensor(val_idx)

    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_val_loss, epochs_no_improve = None, np.inf, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(max_epochs):
        module.train()
        optimizer.zero_grad()
        logits = module(X_tr)
        loss = weighted_cross_entropy_loss(logits, y_tr, w_tr)
        loss.backward()
        optimizer.step()
        history["train_loss"].append(float(loss.item()))

        if has_val:
            module.eval()
            with torch.no_grad():
                val_logits = module(X_val)
                val_loss = weighted_cross_entropy_loss(val_logits, y_val, w_val).item()
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss - 1e-5:
                best_val_loss, epochs_no_improve = val_loss, 0
                best_state = {k: v.clone() for k, v in module.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    if has_val and best_state is not None:
        module.load_state_dict(best_state)
        history["stopped_epoch"] = epoch - epochs_no_improve
    else:
        history["stopped_epoch"] = epoch  # no val signal (e.g. degenerate split) -- ran to max_epochs

    return module, history


def predict_proba_torch_multiclass(module: nn.Module, X: np.ndarray, reshape_for_conv: bool = False) -> np.ndarray:
    module.eval()
    x = X[:, None, :] if reshape_for_conv else X
    with torch.no_grad():
        logits = module(torch.tensor(x, dtype=torch.float32))
        prob = torch.softmax(logits, dim=1).numpy()
    return prob
