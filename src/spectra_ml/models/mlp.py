"""Stage 11: configurable MLP (build_project_and_exp.md section 14)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from spectra_ml.models.base import ModelCandidate
from spectra_ml.models.torch_common import (predict_proba_torch, predict_proba_torch_multiclass,
                                             train_torch_binary_classifier, train_torch_multiclass_classifier)

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def _build_mlp(n_features: int, hidden_dims: list[int], activation: str, dropout: float,
               batch_norm: bool) -> nn.Module:
    act_cls = _ACTIVATIONS[activation]
    layers, in_dim = [], n_features
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))  # single logit (section 14)
    return nn.Sequential(*layers)


class MLPBinaryClassifier:
    """sklearn-like interface (fit/predict/predict_proba/decision_function/
    get_params/set_params/save/load), PyTorch-backed (section 16)."""

    def __init__(self, hidden_dims=(64, 32), activation: str = "relu", dropout: float = 0.2,
                 batch_norm: bool = False, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs: int = 300, patience: int = 20, val_fraction: float = 0.15, random_state: int = 42):
        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.module_ = None
        self.history_ = None
        self.n_features_in_ = None

    def get_params(self, deep=True):
        return {"hidden_dims": self.hidden_dims, "activation": self.activation, "dropout": self.dropout,
                "batch_norm": self.batch_norm, "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay, "max_epochs": self.max_epochs, "patience": self.patience,
                "val_fraction": self.val_fraction, "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
            patient_id: np.ndarray | None = None) -> "MLPBinaryClassifier":
        self.n_features_in_ = X.shape[1]
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        module = _build_mlp(self.n_features_in_, self.hidden_dims, self.activation, self.dropout, self.batch_norm)
        self.module_, self.history_ = train_torch_binary_classifier(
            module, X, y, sample_weight, patient_id, lr=self.learning_rate, weight_decay=self.weight_decay,
            max_epochs=self.max_epochs, patience=self.patience, val_fraction=self.val_fraction,
            seed=self.random_state,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.module_ is None:
            raise RuntimeError("MLPBinaryClassifier must be fit before predict_proba.")
        return predict_proba_torch(self.module_, X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.module_.state_dict(), "params": self.get_params(),
                    "n_features_in_": self.n_features_in_}, path)

    @classmethod
    def load(cls, path: str) -> "MLPBinaryClassifier":
        ckpt = torch.load(path, weights_only=False)
        obj = cls(**ckpt["params"])
        obj.n_features_in_ = ckpt["n_features_in_"]
        obj.module_ = _build_mlp(obj.n_features_in_, obj.hidden_dims, obj.activation, obj.dropout, obj.batch_norm)
        obj.module_.load_state_dict(ckpt["state_dict"])
        return obj


def _build_mlp_multiclass(n_features: int, n_classes: int, hidden_dims: list[int], activation: str,
                           dropout: float, batch_norm: bool) -> nn.Module:
    """Multiclass sibling of _build_mlp: identical hidden-layer stack, final
    nn.Linear(in_dim, n_classes) with NO activation (nn.CrossEntropyLoss
    expects raw logits, not softmax-ed probabilities)."""
    act_cls = _ACTIVATIONS[activation]
    layers, in_dim = [], n_features
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, n_classes))
    return nn.Sequential(*layers)


class MLPMulticlassClassifier:
    """Multiclass sibling of MLPBinaryClassifier (does NOT modify/replace it,
    see module note above). Operates on a PCA(n=10)-reduced input space
    (`reducer_kind`/`n_components`, matching the other reduced-nonlinear
    multiclass candidates in spectra_ml.models.classical): the reducer is fit
    INSIDE this class's own .fit() (never in an external sklearn Pipeline),
    the same convention already used by GroupAwareCalibratedLinearSVC, because
    this class's .fit() takes a `patient_id=` kwarg for its own group-aware
    early-stopping split (requires_patient_groups=True routing in
    spectra_ml.experiment._fit_with_sample_weight calls
    `model.fit(X, y, sample_weight=..., patient_id=...)` directly -- a bare
    sklearn Pipeline wrapping this class would not accept those two kwargs)."""

    def __init__(self, hidden_dims=(64, 16), activation: str = "relu", dropout: float = 0.3,
                 batch_norm: bool = False, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs: int = 150, patience: int = 15, val_fraction: float = 0.15,
                 reducer_kind: str | None = "pca", n_components: int = 10, random_state: int = 42):
        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.reducer_kind = reducer_kind
        self.n_components = n_components
        self.random_state = random_state
        self.reducer_ = None
        self.module_ = None
        self.history_ = None
        self.n_features_in_ = None
        self.classes_ = None

    def get_params(self, deep=True):
        return {"hidden_dims": self.hidden_dims, "activation": self.activation, "dropout": self.dropout,
                "batch_norm": self.batch_norm, "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay, "max_epochs": self.max_epochs, "patience": self.patience,
                "val_fraction": self.val_fraction, "reducer_kind": self.reducer_kind,
                "n_components": self.n_components, "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _reduce_fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.reducer_kind is None:
            self.reducer_ = None
            return X
        from spectra_ml.models.classical import _make_reducer

        self.reducer_ = _make_reducer(self.reducer_kind, self.n_components)
        self.reducer_.fit(X, y)
        return self.reducer_.transform(X)

    def _reduce_transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer_.transform(X) if self.reducer_ is not None else X

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
            patient_id: np.ndarray | None = None) -> "MLPMulticlassClassifier":
        self.classes_ = sorted(np.unique(y).tolist())
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        Xt = self._reduce_fit(X, y)
        self.n_features_in_ = Xt.shape[1]
        module = _build_mlp_multiclass(self.n_features_in_, len(self.classes_), self.hidden_dims,
                                        self.activation, self.dropout, self.batch_norm)
        self.module_, self.history_ = train_torch_multiclass_classifier(
            module, Xt, y, sample_weight, patient_id, self.classes_, lr=self.learning_rate,
            weight_decay=self.weight_decay, max_epochs=self.max_epochs, patience=self.patience,
            val_fraction=self.val_fraction, seed=self.random_state,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.module_ is None:
            raise RuntimeError("MLPMulticlassClassifier must be fit before predict_proba.")
        return predict_proba_torch_multiclass(self.module_, self._reduce_transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes_[i] for i in idx])


def build_mlp_multiclass_candidates(preprocessing_variants=("D",), seed: int = 42) -> list[ModelCandidate]:
    """ONE fixed candidate (no grid, per runtime discipline): hidden_dims=
    [64, 16], dropout=0.3, PCA(n=10)-reduced input, max_epochs=150 with
    group-aware early stopping, lr=1e-3/Adam. requires_patient_groups=True
    (same convention as the existing binary mlp/cnn1d candidates and
    linear_svm_mc_reduced -- see MLPMulticlassClassifier docstring)."""
    out = []
    for variant in preprocessing_variants:
        params = {"hidden_dims": [64, 16], "dropout": 0.3, "max_epochs": 150, "patience": 15,
                  "learning_rate": 1e-3, "n_components": 10}
        out.append(ModelCandidate(
            f"mlp_multiclass_pca10_{variant}", "mlp_multiclass", variant, params,
            lambda params=params: MLPMulticlassClassifier(
                random_state=seed, hidden_dims=params["hidden_dims"], dropout=params["dropout"],
                max_epochs=params["max_epochs"], patience=params["patience"],
                learning_rate=params["learning_rate"], reducer_kind="pca", n_components=params["n_components"]),
            requires_patient_groups=True,
        ))
    return out


def build_mlp_candidates(preprocessing_variants=("A",), seed: int = 42) -> list[ModelCandidate]:
    # Deliberately small search space (section 14 lists up to 3 hidden layers as a
    # ceiling, not a target) given ~140 patients total -- runtime and overfitting
    # risk both argue against a wide grid here.
    hidden_dims_grid = [(64,), (64, 32)]
    out = []
    for variant in preprocessing_variants:
        for hidden_dims in hidden_dims_grid:
            params = {"hidden_dims": hidden_dims, "dropout": 0.3, "weight_decay": 1e-3}
            out.append(ModelCandidate(
                f"mlp_{'x'.join(map(str, hidden_dims))}_{variant}", "mlp", variant, params,
                lambda params=params: MLPBinaryClassifier(random_state=seed, **params),
                requires_patient_groups=True,
            ))
    return out
