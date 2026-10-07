"""Stage 12: configurable 1D CNN (build_project_and_exp.md section 15).

Wavelength grid uniformity check (section 15: "если сетка неравномерная --
реализовать interpolation на uniform grid"): the actual grid used throughout
this pipeline (data/patients_database.xlsx, 401.8-681.9 nm, 278 points) has
step 1.011 +/- 0.010 nm (~1% relative variation) -- close enough to uniform
that no extra interpolation transformer is used; Conv1D treats the sequence
as evenly spaced.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from spectra_ml.models.base import ModelCandidate
from spectra_ml.models.torch_common import (predict_proba_torch, predict_proba_torch_multiclass,
                                             train_torch_binary_classifier, train_torch_multiclass_classifier)

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def _build_cnn1d(n_wave: int, channels: list[int], kernel_sizes: list[int], dropout: float,
                  pool_size: int, activation: str, dense_dim: int) -> nn.Module:
    assert len(channels) == len(kernel_sizes), "channels and kernel_sizes must be the same length"
    act_cls = _ACTIVATIONS[activation]

    conv_layers, in_ch = [], 1
    for out_ch, k in zip(channels, kernel_sizes):
        conv_layers += [
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
            act_cls(),
            nn.MaxPool1d(pool_size),
        ]
        in_ch = out_ch
    conv_layers.append(nn.AdaptiveAvgPool1d(1))
    conv = nn.Sequential(*conv_layers)

    head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_ch, dense_dim),
        act_cls(),
        nn.Dropout(dropout),
        nn.Linear(dense_dim, 1),
    )
    return nn.Sequential(conv, head)


class CNN1DBinaryClassifier:
    """Same sklearn-like interface as MLPBinaryClassifier (section 16)."""

    def __init__(self, channels=(16, 32), kernel_sizes=(5, 5), dropout: float = 0.2, pool_size: int = 2,
                 activation: str = "relu", dense_dim: int = 32, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, max_epochs: int = 300, patience: int = 20,
                 val_fraction: float = 0.15, random_state: int = 42):
        self.channels = list(channels)
        self.kernel_sizes = list(kernel_sizes)
        self.dropout = dropout
        self.pool_size = pool_size
        self.activation = activation
        self.dense_dim = dense_dim
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
        return {"channels": self.channels, "kernel_sizes": self.kernel_sizes, "dropout": self.dropout,
                "pool_size": self.pool_size, "activation": self.activation, "dense_dim": self.dense_dim,
                "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs, "patience": self.patience, "val_fraction": self.val_fraction,
                "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
            patient_id: np.ndarray | None = None) -> "CNN1DBinaryClassifier":
        self.n_features_in_ = X.shape[1]
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        module = _build_cnn1d(self.n_features_in_, self.channels, self.kernel_sizes, self.dropout,
                               self.pool_size, self.activation, self.dense_dim)
        self.module_, self.history_ = train_torch_binary_classifier(
            module, X, y, sample_weight, patient_id, lr=self.learning_rate, weight_decay=self.weight_decay,
            max_epochs=self.max_epochs, patience=self.patience, val_fraction=self.val_fraction,
            seed=self.random_state, reshape_for_conv=True,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.module_ is None:
            raise RuntimeError("CNN1DBinaryClassifier must be fit before predict_proba.")
        return predict_proba_torch(self.module_, X, reshape_for_conv=True)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.module_.state_dict(), "params": self.get_params(),
                    "n_features_in_": self.n_features_in_}, path)

    @classmethod
    def load(cls, path: str) -> "CNN1DBinaryClassifier":
        ckpt = torch.load(path, weights_only=False)
        obj = cls(**ckpt["params"])
        obj.n_features_in_ = ckpt["n_features_in_"]
        obj.module_ = _build_cnn1d(obj.n_features_in_, obj.channels, obj.kernel_sizes, obj.dropout,
                                    obj.pool_size, obj.activation, obj.dense_dim)
        obj.module_.load_state_dict(ckpt["state_dict"])
        return obj


def build_cnn1d_candidates(preprocessing_variants=("A",), seed: int = 42) -> list[ModelCandidate]:
    # 2 conv blocks max (section 15: "не делать глубокую CNN"), small kernel-size grid.
    configs = [
        {"channels": (16, 32), "kernel_sizes": (5, 5)},
        {"channels": (16, 32), "kernel_sizes": (11, 11)},
    ]
    out = []
    for variant in preprocessing_variants:
        for cfg in configs:
            params = {**cfg, "dropout": 0.3, "weight_decay": 1e-3}
            out.append(ModelCandidate(
                f"cnn1d_k{cfg['kernel_sizes'][0]}_{variant}", "cnn1d", variant, params,
                lambda params=params: CNN1DBinaryClassifier(random_state=seed, **params),
                requires_patient_groups=True,
            ))
    return out


def _build_cnn1d_multiclass(n_wave: int, n_classes: int, channels: list[int], kernel_sizes: list[int],
                             dropout: float, pool_size: int, activation: str, dense_dim: int) -> nn.Module:
    """Multiclass sibling of _build_cnn1d: identical conv/pool stack, final
    head layer widened to n_classes raw logits (no softmax -- CrossEntropyLoss
    expects raw logits)."""
    assert len(channels) == len(kernel_sizes), "channels and kernel_sizes must be the same length"
    act_cls = _ACTIVATIONS[activation]

    conv_layers, in_ch = [], 1
    for out_ch, k in zip(channels, kernel_sizes):
        conv_layers += [
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
            act_cls(),
            nn.MaxPool1d(pool_size),
        ]
        in_ch = out_ch
    conv_layers.append(nn.AdaptiveAvgPool1d(1))
    conv = nn.Sequential(*conv_layers)

    head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_ch, dense_dim),
        act_cls(),
        nn.Dropout(dropout),
        nn.Linear(dense_dim, n_classes),
    )
    return nn.Sequential(conv, head)


class CNN1DMulticlassClassifier:
    """Multiclass sibling of CNN1DBinaryClassifier (does NOT modify/replace
    it, see module note above). Operates on the RAW area-normalized
    278-wavelength sequence (no PCA reduction -- a 1D conv over PCA
    components loses the contiguous-wavelength structure a conv kernel is
    meant to exploit), same sklearn-like interface convention as the binary
    class (fit/predict/predict_proba/get_params/set_params/save/load)."""

    def __init__(self, channels=(16, 32), kernel_sizes=(7, 5), dropout: float = 0.3, pool_size: int = 2,
                 activation: str = "relu", dense_dim: int = 32, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-3, max_epochs: int = 150, patience: int = 15,
                 val_fraction: float = 0.15, random_state: int = 42):
        self.channels = list(channels)
        self.kernel_sizes = list(kernel_sizes)
        self.dropout = dropout
        self.pool_size = pool_size
        self.activation = activation
        self.dense_dim = dense_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.module_ = None
        self.history_ = None
        self.n_features_in_ = None
        self.classes_ = None

    def get_params(self, deep=True):
        return {"channels": self.channels, "kernel_sizes": self.kernel_sizes, "dropout": self.dropout,
                "pool_size": self.pool_size, "activation": self.activation, "dense_dim": self.dense_dim,
                "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs, "patience": self.patience, "val_fraction": self.val_fraction,
                "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
            patient_id: np.ndarray | None = None) -> "CNN1DMulticlassClassifier":
        self.classes_ = sorted(np.unique(y).tolist())
        self.n_features_in_ = X.shape[1]
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        module = _build_cnn1d_multiclass(self.n_features_in_, len(self.classes_), self.channels,
                                          self.kernel_sizes, self.dropout, self.pool_size, self.activation,
                                          self.dense_dim)
        self.module_, self.history_ = train_torch_multiclass_classifier(
            module, X, y, sample_weight, patient_id, self.classes_, lr=self.learning_rate,
            weight_decay=self.weight_decay, max_epochs=self.max_epochs, patience=self.patience,
            val_fraction=self.val_fraction, seed=self.random_state, reshape_for_conv=True,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.module_ is None:
            raise RuntimeError("CNN1DMulticlassClassifier must be fit before predict_proba.")
        return predict_proba_torch_multiclass(self.module_, X, reshape_for_conv=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes_[i] for i in idx])


def build_cnn1d_multiclass_candidates(preprocessing_variants=("D",), seed: int = 42) -> list[ModelCandidate]:
    """ONE fixed candidate (no grid, per runtime discipline): 2 conv blocks
    [16, 32] channels, kernel sizes [7, 5], global average pool, dense head to
    n_classes, on the RAW area-normalized 278-wavelength input (variant "D",
    NOT PCA-reduced -- see class docstring). max_epochs=150 with group-aware
    early stopping, lr=1e-3/Adam. requires_patient_groups=True (same
    convention as build_cnn1d_candidates / build_mlp_multiclass_candidates)."""
    out = []
    for variant in preprocessing_variants:
        params = {"channels": (16, 32), "kernel_sizes": (7, 5), "dropout": 0.3, "max_epochs": 150,
                  "patience": 15, "learning_rate": 1e-3}
        out.append(ModelCandidate(
            f"cnn1d_multiclass_k7-5_{variant}", "cnn1d_multiclass", variant, params,
            lambda params=params: CNN1DMulticlassClassifier(
                random_state=seed, channels=params["channels"], kernel_sizes=params["kernel_sizes"],
                dropout=params["dropout"], max_epochs=params["max_epochs"], patience=params["patience"],
                learning_rate=params["learning_rate"]),
            requires_patient_groups=True,
        ))
    return out
