"""Shared scoring helper: apply an already-fitted preprocessing pipeline +
model to raw spectra and return per-spectrum P(Parkinson). Used by
freeze_final_model.py (development OOF / refit sanity checks,
interpretability) and predict_blinded.py (the actual blinded scoring path) so
the exact same code computes a "probability" in both places -- no separate
reimplementation that could silently drift from what nested CV/freeze used.
"""
from __future__ import annotations

import numpy as np


def score_spectra(preprocessing, model, X_raw: np.ndarray, has_decision_function: bool = False) -> np.ndarray:
    Xt = preprocessing.transform(X_raw)
    if has_decision_function:
        raw = model.decision_function(Xt)
        return 1 / (1 + np.exp(-raw))
    return model.predict_proba(Xt)[:, 1]
