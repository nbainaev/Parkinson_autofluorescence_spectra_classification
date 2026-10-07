"""Common candidate-model description used by the (classical, for now) model zoo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelCandidate:
    name: str                 # e.g. "logreg_l2_C1.0_A" (unique within a family)
    family: str                # e.g. "logreg" -- groups hyperparameter variants for model-vs-model comparison
    preprocessing_variant: str  # "A" | "B" | "C" | "D" (spectra_ml.preprocessing.build_preprocessing_pipeline)
    params: dict               # model hyperparameters (for logging/reproducibility)
    build_fn: Callable[[], object]  # () -> sklearn-compatible estimator (unfit)
    has_decision_function: bool = False  # True for LinearSVC-like models without predict_proba
    requires_patient_groups: bool = False  # True for torch models that need patient_id at fit() for a
    # group-aware internal early-stopping split (never using outer_test for that split)
