"""Stage 13: Integrated Gradients for MLP/CNN1D (build_project_and_exp.md
section 26). Attribution is computed against the base model's LOGIT (not the
calibrated probability -- calibration is a separate monotonic mapping fit
afterwards, section 26), and every baseline is built only from a fold's own
training data, never from test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _forward_logit(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return module(x).squeeze(-1)


def integrated_gradients(module: torch.nn.Module, X: np.ndarray, baseline: np.ndarray, steps: int = 50,
                          reshape_for_conv: bool = False) -> np.ndarray:
    """X: (n_samples, n_wave). baseline: (n_wave,) or (n_samples, n_wave) (broadcastable).
    Returns attribution (n_samples, n_wave), same units as (X - baseline) * d(logit)/dX
    (Riemann-sum approximation of the path integral, section 26)."""
    module.eval()
    n_samples, n_wave = X.shape
    x = torch.tensor(X, dtype=torch.float32)                                    # (n, W)
    b = torch.tensor(np.broadcast_to(baseline, X.shape).copy(), dtype=torch.float32)  # (n, W)
    diff = x - b                                                                 # (n, W)

    alphas = torch.linspace(0, 1, steps).view(steps, 1, 1)                       # (steps, 1, 1)
    interpolated = b.unsqueeze(0) + alphas * diff.unsqueeze(0)                   # (steps, n, W)
    interpolated = interpolated.reshape(steps * n_samples, n_wave)
    if reshape_for_conv:
        interpolated = interpolated.unsqueeze(1)                                 # (steps*n, 1, W)
    interpolated.requires_grad_(True)

    logits = _forward_logit(module, interpolated)                                # (steps*n,)
    grads = torch.autograd.grad(logits.sum(), interpolated)[0]                   # same shape as interpolated
    if reshape_for_conv:
        grads = grads.squeeze(1)
    grads = grads.reshape(steps, n_samples, n_wave).mean(dim=0)                  # avg over path -> (n, W)

    attribution = diff * grads
    return attribution.detach().numpy()


class _SqueezeWrapper(torch.nn.Module):
    """captum's GradientShap expects a model output shaped (batch, ...) it can
    index by `target`; for our single-logit models the cleanest fit is a plain
    (batch,) regression-style output, so squeeze the trailing 1-dim here rather
    than inside the model itself (keeps torch_common's shared trainer untouched)."""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x).squeeze(-1)


def gradient_shap_attribution(module: torch.nn.Module, X: np.ndarray, baseline_samples: np.ndarray,
                               n_samples: int = 50, stdevs: float = 0.1, seed: int = 42,
                               reshape_for_conv: bool = False) -> np.ndarray:
    """GradientSHAP (Erion et al.) as a second, independent attribution method
    to cross-check Integrated Gradients (section 26: "GradientSHAP... как
    sensitivity analysis"). `baseline_samples` should be a SET of reference
    spectra (e.g. a subsample of that fold's own outer_train, in the same
    preprocessed feature space as X) -- GradientSHAP integrates over both
    interpolation and a distribution of baselines/noise, unlike IG's single
    fixed baseline."""
    from captum.attr import GradientShap

    torch.manual_seed(seed)
    wrapped = _SqueezeWrapper(module)
    wrapped.eval()

    x = torch.tensor(X, dtype=torch.float32)
    b = torch.tensor(baseline_samples, dtype=torch.float32)
    if reshape_for_conv:
        x, b = x[:, None, :], b[:, None, :]
    x.requires_grad_(True)

    gs = GradientShap(wrapped)
    attribution = gs.attribute(x, baselines=b, n_samples=n_samples, stdevs=stdevs)
    if reshape_for_conv:
        attribution = attribution.squeeze(1)
    return attribution.detach().numpy()


def compute_baselines(X_train: np.ndarray, n_wave: int) -> dict[str, np.ndarray]:
    return {
        "zero": np.zeros(n_wave),
        "train_mean": X_train.mean(axis=0),
        "train_median": np.median(X_train, axis=0),
    }


def aggregate_attribution_to_patient(attribution: np.ndarray, patient_id: np.ndarray) -> pd.DataFrame:
    """One row per patient: mean and mean(|attribution|) across that patient's
    spectra, per wavelength -- returned as two (n_patients, n_wave) arrays via a
    tidy long DataFrame for downstream aggregation."""
    df = pd.DataFrame(attribution)
    df["patient_id"] = patient_id
    mean_df = df.groupby("patient_id").mean()
    abs_df = pd.DataFrame(np.abs(attribution))
    abs_df["patient_id"] = patient_id
    abs_mean_df = abs_df.groupby("patient_id").mean()
    return mean_df, abs_mean_df


def global_attribution_summary(patient_mean_attr: np.ndarray, patient_abs_mean_attr: np.ndarray,
                                waves: np.ndarray) -> pd.DataFrame:
    """patient_mean_attr, patient_abs_mean_attr: (n_patients, n_wave) -- across
    outer-test patients (one fold) or pooled across all outer folds."""
    return pd.DataFrame({
        "wavelength": waves,
        "mean_attribution": patient_mean_attr.mean(axis=0),
        "mean_abs_attribution": patient_abs_mean_attr.mean(axis=0),
        "median_abs_attribution": np.median(patient_abs_mean_attr, axis=0),
    })
