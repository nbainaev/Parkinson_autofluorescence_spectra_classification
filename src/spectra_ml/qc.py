"""Stage 2: automatic, target-blind spectrum QC (build_project_and_exp.md section 3-5).

All thresholds here are fixed engineering rules read from configs/qc.yaml, not
fit from the dataset distribution, so they can be applied directly to any
spectrum (train, val, test, or a full-dataset EDA pass) without a leakage
concern -- except `ShapeOutlierDetector`, which is explicitly fit/transform
and must only ever be `.fit()` on a training fold in the ML pipeline (a
full-dataset fit is allowed for EDA only, per section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

PASS, SOFT_WARNING, HARD_FAIL = "PASS", "SOFT_WARNING", "HARD_FAIL"
_EPS = 1e-12


@dataclass(frozen=True)
class QCConfig:
    hot_pixel_half_window: int
    hot_pixel_mad_threshold: float
    hot_pixel_max_width: int
    hot_pixel_max_count: int
    systematic_channel_fraction: float
    interpolate_hot_pixels: bool
    negative_fraction_hard_fail: float
    negative_area_fraction_hard_fail: float
    savgol_window: int
    savgol_polyorder: int
    snr_hard_fail: float
    snr_soft_warning: float
    saturation_near_max_rel: float
    saturation_plateau_run_hard_fail: int
    flat_dynamic_range_rel_hard_fail: float
    duplicate_value_fraction_hard_fail: float
    shape_outlier_enabled: bool
    shape_outlier_contamination: float
    random_seed: int

    @classmethod
    def from_yaml(cls, path: str) -> "QCConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        qc = raw["qc"]
        return cls(random_seed=int(raw["random_seed"]), **qc)


def _robust_std(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, length)."""
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


@dataclass
class HotPixelResult:
    candidate_mask: np.ndarray       # (n_spectra, n_wave) bool, all spike candidates (any width)
    correctable_mask: np.ndarray     # bool, candidates with run width <= max_width
    wide_artifact_mask: np.ndarray   # bool, candidates with run width > max_width
    systematic_channel: np.ndarray   # (n_wave,) bool, columns flagged in > systematic_channel_fraction of spectra
    hot_pixel_count: np.ndarray      # (n_spectra,) int, correctable & non-systematic spikes per spectrum


def _sliding_windows_excl_center(s: np.ndarray, hw: int) -> np.ndarray:
    """(n_wave, 2*hw) matrix: row j holds up to 2*hw neighbors of j (edge-padded
    with NaN), excluding s[j] itself."""
    n = len(s)
    padded = np.pad(s, hw, mode="constant", constant_values=np.nan)
    win_size = 2 * hw + 1
    windows = np.lib.stride_tricks.sliding_window_view(padded, win_size)  # (n, win_size)
    return np.delete(windows, hw, axis=1)  # drop center column -> (n, win_size-1)


def detect_hot_pixels(spectra: np.ndarray, config: QCConfig) -> HotPixelResult:
    n_spectra, n_wave = spectra.shape
    hw = config.hot_pixel_half_window
    candidate = np.zeros_like(spectra, dtype=bool)

    for i in range(n_spectra):
        s = spectra[i]
        if np.all(np.isnan(s)):
            continue
        neighbors = _sliding_windows_excl_center(s, hw)  # (n_wave, 2*hw)
        with np.errstate(invalid="ignore"):
            m = np.nanmedian(neighbors, axis=1)
            mad = 1.4826 * np.nanmedian(np.abs(neighbors - m[:, None]), axis=1)
        z = np.abs(s - m) / (mad + _EPS)
        row_candidate = (z > config.hot_pixel_mad_threshold) & ~np.isnan(s) & ~np.isnan(m)
        candidate[i] = row_candidate

    # systematic channel: flagged across many spectra -> instrument defect, not per-spectrum artifact
    frac_flagged = candidate.mean(axis=0)
    systematic_channel = frac_flagged > config.systematic_channel_fraction

    correctable = np.zeros_like(candidate)
    wide = np.zeros_like(candidate)
    hot_pixel_count = np.zeros(n_spectra, dtype=int)
    for i in range(n_spectra):
        row = candidate[i] & ~systematic_channel
        for start, length in _runs(row):
            if length <= config.hot_pixel_max_width:
                correctable[i, start:start + length] = True
                hot_pixel_count[i] += 1
            else:
                wide[i, start:start + length] = True

    return HotPixelResult(candidate, correctable, wide, systematic_channel, hot_pixel_count)


def correct_hot_pixels(spectra: np.ndarray, hp: HotPixelResult, config: QCConfig) -> tuple[np.ndarray, list[dict]]:
    """Linear interpolation of correctable hot pixels from immediate non-flagged neighbors.
    Returns corrected spectra and a log of every point changed."""
    corrected = spectra.copy()
    log = []
    if not config.interpolate_hot_pixels:
        return corrected, log
    n_spectra, n_wave = spectra.shape
    x = np.arange(n_wave)
    for i in range(n_spectra):
        bad = hp.correctable_mask[i]
        if not bad.any():
            continue
        good = ~bad & ~np.isnan(spectra[i])
        if good.sum() < 2:
            continue
        before = corrected[i, bad].copy()
        corrected[i, bad] = np.interp(x[bad], x[good], spectra[i, good])
        for idx, old_val, new_val in zip(np.where(bad)[0], before, corrected[i, bad]):
            log.append({"spectrum_index": i, "wave_index": int(idx), "old_value": float(old_val), "new_value": float(new_val)})
    return corrected, log


def _spectrum_features(s: np.ndarray, waves: np.ndarray, config: QCConfig) -> dict:
    valid = ~np.isnan(s)
    if valid.sum() < max(config.savgol_window, 5):
        return dict(negative_fraction=np.nan, minimum_intensity=np.nan, negative_area_fraction=np.nan,
                     snr_score=np.nan, integrated_intensity=np.nan, dynamic_range=np.nan,
                     saturation_score=np.nan, roughness_score=np.nan, median_intensity=np.nan,
                     n_nan=int((~valid).sum()))

    sv, wv = s[valid], waves[valid]
    n_nan = int((~valid).sum())

    negative_fraction = float((sv < 0).mean())
    minimum_intensity = float(sv.min())
    pos_area = np.trapezoid(np.clip(sv, 0, None), wv)
    neg_area = np.trapezoid(np.clip(-sv, 0, None), wv)
    negative_area_fraction = float(neg_area / (pos_area + neg_area + _EPS))

    window = min(config.savgol_window, len(sv) - (1 - len(sv) % 2))
    window = window if window % 2 == 1 else window - 1
    window = max(window, config.savgol_polyorder + 2 + (config.savgol_polyorder % 2 == 0))
    if window >= len(sv):
        window = len(sv) - 1 if (len(sv) - 1) % 2 == 1 else len(sv) - 2
    if window <= config.savgol_polyorder or window < 3:
        smoothed = sv
        noise = _robust_std(np.diff(sv))
    else:
        smoothed = savgol_filter(sv, window_length=window, polyorder=config.savgol_polyorder)
        noise = _robust_std(sv - smoothed)
    signal = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5))
    snr_score = float(signal / (noise + _EPS))
    integrated_intensity = float(np.trapezoid(sv, wv))
    dynamic_range = float(np.percentile(sv, 95) - np.percentile(sv, 5))

    # Saturation/clipping should look like a FLAT plateau at the detector ceiling, not
    # merely "near the peak" -- a smooth fluorescence maximum is naturally broad and
    # would otherwise be mistaken for clipping. Require points within a tight band of
    # the maximum AND a long contiguous run to flag it.
    rng = sv.max() - sv.min() + _EPS
    near_max = sv >= (sv.max() - config.saturation_near_max_rel * rng)
    max_run = max((length for _, length in _runs(near_max)), default=0)
    saturation_score = float(max_run / config.saturation_plateau_run_hard_fail)

    roughness_score = float(_robust_std(np.diff(sv, n=2)))

    return dict(negative_fraction=negative_fraction, minimum_intensity=minimum_intensity,
                negative_area_fraction=negative_area_fraction, snr_score=snr_score,
                integrated_intensity=integrated_intensity, dynamic_range=dynamic_range,
                saturation_score=saturation_score, roughness_score=roughness_score,
                median_intensity=float(np.median(sv)), n_nan=n_nan)


def _status_for_row(feat: dict, hot_pixel_count: int, wide_artifact: bool, config: QCConfig) -> tuple[str, list[str]]:
    reasons = []
    if np.isnan(feat["snr_score"]) or feat["n_nan"] > 0:
        reasons.append("insufficient_valid_points" if np.isnan(feat["snr_score"]) else "contains_nan")
    if wide_artifact:
        reasons.append("wide_spectral_artifact")
    if hot_pixel_count > config.hot_pixel_max_count:
        reasons.append("too_many_hot_pixels")
    if not np.isnan(feat["negative_fraction"]):
        if feat["negative_fraction"] > config.negative_fraction_hard_fail:
            reasons.append("excess_negative_fraction")
        if feat["negative_area_fraction"] > config.negative_area_fraction_hard_fail:
            reasons.append("excess_negative_area")
    if not np.isnan(feat.get("snr_score", np.nan)) and feat["snr_score"] < config.snr_hard_fail:
        reasons.append("low_snr")
    if not np.isnan(feat.get("saturation_score", np.nan)) and feat["saturation_score"] >= 1.0:
        reasons.append("saturation")
    if not np.isnan(feat.get("dynamic_range", np.nan)) and not np.isnan(feat.get("median_intensity", np.nan)):
        level = abs(feat["median_intensity"]) + _EPS
        if (feat["dynamic_range"] / level) < config.flat_dynamic_range_rel_hard_fail:
            reasons.append("flat_spectrum")

    if reasons:
        return HARD_FAIL, reasons

    soft = []
    if not np.isnan(feat.get("snr_score", np.nan)) and feat["snr_score"] < config.snr_soft_warning:
        soft.append("marginal_snr")
    if hot_pixel_count > 0:
        soft.append("corrected_hot_pixels")
    return (SOFT_WARNING, soft) if soft else (PASS, [])


def build_qc_table(spectra: np.ndarray, waves: np.ndarray, patient_id: np.ndarray,
                    spectrum_id: np.ndarray, config: QCConfig) -> tuple[pd.DataFrame, HotPixelResult, list[dict]]:
    """Full Stage-2 QC pass. `spectra` should be the RAW (pre-correction) array;
    hot-pixel correction is computed and logged here, and the returned
    corrected array should be used downstream if config.interpolate_hot_pixels."""
    hp = detect_hot_pixels(spectra, config)
    corrected, correction_log = correct_hot_pixels(spectra, hp, config)

    rows = []
    for i in range(spectra.shape[0]):
        feat = _spectrum_features(corrected[i], waves, config)
        wide_artifact = bool(hp.wide_artifact_mask[i].any())
        status, reasons = _status_for_row(feat, int(hp.hot_pixel_count[i]), wide_artifact, config)
        rows.append({
            "spectrum_id": spectrum_id[i],
            "patient_id": patient_id[i],
            "qc_status": status,
            "hot_pixel_count": int(hp.hot_pixel_count[i]),
            "negative_fraction": feat["negative_fraction"],
            "saturation_score": feat["saturation_score"],
            "snr_score": feat["snr_score"],
            "roughness_score": feat["roughness_score"],
            "integrated_intensity": feat["integrated_intensity"],
            "dynamic_range": feat["dynamic_range"],
            "reason_codes": ";".join(reasons),
        })
    return pd.DataFrame(rows), hp, correction_log


class ShapeOutlierDetector:
    """Unsupervised full-spectrum-shape outlier flag (SOFT_WARNING only, never
    auto-removes). Must be .fit() on TRAINING data only when used inside CV;
    a full-dataset fit is permitted for EDA only (build_project_and_exp.md section 5)."""

    def __init__(self, contamination: float, random_seed: int):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(contamination=contamination, random_state=random_seed)
        self._fitted = False

    def fit(self, spectra: np.ndarray) -> "ShapeOutlierDetector":
        x = _snv(spectra)
        x = np.nan_to_num(x, nan=0.0)
        self.model.fit(x)
        self._fitted = True
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ShapeOutlierDetector must be fit before transform.")
        x = _snv(spectra)
        x = np.nan_to_num(x, nan=0.0)
        return self.model.predict(x) == -1  # True = flagged outlier


def _snv(spectra: np.ndarray) -> np.ndarray:
    mean = np.nanmean(spectra, axis=1, keepdims=True)
    std = np.nanstd(spectra, axis=1, keepdims=True)
    return (spectra - mean) / (std + _EPS)
