"""Phase 1 (current_tasks/double_blind_exp.md Parts I, IV-XVI): instrument
comparison and empirical mean-spectrum instrument harmonization.

Terminology (Part XXXVII): the wavelength-dependent coefficients built here are
*empirical mean-spectrum instrument harmonization* / *empirical
wavelength-dependent transfer coefficients* -- NOT a physical intensity
calibration-transfer function (the same physical sample was never measured on
both instruments).

Scope: Phase 1 only. Blinding/freeze/final-model machinery (Parts II-III,
XVII+) is explicitly out of scope here; where the spec assumes a
development/blind-test split that does not exist yet in this phase, the
functions below operate on the full available cohort and the caller
(scripts/build_transfer_functions.py) is responsible for stamping the
resulting artifacts with a "full-cohort, not dev-only" caveat.

No function in this module ever branches on `target` to select an
instrument-specific transformation -- transfer curves are indexed only by
which GROUP (old/new_pd/new_control x class) was used to *build* them; once
built, a curve is applied uniformly to every spectrum of its owning
instrument (Part X/XXXIV).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

from spectra_ml.statistics import PatientRepr, compute_patient_representative_spectra

_EPS = 1e-12

PASS, WARNING, FAIL = "PASS", "WARNING", "FAIL"

STOP_TRANSFER_MESSAGE = (
    "New and OLD spectra show evidence of wavelength-axis misalignment; multiplicative "
    "transfer coefficients cannot reliably correct this effect."
)
PD_CONTROL_DIVERGENCE_WARNING = (
    "Estimated transfer function depends strongly on the clinical subgroup and therefore "
    "cannot be interpreted as a pure instrument response correction."
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentRoleConfig:
    config: str
    role: str


@dataclass(frozen=True)
class WavelengthAlignmentConfig:
    shift_search_range_nm: tuple[float, float]
    shift_search_step_nm: float
    max_allowed_shift_nm: float
    max_allowed_shift_bins: float
    n_bootstrap: int
    bootstrap_seed: int


@dataclass(frozen=True)
class SavgolConfig:
    window: int
    polyorder: int


@dataclass(frozen=True)
class TransferSmoothingConfig:
    method: str
    window: int
    polyorder: int
    denom_near_zero_rel_threshold: float
    edge_exclude_n_points: int


@dataclass(frozen=True)
class TransferDivergenceConfig:
    min_correlation: float
    max_spectral_angle_rad: float


@dataclass(frozen=True)
class HarmonizationConfig:
    instruments: dict           # role_key -> InstrumentRoleConfig
    wavelength_alignment: WavelengthAlignmentConfig
    savgol: SavgolConfig
    transfer_smoothing: TransferSmoothingConfig
    transfer_pd_control_divergence: TransferDivergenceConfig
    random_seed: int

    @classmethod
    def from_yaml(cls, path: str) -> "HarmonizationConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        instruments = {k: InstrumentRoleConfig(**v) for k, v in raw["instruments"].items()}
        wa = raw["wavelength_alignment"]
        wave_align = WavelengthAlignmentConfig(
            shift_search_range_nm=tuple(wa["shift_search_range_nm"]),
            shift_search_step_nm=float(wa["shift_search_step_nm"]),
            max_allowed_shift_nm=float(wa["max_allowed_shift_nm"]),
            max_allowed_shift_bins=float(wa["max_allowed_shift_bins"]),
            n_bootstrap=int(wa["n_bootstrap"]),
            bootstrap_seed=int(wa["bootstrap_seed"]),
        )
        sg = raw["savgol"]
        savgol = SavgolConfig(window=int(sg["window"]), polyorder=int(sg["polyorder"]))
        ts = raw["transfer_smoothing"]
        transfer_smoothing = TransferSmoothingConfig(
            method=ts["method"], window=int(ts["window"]), polyorder=int(ts["polyorder"]),
            denom_near_zero_rel_threshold=float(ts["denom_near_zero_rel_threshold"]),
            edge_exclude_n_points=int(ts["edge_exclude_n_points"]),
        )
        tpd = raw["transfer_pd_control_divergence"]
        divergence = TransferDivergenceConfig(
            min_correlation=float(tpd["min_correlation"]),
            max_spectral_angle_rad=float(tpd["max_spectral_angle_rad"]),
        )
        return cls(instruments=instruments, wavelength_alignment=wave_align, savgol=savgol,
                    transfer_smoothing=transfer_smoothing, transfer_pd_control_divergence=divergence,
                    random_seed=int(raw["random_seed"]))


# ---------------------------------------------------------------------------
# Part IV: reference wavelength grid / reindexing
# ---------------------------------------------------------------------------

def reindex_to_reference_grid(spectra: np.ndarray, waves: np.ndarray, ref_waves: np.ndarray,
                                atol: float = 1e-4) -> np.ndarray:
    """Linear interpolation of `spectra` (n_spectra, n_wave) sampled on `waves`
    onto `ref_waves` (the OLD instrument's grid). If the grids already match
    (same length, all values within `atol`), returns `spectra` unchanged --
    NEVER re-interpolate already-interpolated data (Part IV).

    Raises if `ref_waves` is not fully covered by `waves` (Part IV: "новые
    спектры сначала ограничить диапазоном, который полностью покрывает OLD
    spectrum").
    """
    waves = np.asarray(waves, dtype=float)
    ref_waves = np.asarray(ref_waves, dtype=float)
    if len(waves) == len(ref_waves) and np.allclose(waves, ref_waves, atol=atol):
        return np.asarray(spectra, dtype=float)

    if ref_waves.min() < waves.min() - atol or ref_waves.max() > waves.max() + atol:
        raise ValueError(
            f"reference grid [{ref_waves.min():.3f}, {ref_waves.max():.3f}] nm is not fully covered by "
            f"the source grid [{waves.min():.3f}, {waves.max():.3f}] nm -- cannot interpolate without "
            "extrapolation."
        )
    spectra = np.asarray(spectra, dtype=float)
    out = np.empty((spectra.shape[0], len(ref_waves)), dtype=float)
    for i in range(spectra.shape[0]):
        out[i] = np.interp(ref_waves, waves, spectra[i])
    return out


# ---------------------------------------------------------------------------
# Part VI: area normalization + patient-level representative spectra
# ---------------------------------------------------------------------------

def area_normalize(spectra: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """x_norm(lambda) = x(lambda) / integral(x(lambda) dlambda), trapezoidal,
    over the real (non-uniform-safe) wavelength axis. Reference range = full
    `waves` range (Part VI). Rows with near-zero integrated area are left as
    NaN (flagged, not silently divided) -- caller should already have dropped
    HARD_FAIL/all-NaN rows via QC before this point."""
    spectra = np.asarray(spectra, dtype=float)
    area = np.trapezoid(spectra, waves, axis=1)
    bad = np.abs(area) < _EPS
    out = np.full_like(spectra, np.nan)
    safe_area = np.where(bad, 1.0, area)
    out = spectra / safe_area[:, None]
    out[bad] = np.nan
    return out


def build_patient_area_normalized_repr(spectra: np.ndarray, patient_id: np.ndarray, target: np.ndarray,
                                        waves: np.ndarray, qc_status: np.ndarray | None,
                                        exclude_statuses=("HARD_FAIL", "SOFT_WARNING"),
                                        method: str = "median") -> PatientRepr:
    """Part VI pipeline: area-normalize every individual (QC-corrected) spectrum
    first, THEN aggregate to one representative spectrum per patient (median
    primary / mean sensitivity), each patient carrying equal weight downstream."""
    normalized = area_normalize(spectra, waves)
    return compute_patient_representative_spectra(
        normalized, patient_id, target, waves, qc_status=qc_status,
        exclude_statuses=exclude_statuses, method=method,
    )


# ---------------------------------------------------------------------------
# Generic patient-level bootstrap helper (no equivalent generic version exists
# in spectra_ml.bootstrap, which is specialised to the OOF-prediction table).
# ---------------------------------------------------------------------------

def patient_bootstrap_curve(patient_spectra: np.ndarray, n_boot: int, seed: int, ci: float = 95,
                             estimator=np.mean) -> dict:
    """Patient-level bootstrap of an arbitrary per-wavelength statistic
    (typically the group mean curve). `patient_spectra`: (n_patients, n_wave),
    one row per patient (already representative spectra) -- resampling here is
    over PATIENTS, never over that patient's individual spectra."""
    n = patient_spectra.shape[0]
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, patient_spectra.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        reps[b] = estimator(patient_spectra[idx], axis=0)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "point_estimate": estimator(patient_spectra, axis=0),
        "ci_lo": np.percentile(reps, lo_q, axis=0),
        "ci_hi": np.percentile(reps, hi_q, axis=0),
        "replicates": reps,
    }


def class_group_mean(repr_: PatientRepr, target_value: float | None, n_boot: int, seed: int,
                      ci: float = 95) -> dict:
    """Group mean curve (Part VI, N_g patients each weighted equally) + patient
    bootstrap 95% CI. target_value=None -> use ALL patients regardless of
    target (used for the target-blind instrument-vs-instrument shift/shape
    diagnostics, Part VII/IX, which must not depend on target)."""
    if target_value is None:
        mask = np.ones(len(repr_.patient_id), dtype=bool)
    else:
        mask = repr_.target == target_value
    sub = repr_.spectra[mask]
    valid_rows = ~np.isnan(sub).any(axis=1)
    sub = sub[valid_rows]
    n_patients = int(sub.shape[0])
    if n_patients == 0:
        return {"mean": np.full(repr_.spectra.shape[1], np.nan), "ci_lo": None, "ci_hi": None,
                "n_patients": 0, "patient_ids": []}
    if n_patients == 1:
        mean = sub[0]
        return {"mean": mean, "ci_lo": mean.copy(), "ci_hi": mean.copy(), "n_patients": 1,
                "patient_ids": repr_.patient_id[mask][valid_rows].tolist(),
                "note": "n=1 patient: no bootstrap CI possible (CI collapses to the point curve)."}
    boot = patient_bootstrap_curve(sub, n_boot=n_boot, seed=seed, ci=ci, estimator=np.mean)
    return {"mean": boot["point_estimate"], "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
            "n_patients": n_patients, "patient_ids": repr_.patient_id[mask][valid_rows].tolist()}


# ---------------------------------------------------------------------------
# Part VII: wavelength-shift diagnostics
# ---------------------------------------------------------------------------

def _sg_deriv(curve: np.ndarray, window: int, polyorder: int, deriv: int = 1) -> np.ndarray:
    w = min(window, len(curve) - (1 - len(curve) % 2))
    w = w if w % 2 == 1 else w - 1
    w = max(w, polyorder + 2 + (polyorder % 2 == 0))
    if w <= polyorder or w >= len(curve):
        w = len(curve) - 1 if (len(curve) - 1) % 2 == 1 else len(curve) - 2
    return savgol_filter(curve, window_length=w, polyorder=polyorder, deriv=deriv)


def estimate_wavelength_shift(waves: np.ndarray, old_curve: np.ndarray, new_curve: np.ndarray,
                               search_range_nm: tuple[float, float] = (-5.0, 5.0),
                               step_nm: float = 0.02, savgol_window: int = 11,
                               savgol_polyorder: int = 3) -> dict:
    """Part VII.2-3: SG first-derivative + cross-correlation wavelength-shift
    estimate with sub-grid (parabolic) refinement.

    Sign convention: if NEW(lambda) == OLD(lambda - s_true) (i.e. a feature at
    true position lambda0 is reported by the new instrument at lambda0 +
    s_true), this function returns delta_lambda_nm ~= +s_true.
    """
    waves = np.asarray(waves, dtype=float)
    old_d = _sg_deriv(np.asarray(old_curve, dtype=float), savgol_window, savgol_polyorder)
    new_d = _sg_deriv(np.asarray(new_curve, dtype=float), savgol_window, savgol_polyorder)

    lo, hi = search_range_nm
    n_steps = int(round((hi - lo) / step_nm)) + 1
    shifts = lo + np.arange(n_steps) * step_nm
    wmin, wmax = waves.min(), waves.max()

    corrs = np.full(len(shifts), np.nan)
    for i, s in enumerate(shifts):
        shifted_x = waves + s
        valid = (shifted_x >= wmin) & (shifted_x <= wmax)
        if valid.sum() < 0.5 * len(waves):
            continue
        new_d_shifted = np.interp(shifted_x[valid], waves, new_d)
        o = old_d[valid]
        if np.std(new_d_shifted) < 1e-15 or np.std(o) < 1e-15:
            continue
        corrs[i] = np.corrcoef(o, new_d_shifted)[0, 1]

    if np.all(np.isnan(corrs)):
        return {"delta_lambda_nm": np.nan, "best_correlation": np.nan,
                "shift_grid_nm": shifts, "correlation_curve": corrs}

    best_i = int(np.nanargmax(corrs))
    delta_lambda = shifts[best_i]
    if 0 < best_i < len(shifts) - 1 and not np.isnan(corrs[best_i - 1]) and not np.isnan(corrs[best_i + 1]):
        y0, y1, y2 = corrs[best_i - 1], corrs[best_i], corrs[best_i + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            sub_shift = 0.5 * (y0 - y2) / denom
            sub_shift = float(np.clip(sub_shift, -1.0, 1.0))
            delta_lambda = shifts[best_i] + sub_shift * step_nm

    return {"delta_lambda_nm": float(delta_lambda), "best_correlation": float(corrs[best_i]),
            "shift_grid_nm": shifts, "correlation_curve": corrs}


def patient_bootstrap_shift(old_patient_spectra: np.ndarray, new_patient_spectra: np.ndarray, waves: np.ndarray,
                             n_boot: int, seed: int, search_range_nm=(-5.0, 5.0), step_nm: float = 0.02,
                             savgol_window: int = 11, savgol_polyorder: int = 3, ci: float = 95) -> dict:
    """Part VII.4: independently bootstrap OLD and NEW reference-group patients,
    re-estimate delta_lambda for every replicate; return median + CI."""
    rng = np.random.default_rng(seed)
    n_old, n_new = old_patient_spectra.shape[0], new_patient_spectra.shape[0]
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        old_idx = rng.integers(0, n_old, n_old)
        new_idx = rng.integers(0, n_new, n_new)
        old_mean = old_patient_spectra[old_idx].mean(axis=0)
        new_mean = new_patient_spectra[new_idx].mean(axis=0)
        res = estimate_wavelength_shift(waves, old_mean, new_mean, search_range_nm, step_nm,
                                         savgol_window, savgol_polyorder)
        deltas[b] = res["delta_lambda_nm"]
    deltas = deltas[~np.isnan(deltas)]
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "median_delta_lambda_nm": float(np.median(deltas)) if len(deltas) else np.nan,
        "ci_lo": float(np.percentile(deltas, lo_q)) if len(deltas) else np.nan,
        "ci_hi": float(np.percentile(deltas, hi_q)) if len(deltas) else np.nan,
        "n_boot_used": int(len(deltas)), "n_boot_requested": n_boot,
    }


def find_peaks_and_inflections(curve: np.ndarray, waves: np.ndarray, window: int, polyorder: int,
                                prominence_rel: float = 0.05) -> dict:
    """Part VII.5: local maxima (peaks) of the SG-smoothed curve and
    zero-crossings of the second derivative (inflection points), for a
    non-primary corroborating check. `prominence_rel`: minimum peak height
    (relative to the curve's own range) to be reported."""
    from scipy.signal import find_peaks

    smoothed = savgol_filter(curve, window_length=min(window, len(curve) - 1 - (len(curve) % 2 == 0)),
                              polyorder=polyorder)
    rng = smoothed.max() - smoothed.min() + _EPS
    peak_idx, _ = find_peaks(smoothed, prominence=prominence_rel * rng)
    second_deriv = _sg_deriv(curve, window, polyorder, deriv=2)
    sign = np.sign(second_deriv)
    zero_cross = np.where(np.diff(sign) != 0)[0]
    return {"peak_wavelengths": waves[peak_idx].tolist(), "inflection_wavelengths": waves[zero_cross].tolist()}


# ---------------------------------------------------------------------------
# Part VIII: PASS/WARNING/FAIL gate
# ---------------------------------------------------------------------------

@dataclass
class ShiftDecision:
    status: str          # PASS | WARNING | FAIL
    reason: str
    point_estimate_nm: float
    ci_lo_nm: float
    ci_hi_nm: float
    max_allowed_shift_nm: float
    stop_transfer_message: str | None = None

    def to_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "point_estimate_nm": self.point_estimate_nm,
                "ci_lo_nm": self.ci_lo_nm, "ci_hi_nm": self.ci_hi_nm,
                "max_allowed_shift_nm": self.max_allowed_shift_nm,
                "stop_transfer_message": self.stop_transfer_message}


def evaluate_shift_gate(point_estimate_nm: float, ci_lo_nm: float, ci_hi_nm: float,
                         max_allowed_shift_nm: float) -> ShiftDecision:
    """Part VIII gate, threshold fixed BEFORE inspecting any result (configs/harmonization.yaml).

    FAIL:    |point estimate| itself exceeds the threshold -> STOP TRANSFER regardless of CI width.
    WARNING: point estimate is within tolerance, but the bootstrap CI extends beyond it (uncertain).
    PASS:    point estimate AND the full CI stay within tolerance.
    """
    abs_point = abs(point_estimate_nm)
    ci_exceeds = (not np.isnan(ci_lo_nm) and not np.isnan(ci_hi_nm)
                  and (abs(ci_lo_nm) > max_allowed_shift_nm or abs(ci_hi_nm) > max_allowed_shift_nm))

    if np.isnan(point_estimate_nm):
        return ShiftDecision(FAIL, "shift estimation failed (no valid cross-correlation optimum found)",
                              point_estimate_nm, ci_lo_nm, ci_hi_nm, max_allowed_shift_nm, STOP_TRANSFER_MESSAGE)
    if abs_point > max_allowed_shift_nm:
        return ShiftDecision(FAIL, f"|delta_lambda|={abs_point:.4f} nm exceeds max_allowed_shift_nm="
                              f"{max_allowed_shift_nm:.4f} nm", point_estimate_nm, ci_lo_nm, ci_hi_nm,
                              max_allowed_shift_nm, STOP_TRANSFER_MESSAGE)
    if ci_exceeds:
        return ShiftDecision(WARNING, f"point estimate ({point_estimate_nm:+.4f} nm) is within tolerance but "
                              f"the bootstrap 95% CI [{ci_lo_nm:+.4f}, {ci_hi_nm:+.4f}] extends beyond "
                              f"max_allowed_shift_nm={max_allowed_shift_nm:.4f} nm", point_estimate_nm, ci_lo_nm,
                              ci_hi_nm, max_allowed_shift_nm, None)
    return ShiftDecision(PASS, f"|delta_lambda|={abs_point:.4f} nm and its 95% CI are within "
                          f"max_allowed_shift_nm={max_allowed_shift_nm:.4f} nm", point_estimate_nm, ci_lo_nm,
                          ci_hi_nm, max_allowed_shift_nm, None)


# ---------------------------------------------------------------------------
# Part IX / XIV: shape-similarity metrics
# ---------------------------------------------------------------------------

def compute_similarity_metrics(a: np.ndarray, b: np.ndarray, waves: np.ndarray, savgol_window: int = 11,
                                savgol_polyorder: int = 3) -> dict:
    """Pearson/Spearman/cosine/spectral-angle/RMSE/nRMSE/derivative-correlation
    + mean/max absolute residual between two area-normalized mean spectra `a`
    (e.g. OLD) and `b` (e.g. NEW). Symmetric in intent (order only matters for
    the sign of residual = a - b and for nRMSE's normalization, which uses a's
    own dynamic range)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pearson = float(np.corrcoef(a, b)[0, 1])
    spearman = float(spearmanr(a, b).statistic)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + _EPS))
    spectral_angle_rad = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    residual = a - b
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    a_range = a.max() - a.min() + _EPS
    nrmse = float(rmse / a_range)
    a_d = _sg_deriv(a, savgol_window, savgol_polyorder)
    b_d = _sg_deriv(b, savgol_window, savgol_polyorder)
    derivative_corr = float(np.corrcoef(a_d, b_d)[0, 1])
    return {
        "pearson": pearson, "spearman": spearman, "cosine_similarity": cosine,
        "spectral_angle_rad": spectral_angle_rad, "spectral_angle_deg": float(np.degrees(spectral_angle_rad)),
        "rmse": rmse, "nrmse": nrmse, "derivative_correlation": derivative_corr,
        "mean_abs_residual": float(np.mean(np.abs(residual))), "max_abs_residual": float(np.max(np.abs(residual))),
    }


# ---------------------------------------------------------------------------
# Part X/XII: transfer coefficient construction, guard, smoothing, bootstrap
# ---------------------------------------------------------------------------

def build_transfer_curve_raw(numerator_curve: np.ndarray, denominator_curve: np.ndarray,
                              denom_near_zero_rel_threshold: float, edge_exclude_n_points: int) -> dict:
    """K_raw(lambda) = numerator(lambda) / denominator(lambda) (Part X), with
    Part XII.1-2 guards: (1) denominator points below
    `denom_near_zero_rel_threshold` * max(|denominator|) are flagged unreliable
    (K set to NaN there); (2) the outermost `edge_exclude_n_points` wavelengths
    at both ends are excluded regardless of denominator magnitude."""
    numerator = np.asarray(numerator_curve, dtype=float)
    denominator = np.asarray(denominator_curve, dtype=float)
    denom_thresh = denom_near_zero_rel_threshold * (np.max(np.abs(denominator)) + _EPS)
    unreliable = np.abs(denominator) < denom_thresh
    if edge_exclude_n_points > 0:
        unreliable[:edge_exclude_n_points] = True
        unreliable[-edge_exclude_n_points:] = True

    k_raw = np.full_like(numerator, np.nan)
    safe_denom = np.where(unreliable, 1.0, denominator)
    k_raw = numerator / safe_denom
    k_raw[unreliable] = np.nan
    return {"k_raw": k_raw, "unreliable_mask": unreliable}


def smooth_transfer_curve(k_raw: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Part XII.4: fill NaN gaps (interior linear interpolation; edges held
    flat at the nearest reliable value so the curve stays fully defined on the
    whole reference grid -- needed because Part XIII multiplies it against the
    FULL spectrum), then apply a moderate Savitzky-Golay smoothing pass.
    Does NOT touch real multi-nm-scale instrument-response structure by
    construction of the (conservative, see configs/harmonization.yaml)
    window/polyorder."""
    k = np.asarray(k_raw, dtype=float).copy()
    n = len(k)
    idx = np.arange(n)
    valid = ~np.isnan(k)
    if valid.sum() < 2:
        raise ValueError("transfer curve has fewer than 2 reliable points; cannot smooth/extrapolate.")
    filled = np.interp(idx, idx[valid], k[valid])  # np.interp holds edges flat outside [idx[valid].min(), max()]

    w = min(window, n - (1 - n % 2))
    w = w if w % 2 == 1 else w - 1
    w = max(w, polyorder + 2 + (polyorder % 2 == 0))
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    return savgol_filter(filled, window_length=w, polyorder=polyorder)


def bootstrap_transfer_curve(old_patient_spectra: np.ndarray, new_patient_spectra: np.ndarray,
                              denom_near_zero_rel_threshold: float, edge_exclude_n_points: int,
                              smoothing_window: int, smoothing_polyorder: int, n_boot: int, seed: int,
                              ci: float = 95) -> pd.DataFrame:
    """Part XII.3: independently patient-bootstrap the OLD and NEW reference
    groups, rebuild+smooth K for every replicate, return per-wavelength
    median/CI -- also used as the bootstrap-stability check (Part XXXVIII.4:
    result must not hinge catastrophically on a few patients)."""
    n_old, n_new = old_patient_spectra.shape[0], new_patient_spectra.shape[0]
    n_wave = old_patient_spectra.shape[1]
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, n_wave))
    for b in range(n_boot):
        old_idx = rng.integers(0, n_old, n_old)
        new_idx = rng.integers(0, n_new, n_new)
        old_mean = old_patient_spectra[old_idx].mean(axis=0)
        new_mean = new_patient_spectra[new_idx].mean(axis=0)
        raw = build_transfer_curve_raw(old_mean, new_mean, denom_near_zero_rel_threshold,
                                        edge_exclude_n_points)["k_raw"]
        reps[b] = smooth_transfer_curve(raw, smoothing_window, smoothing_polyorder)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    return pd.DataFrame({
        "wave_index": np.arange(n_wave), "bootstrap_median": np.median(reps, axis=0),
        "ci_lo": np.percentile(reps, lo_q, axis=0), "ci_hi": np.percentile(reps, hi_q, axis=0),
        "bootstrap_std": reps.std(axis=0),
    })


# ---------------------------------------------------------------------------
# Part XI: PD-derived vs Control-derived curve comparison (diagnostic only)
# ---------------------------------------------------------------------------

def compare_pd_control_transfer_curves(k_pd: np.ndarray, k_control: np.ndarray, waves: np.ndarray,
                                        min_correlation: float, max_spectral_angle_rad: float,
                                        savgol_window: int = 11, savgol_polyorder: int = 3) -> dict:
    metrics = compute_similarity_metrics(k_pd, k_control, waves, savgol_window, savgol_polyorder)
    relative_deviation = (k_pd - k_control) / (np.abs(k_control) + _EPS)
    diverges = (metrics["pearson"] < min_correlation) or (metrics["spectral_angle_rad"] > max_spectral_angle_rad)
    return {
        "metrics": metrics, "relative_deviation_vs_wavelength": relative_deviation.tolist(),
        "diverges": bool(diverges), "warning": PD_CONTROL_DIVERGENCE_WARNING if diverges else None,
        "threshold_min_correlation": min_correlation, "threshold_max_spectral_angle_rad": max_spectral_angle_rad,
    }


# ---------------------------------------------------------------------------
# Part XIII: apply transfer + re-normalize
# ---------------------------------------------------------------------------

def apply_transfer(spectra_norm: np.ndarray, waves: np.ndarray, k_smoothed: np.ndarray) -> dict:
    """X_before_renorm = X_norm * K ; X_harm = X_before_renorm / integral(X_before_renorm dlambda).
    Both are returned -- the pre-renormalization variant is diagnostic only
    (Part XIII)."""
    before_renorm = np.asarray(spectra_norm, dtype=float) * k_smoothed[None, :]
    harm = area_normalize(before_renorm, waves)
    return {"before_renorm": before_renorm, "harmonized": harm}


# ---------------------------------------------------------------------------
# Part XV: PD-Control contrast preservation
# ---------------------------------------------------------------------------

def pd_control_contrast(pd_patient_spectra: np.ndarray, control_patient_spectra: np.ndarray) -> dict:
    """D(lambda) = mean_PD(lambda) - mean_Control(lambda), patient-level,
    already-area-normalized spectra. Also the relative variant D/|mean_control|."""
    n_pd, n_ctrl = pd_patient_spectra.shape[0], control_patient_spectra.shape[0]
    mean_pd = pd_patient_spectra.mean(axis=0)
    mean_ctrl = control_patient_spectra.mean(axis=0)
    d = mean_pd - mean_ctrl
    d_rel = d / (np.abs(mean_ctrl) + _EPS)
    caveats = []
    if n_pd < 5:
        caveats.append(f"PD group n={n_pd} patients: contrast estimate unstable, small-sample caveat.")
    if n_ctrl < 5:
        caveats.append(f"Control group n={n_ctrl} patients: contrast estimate unstable, small-sample caveat.")
    return {"D": d, "D_relative": d_rel, "n_pd": int(n_pd), "n_control": int(n_ctrl), "caveats": caveats}


def compare_contrast_curves(d_new: np.ndarray, d_old: np.ndarray, waves: np.ndarray, n_regions: int = 5) -> dict:
    pearson = float(np.corrcoef(d_new, d_old)[0, 1]) if np.std(d_new) > 0 and np.std(d_old) > 0 else np.nan
    cosine = float(np.dot(d_new, d_old) / (np.linalg.norm(d_new) * np.linalg.norm(d_old) + _EPS))
    spectral_angle_rad = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    sign_agreement = float(np.mean(np.sign(d_new) == np.sign(d_old)))

    edges = np.linspace(waves.min(), waves.max(), n_regions + 1)
    region_rows = []
    for r in range(n_regions):
        mask = (waves >= edges[r]) & (waves <= edges[r + 1])
        if mask.sum() < 2:
            continue
        dn, do = d_new[mask], d_old[mask]
        region_corr = float(np.corrcoef(dn, do)[0, 1]) if np.std(dn) > 0 and np.std(do) > 0 else np.nan
        region_sign_agree = float(np.mean(np.sign(dn) == np.sign(do)))
        region_rows.append({"wavelength_start": float(edges[r]), "wavelength_end": float(edges[r + 1]),
                             "correlation": region_corr, "sign_agreement": region_sign_agree})
    return {"correlation": pearson, "spectral_angle_rad": spectral_angle_rad,
            "sign_agreement_fraction": sign_agreement, "region_wise": region_rows}


# ---------------------------------------------------------------------------
# Part XVI: instrument-separability diagnostic classifier
# ---------------------------------------------------------------------------

def instrument_separability_auc(spectra_a: np.ndarray, patient_id_a: np.ndarray, spectra_b: np.ndarray,
                                  patient_id_b: np.ndarray, seed: int = 42, n_splits: int = 5,
                                  C: float = 1.0) -> dict:
    """Simple diagnostic: OLD (label 0) vs NEW (label 1) logistic regression on
    SNV-normalized, already area-normalized patient-representative spectra
    from a comparable reference population (e.g. control vs control),
    patient-grouped CV. This is a diagnostic, not a deliverable model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from spectra_ml.splitting import group_stratified_splits

    spectra_a = np.asarray(spectra_a, dtype=float)
    spectra_b = np.asarray(spectra_b, dtype=float)
    X = np.vstack([spectra_a, spectra_b])
    y = np.concatenate([np.zeros(len(spectra_a)), np.ones(len(spectra_b))])
    pid = np.concatenate([np.array([f"old__{p}" for p in patient_id_a]),
                           np.array([f"new__{p}" for p in patient_id_b])])

    def _snv(m):
        mean = m.mean(axis=1, keepdims=True)
        std = m.std(axis=1, keepdims=True)
        return (m - mean) / (std + _EPS)

    folds = group_stratified_splits(pid, y, n_splits=None, seed=seed, max_splits=n_splits, min_splits=3)
    fold_aucs = []
    for f in folds:
        X_tr, y_tr = _snv(X[f.train_idx]), y[f.train_idx]
        X_te, y_te = _snv(X[f.test_idx]), y[f.test_idx]
        model = LogisticRegression(C=C, max_iter=5000)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        if len(set(y_te.tolist())) > 1:
            fold_aucs.append(float(roc_auc_score(y_te, prob)))
    return {
        "fold_aucs": fold_aucs, "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else np.nan,
        "n_a": int(len(spectra_a)), "n_b": int(len(spectra_b)), "n_folds": len(fold_aucs),
    }
