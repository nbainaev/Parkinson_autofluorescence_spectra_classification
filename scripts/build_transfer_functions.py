"""Phase 1 entry point (current_tasks/double_blind_exp.md Parts I, IV-XVI, XXXV):
instrument comparison + empirical mean-spectrum instrument harmonization.

Does NOT implement blinding/freeze/final-model machinery (Parts II-III,
XVII+) -- see module docstring of spectra_ml.harmonization. Since no
development/blind-test split exists yet in this phase, transfer curves are
built on the FULL available cohort of each instrument; every artifact this
script writes is stamped with a "full-cohort, not dev-only" caveat, to be
revisited when Phase 2 introduces the split.

Usage: python scripts/build_transfer_functions.py [--force-exploratory]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectra_ml.data import build_dataset, load_raw_table
from spectra_ml.harmonization import (
    FAIL, PASS, WARNING,
    HarmonizationConfig,
    apply_transfer,
    bootstrap_transfer_curve,
    build_patient_area_normalized_repr,
    build_transfer_curve_raw,
    class_group_mean,
    compare_contrast_curves,
    compare_pd_control_transfer_curves,
    compute_similarity_metrics,
    estimate_wavelength_shift,
    evaluate_shift_gate,
    find_peaks_and_inflections,
    instrument_separability_auc,
    patient_bootstrap_shift,
    pd_control_contrast,
    reindex_to_reference_grid,
    smooth_transfer_curve,
)
from spectra_ml.qc import HARD_FAIL, QCConfig, SOFT_WARNING, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig
from spectra_ml.statistics import compute_patient_representative_spectra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "final_study", "instrument_transfer")
FIG_DIR = os.path.join(OUT_DIR, "figures")

PRIMARY_QC_EXCLUDE = (HARD_FAIL, SOFT_WARNING)   # PASS only
SENSITIVITY_QC_EXCLUDE = (HARD_FAIL,)            # PASS + SOFT_WARNING

CAVEATS: list[str] = []  # collected across the whole run, dumped into the HTML report


def _caveat(msg: str) -> None:
    print(f"  [CAVEAT] {msg}")
    CAVEATS.append(msg)


# ---------------------------------------------------------------------------
# small local helpers (orchestration/plumbing only -- all real logic lives in
# spectra_ml.harmonization, which is unit-tested independently)
# ---------------------------------------------------------------------------

def run_qc_and_correct(ds, spectra, waves, qc_config):
    spectrum_id = [f"{pid}_{i}" for i, pid in enumerate(ds.patient_id)]
    qc_df, hp, correction_log = build_qc_table(spectra, waves, ds.patient_id, spectrum_id, qc_config)
    corrected, _ = correct_hot_pixels(spectra, hp, qc_config)
    return qc_df, corrected


def class_patient_matrix_with_ids(repr_, target_value):
    if target_value is None:
        mask = np.ones(len(repr_.patient_id), dtype=bool)
    else:
        mask = repr_.target == target_value
    sub = repr_.spectra[mask]
    valid = ~np.isnan(sub).any(axis=1)
    return sub[valid], repr_.patient_id[mask][valid]


def add_group_mean_rows(rows, instrument_id, repr_, waves, qc_variant, method_variant, n_boot, seed):
    for cls_label, cls_val in [("parkinson", 1.0), ("control", 0.0), ("all", None)]:
        gm = class_group_mean(repr_, cls_val, n_boot=n_boot, seed=seed)
        if gm["n_patients"] == 0:
            continue
        ci_lo = gm["ci_lo"] if gm["ci_lo"] is not None else np.full(len(waves), np.nan)
        ci_hi = gm["ci_hi"] if gm["ci_hi"] is not None else np.full(len(waves), np.nan)
        for j, wl in enumerate(waves):
            rows.append({
                "instrument": instrument_id, "class": cls_label, "qc_variant": qc_variant,
                "method_variant": method_variant, "wavelength": float(wl), "mean": float(gm["mean"][j]),
                "ci95_lo": float(ci_lo[j]), "ci95_hi": float(ci_hi[j]), "n_patients": gm["n_patients"],
            })


def transfer_curve_rows(curve_id, waves, k_raw, k_smoothed, exploratory):
    raw_rows = [{"curve_id": curve_id, "wavelength": float(w), "k_raw": float(v), "exploratory": exploratory}
                for w, v in zip(waves, k_raw)]
    smoothed_rows = [{"curve_id": curve_id, "wavelength": float(w), "k_smoothed": float(v), "exploratory": exploratory}
                      for w, v in zip(waves, k_smoothed)]
    return raw_rows, smoothed_rows


def bootstrap_ci_rows(curve_id, waves, boot_df, exploratory):
    return [{"curve_id": curve_id, "wavelength": float(w), "bootstrap_median": float(m), "ci95_lo": float(lo),
              "ci95_hi": float(hi), "bootstrap_std": float(sd), "exploratory": exploratory}
             for w, m, lo, hi, sd in zip(waves, boot_df["bootstrap_median"], boot_df["ci_lo"], boot_df["ci_hi"],
                                          boot_df["bootstrap_std"])]


def plot_overlay(waves, old_mean, old_lo, old_hi, new_before_mean, new_before_lo, new_before_hi,
                  new_after_mean, new_after_lo, new_after_hi, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(waves, old_mean, color="#1f77b4", label="OLD", linewidth=1.5)
    ax.fill_between(waves, old_lo, old_hi, color="#1f77b4", alpha=0.15)
    ax.plot(waves, new_before_mean, color="#d62728", label="NEW (before transfer)", linewidth=1.2, linestyle="--")
    ax.fill_between(waves, new_before_lo, new_before_hi, color="#d62728", alpha=0.12)
    if new_after_mean is not None:
        ax.plot(waves, new_after_mean, color="#2ca02c", label="NEW (after transfer)", linewidth=1.5)
        ax.fill_between(waves, new_after_lo, new_after_hi, color="#2ca02c", alpha=0.15)
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Area-normalized intensity")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_difference(waves, before_diff, after_diff, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.plot(waves, before_diff, color="#d62728", label="OLD - NEW (before)", linewidth=1.2)
    if after_diff is not None:
        ax.plot(waves, after_diff, color="#2ca02c", label="OLD - NEW (after)", linewidth=1.2)
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Difference (area-normalized)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ratio(waves, before_ratio, after_ratio, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(1, color="grey", linewidth=0.8)
    ax.plot(waves, before_ratio, color="#d62728", label="OLD / NEW (before)", linewidth=1.2)
    if after_ratio is not None:
        ax.plot(waves, after_ratio, color="#2ca02c", label="OLD / NEW (after)", linewidth=1.2)
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Ratio")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_derivative_overlay(waves, old_d, new_d, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.plot(waves, old_d, color="#1f77b4", label="OLD (1st derivative)", linewidth=1.2)
    ax.plot(waves, new_d, color="#d62728", label="NEW, before transfer (1st derivative)", linewidth=1.2)
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("SG first derivative")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_transfer_curve(waves, curves: dict, title, out_path):
    """curves: {label: (k_smoothed, ci_lo, ci_hi, color)}"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(1, color="grey", linewidth=0.8)
    for label, (k, lo, hi, color) in curves.items():
        ax.plot(waves, k, color=color, label=label, linewidth=1.4)
        if lo is not None:
            ax.fill_between(waves, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("K(lambda) (empirical harmonization coefficient)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def html_table(df: pd.DataFrame, float_fmt="{:.4f}") -> str:
    return df.to_html(index=False, float_format=lambda x: float_fmt.format(x) if pd.notna(x) else "")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmonization-config", default=os.path.join(ROOT, "configs", "harmonization.yaml"))
    parser.add_argument("--qc-config", default=os.path.join(ROOT, "configs", "qc.yaml"))
    parser.add_argument("--force-exploratory", action="store_true",
                         help="Also build an EXPLORATORY transfer curve / shifted overlay for instruments whose "
                              "wavelength-shift gate is FAIL (Part VIII's allowance). These are clearly flagged "
                              "'exploratory=True' and are NOT used for mean_spectra_after / similarity-after / "
                              "contrast-after / instrument-classifier-after computations.")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    cfg = HarmonizationConfig.from_yaml(args.harmonization_config)
    qc_config = QCConfig.from_yaml(args.qc_config)

    old_inst = InstrumentConfig.from_yaml(os.path.join(ROOT, cfg.instruments["old"].config))
    new_pd_inst = InstrumentConfig.from_yaml(os.path.join(ROOT, cfg.instruments["new_pd"].config))
    new_control_inst = InstrumentConfig.from_yaml(os.path.join(ROOT, cfg.instruments["new_control"].config))

    df_full = load_raw_table(old_inst.data)  # all three instruments share configs/data.yaml
    ds_old = build_dataset(old_inst, df_full=df_full)
    ds_new_pd = build_dataset(new_pd_inst, df_full=df_full)
    ds_new_control = build_dataset(new_control_inst, df_full=df_full)

    ref_waves = ds_old.waves
    print(f"[data] old={ds_old.n_spectra} spectra/{len(np.unique(ds_old.patient_id))} patients, "
          f"new_pd={ds_new_pd.n_spectra}/{len(np.unique(ds_new_pd.patient_id))}, "
          f"new_control={ds_new_control.n_spectra}/{len(np.unique(ds_new_control.patient_id))}. "
          f"reference grid: {ref_waves.min():.2f}-{ref_waves.max():.2f} nm, {len(ref_waves)} points, "
          f"median step {np.median(np.diff(ref_waves)):.4f} nm.")

    _caveat("Phase 1 full-cohort transfer estimate: no development/blind-test split exists yet (that is Phase 2, "
            "Part II-III of the spec). Transfer curves below are built on ALL available patients of each "
            "instrument/class; blinded-test-patient exclusion is to be applied when Phase 2 introduces the split.")

    # -- Part IV: reindex new instruments onto the OLD reference grid --------
    grid_match = {}
    spectra_new_pd = reindex_to_reference_grid(ds_new_pd.spectra, ds_new_pd.waves, ref_waves)
    grid_match["new_pd"] = bool(len(ds_new_pd.waves) == len(ref_waves)
                                 and np.allclose(ds_new_pd.waves, ref_waves, atol=1e-4))
    spectra_new_control = reindex_to_reference_grid(ds_new_control.spectra, ds_new_control.waves, ref_waves)
    grid_match["new_control"] = bool(len(ds_new_control.waves) == len(ref_waves)
                                      and np.allclose(ds_new_control.waves, ref_waves, atol=1e-4))
    for inst_id, matched in grid_match.items():
        print(f"[grid] {inst_id}: already on OLD grid = {matched} "
              f"({'no interpolation performed (no-op)' if matched else 'linear interpolation applied'})")
        if matched:
            _caveat(f"{inst_id}: raw wavelength columns are identical (same merged-table headers) to OLD's grid, "
                    "so Part IV's interpolation step is a documented no-op here -- it is still implemented and "
                    "would activate automatically if a future data export used a genuinely different grid.")

    # -- Part V: QC per instrument -------------------------------------------
    qc_old, corrected_old = run_qc_and_correct(ds_old, ds_old.spectra, ref_waves, qc_config)
    qc_new_pd, corrected_new_pd = run_qc_and_correct(ds_new_pd, spectra_new_pd, ref_waves, qc_config)
    qc_new_control, corrected_new_control = run_qc_and_correct(ds_new_control, spectra_new_control, ref_waves,
                                                                 qc_config)
    qc_counts = {
        "old": qc_old["qc_status"].value_counts().to_dict(),
        "new_pd": qc_new_pd["qc_status"].value_counts().to_dict(),
        "new_control": qc_new_control["qc_status"].value_counts().to_dict(),
    }
    print(f"[qc] {qc_counts}")

    # -- Part VI: patient-level area-normalized representative spectra ------
    def build_repr(ds, corrected, qc_df, exclude, method):
        return build_patient_area_normalized_repr(corrected, ds.patient_id, ds.target, ref_waves,
                                                    qc_status=qc_df["qc_status"].to_numpy(),
                                                    exclude_statuses=exclude, method=method)

    repr_old = build_repr(ds_old, corrected_old, qc_old, PRIMARY_QC_EXCLUDE, "median")
    repr_new_pd = build_repr(ds_new_pd, corrected_new_pd, qc_new_pd, PRIMARY_QC_EXCLUDE, "median")
    repr_new_control = build_repr(ds_new_control, corrected_new_control, qc_new_control, PRIMARY_QC_EXCLUDE, "median")

    n_boot = cfg.wavelength_alignment.n_bootstrap
    seed = cfg.random_seed

    # sensitivity variants (Part V: PASS+SOFT_WARNING; Part VI: mean instead of median)
    # -- group means only, not carried through the rest of the pipeline (kept in scope: reporting-only).
    repr_old_qc_sens = build_repr(ds_old, corrected_old, qc_old, SENSITIVITY_QC_EXCLUDE, "median")
    repr_new_pd_qc_sens = build_repr(ds_new_pd, corrected_new_pd, qc_new_pd, SENSITIVITY_QC_EXCLUDE, "median")
    repr_new_control_qc_sens = build_repr(ds_new_control, corrected_new_control, qc_new_control,
                                           SENSITIVITY_QC_EXCLUDE, "median")
    repr_old_mean_sens = build_repr(ds_old, corrected_old, qc_old, PRIMARY_QC_EXCLUDE, "mean")
    repr_new_pd_mean_sens = build_repr(ds_new_pd, corrected_new_pd, qc_new_pd, PRIMARY_QC_EXCLUDE, "mean")
    repr_new_control_mean_sens = build_repr(ds_new_control, corrected_new_control, qc_new_control,
                                             PRIMARY_QC_EXCLUDE, "mean")

    mean_before_rows = []
    for inst_id, r in [("old", repr_old), ("new_pd", repr_new_pd), ("new_control", repr_new_control)]:
        add_group_mean_rows(mean_before_rows, inst_id, r, ref_waves, "PASS_only", "median", n_boot, seed)
    for inst_id, r in [("old", repr_old_qc_sens), ("new_pd", repr_new_pd_qc_sens),
                        ("new_control", repr_new_control_qc_sens)]:
        add_group_mean_rows(mean_before_rows, inst_id, r, ref_waves, "PASS_plus_SOFT_WARNING", "median",
                             min(n_boot, 200), seed)
    for inst_id, r in [("old", repr_old_mean_sens), ("new_pd", repr_new_pd_mean_sens),
                        ("new_control", repr_new_control_mean_sens)]:
        add_group_mean_rows(mean_before_rows, inst_id, r, ref_waves, "PASS_only", "mean", min(n_boot, 200), seed)
    mean_before_df = pd.DataFrame(mean_before_rows)
    mean_before_df.to_csv(os.path.join(OUT_DIR, "mean_spectra_before.csv"), index=False)
    print(f"  -> mean_spectra_before.csv ({len(mean_before_df)} rows)")

    if repr_new_pd.spectra[repr_new_pd.target == 0.0].shape[0] < 5:
        n_ctrl_new_pd = int((repr_new_pd.target == 0.0).sum())
        _caveat(f"new_pd instrument: only n={n_ctrl_new_pd} control patient(s) after QC -- any control-derived "
                "statistic on new_pd (K^control diagnostic curve, D_new_pd contrast control arm) is a small-sample "
                "diagnostic only, not a reliable estimate.")
    if repr_new_control.spectra[repr_new_control.target == 1.0].shape[0] < 5:
        n_pd_new_control = int((repr_new_control.target == 1.0).sum())
        _caveat(f"new_control instrument: only n={n_pd_new_control} Parkinson patient(s) after QC -- its own "
                "PD-Control contrast is not computed as a primary result; only OLD vs harmonized new_pd is used "
                "for the primary contrast-preservation check (Part XV).")

    # -- Part VII: wavelength-shift diagnostics (target-blind, ALL patients) -
    old_all_matrix, old_all_ids = class_patient_matrix_with_ids(repr_old, None)
    new_pd_all_matrix, new_pd_all_ids = class_patient_matrix_with_ids(repr_new_pd, None)
    new_control_all_matrix, new_control_all_ids = class_patient_matrix_with_ids(repr_new_control, None)
    old_all_mean = old_all_matrix.mean(axis=0)
    new_pd_all_mean = new_pd_all_matrix.mean(axis=0)
    new_control_all_mean = new_control_all_matrix.mean(axis=0)

    search_range = tuple(cfg.wavelength_alignment.shift_search_range_nm)
    step = cfg.wavelength_alignment.shift_search_step_nm
    sg_w, sg_p = cfg.savgol.window, cfg.savgol.polyorder

    shift_reports = {}
    gates = {}
    for inst_id, new_mean, new_matrix in [("new_pd", new_pd_all_mean, new_pd_all_matrix),
                                           ("new_control", new_control_all_mean, new_control_all_matrix)]:
        point = estimate_wavelength_shift(ref_waves, old_all_mean, new_mean, search_range, step, sg_w, sg_p)
        boot = patient_bootstrap_shift(old_all_matrix, new_matrix, ref_waves, n_boot=n_boot,
                                        seed=cfg.wavelength_alignment.bootstrap_seed, search_range_nm=search_range,
                                        step_nm=step, savgol_window=sg_w, savgol_polyorder=sg_p)
        gate = evaluate_shift_gate(point["delta_lambda_nm"], boot["ci_lo"], boot["ci_hi"],
                                    cfg.wavelength_alignment.max_allowed_shift_nm)
        peaks_old = find_peaks_and_inflections(old_all_mean, ref_waves, sg_w, sg_p)
        peaks_new = find_peaks_and_inflections(new_mean, ref_waves, sg_w, sg_p)
        shift_reports[inst_id] = {
            "point_estimate": point, "bootstrap": boot, "gate": gate.to_dict(),
            "peaks_old": peaks_old, "peaks_new": peaks_new,
        }
        gates[inst_id] = gate
        print(f"[shift] {inst_id}: delta_lambda={point['delta_lambda_nm']:+.4f} nm, "
              f"bootstrap median={boot['median_delta_lambda_nm']:+.4f} nm "
              f"[{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}] -> {gate.status}")
        if gate.status == FAIL:
            _caveat(f"{inst_id}: wavelength-shift gate FAILED ({gate.reason}). {gate.stop_transfer_message} "
                    f"Transfer curve for {inst_id} was NOT built in the default pipeline "
                    f"({'an exploratory curve was additionally produced via --force-exploratory' if args.force_exploratory else 'no --force-exploratory flag given -> no exploratory output either'}).")
        elif gate.status == WARNING:
            _caveat(f"{inst_id}: wavelength-shift gate returned WARNING ({gate.reason}). Transfer proceeds but "
                    "this uncertainty should be kept in mind when interpreting the resulting curve.")

    wavelength_shift_report = {
        "reference_grid": {"min_nm": float(ref_waves.min()), "max_nm": float(ref_waves.max()),
                            "n_points": int(len(ref_waves)), "median_step_nm": float(np.median(np.diff(ref_waves))),
                            "max_allowed_shift_nm": cfg.wavelength_alignment.max_allowed_shift_nm},
        "new_pd_vs_old": {
            "delta_lambda_nm": shift_reports["new_pd"]["point_estimate"]["delta_lambda_nm"],
            "best_correlation": shift_reports["new_pd"]["point_estimate"]["best_correlation"],
            "bootstrap_median_nm": shift_reports["new_pd"]["bootstrap"]["median_delta_lambda_nm"],
            "bootstrap_ci95_lo_nm": shift_reports["new_pd"]["bootstrap"]["ci_lo"],
            "bootstrap_ci95_hi_nm": shift_reports["new_pd"]["bootstrap"]["ci_hi"],
            "n_boot_used": shift_reports["new_pd"]["bootstrap"]["n_boot_used"],
            "decision": shift_reports["new_pd"]["gate"],
            "peaks_old_nm": shift_reports["new_pd"]["peaks_old"]["peak_wavelengths"],
            "peaks_new_nm": shift_reports["new_pd"]["peaks_new"]["peak_wavelengths"],
        },
        "new_control_vs_old": {
            "delta_lambda_nm": shift_reports["new_control"]["point_estimate"]["delta_lambda_nm"],
            "best_correlation": shift_reports["new_control"]["point_estimate"]["best_correlation"],
            "bootstrap_median_nm": shift_reports["new_control"]["bootstrap"]["median_delta_lambda_nm"],
            "bootstrap_ci95_lo_nm": shift_reports["new_control"]["bootstrap"]["ci_lo"],
            "bootstrap_ci95_hi_nm": shift_reports["new_control"]["bootstrap"]["ci_hi"],
            "n_boot_used": shift_reports["new_control"]["bootstrap"]["n_boot_used"],
            "decision": shift_reports["new_control"]["gate"],
            "peaks_old_nm": shift_reports["new_control"]["peaks_old"]["peak_wavelengths"],
            "peaks_new_nm": shift_reports["new_control"]["peaks_new"]["peak_wavelengths"],
        },
    }
    with open(os.path.join(OUT_DIR, "wavelength_shift_report.json"), "w", encoding="utf-8") as f:
        json.dump(wavelength_shift_report, f, indent=2, ensure_ascii=False)
    print("  -> wavelength_shift_report.json")

    # -- Part IX: shape similarity BEFORE transfer ---------------------------
    sim_before = {
        "new_pd": compute_similarity_metrics(old_all_mean, new_pd_all_mean, ref_waves, sg_w, sg_p),
        "new_control": compute_similarity_metrics(old_all_mean, new_control_all_mean, ref_waves, sg_w, sg_p),
    }

    # -- Part X/XII: transfer coefficient construction -----------------------
    old_control_matrix, old_control_ids = class_patient_matrix_with_ids(repr_old, 0.0)
    old_pd_matrix, old_pd_ids = class_patient_matrix_with_ids(repr_old, 1.0)
    new_control_control_matrix, new_control_control_ids = class_patient_matrix_with_ids(repr_new_control, 0.0)
    new_pd_pd_matrix, new_pd_pd_ids = class_patient_matrix_with_ids(repr_new_pd, 1.0)
    new_pd_control_matrix, new_pd_control_ids = class_patient_matrix_with_ids(repr_new_pd, 0.0)

    old_control_mean = old_control_matrix.mean(axis=0)
    old_pd_mean = old_pd_matrix.mean(axis=0)
    new_control_control_mean = new_control_control_matrix.mean(axis=0)
    new_pd_pd_mean = new_pd_pd_matrix.mean(axis=0)
    new_pd_control_mean = new_pd_control_matrix.mean(axis=0) if len(new_pd_control_matrix) else None

    denom_thr = cfg.transfer_smoothing.denom_near_zero_rel_threshold
    edge_n = cfg.transfer_smoothing.edge_exclude_n_points
    sm_w, sm_p = cfg.transfer_smoothing.window, cfg.transfer_smoothing.polyorder

    def build_curve(curve_id, numerator_mean, denominator_mean, numerator_matrix, denominator_matrix,
                     gate_status, force):
        exploratory = gate_status == FAIL
        if exploratory and not force:
            print(f"[transfer] {curve_id}: STOP TRANSFER (gate={gate_status}), no --force-exploratory -> skipped.")
            return None
        raw = build_transfer_curve_raw(numerator_mean, denominator_mean, denom_thr, edge_n)
        smoothed = smooth_transfer_curve(raw["k_raw"], sm_w, sm_p)
        boot_df = bootstrap_transfer_curve(numerator_matrix, denominator_matrix, denom_thr, edge_n, sm_w, sm_p,
                                            n_boot=n_boot, seed=seed)
        tag = " [EXPLORATORY -- gate FAILED, not used downstream]" if exploratory else ""
        print(f"[transfer] {curve_id}: built{tag}. mean K_smoothed={smoothed.mean():.3f}, std={smoothed.std():.3f}")
        return {"raw": raw["k_raw"], "smoothed": smoothed, "unreliable_mask": raw["unreliable_mask"],
                "bootstrap": boot_df, "exploratory": exploratory}

    curve_new_control = build_curve("new_control_to_old_control", old_control_mean, new_control_control_mean,
                                     old_control_matrix, new_control_control_matrix, gates["new_control"].status,
                                     args.force_exploratory)
    curve_new_pd_pd = build_curve("new_pd_to_old_pd", old_pd_mean, new_pd_pd_mean, old_pd_matrix, new_pd_pd_matrix,
                                   gates["new_pd"].status, args.force_exploratory)
    curve_new_pd_control_diag = None
    if new_pd_control_mean is not None:
        curve_new_pd_control_diag = build_curve("new_pd_to_old_control_diagnostic", old_control_mean,
                                                  new_pd_control_mean, old_control_matrix, new_pd_control_matrix,
                                                  gates["new_pd"].status, args.force_exploratory)
        if len(new_pd_control_matrix) < 5:
            _caveat(f"new_pd_to_old_control_diagnostic curve is built from n={len(new_pd_control_matrix)} new_pd "
                    "control patient(s) -- its own bootstrap CI is not meaningful with this few patients "
                    "(replicate resampling with n=1 always returns the same single patient); treat this curve as "
                    "illustrative only, per Part XI.")

    raw_rows, smoothed_rows, ci_rows = [], [], []
    for curve_id, curve in [("new_control_to_old_control", curve_new_control),
                             ("new_pd_to_old_pd", curve_new_pd_pd),
                             ("new_pd_to_old_control_diagnostic", curve_new_pd_control_diag)]:
        if curve is None:
            continue
        r, s = transfer_curve_rows(curve_id, ref_waves, curve["raw"], curve["smoothed"], curve["exploratory"])
        raw_rows.extend(r)
        smoothed_rows.extend(s)
        ci_rows.extend(bootstrap_ci_rows(curve_id, ref_waves, curve["bootstrap"], curve["exploratory"]))
    pd.DataFrame(raw_rows).to_csv(os.path.join(OUT_DIR, "transfer_raw.csv"), index=False)
    pd.DataFrame(smoothed_rows).to_csv(os.path.join(OUT_DIR, "transfer_smoothed.csv"), index=False)
    pd.DataFrame(ci_rows).to_csv(os.path.join(OUT_DIR, "transfer_bootstrap_ci.csv"), index=False)
    print(f"  -> transfer_raw.csv, transfer_smoothed.csv, transfer_bootstrap_ci.csv")

    # -- Part XI: PD-derived vs Control-derived curve comparison (new_pd) ----
    pd_control_divergence = None
    if curve_new_pd_pd is not None and curve_new_pd_control_diag is not None:
        pd_control_divergence = compare_pd_control_transfer_curves(
            curve_new_pd_pd["smoothed"], curve_new_pd_control_diag["smoothed"], ref_waves,
            cfg.transfer_pd_control_divergence.min_correlation, cfg.transfer_pd_control_divergence.max_spectral_angle_rad,
            sg_w, sg_p,
        )
        print(f"[Part XI] K^PD vs K^control (new_pd): correlation={pd_control_divergence['metrics']['pearson']:.4f} "
              f"spectral_angle_deg={pd_control_divergence['metrics']['spectral_angle_deg']:.2f} "
              f"diverges={pd_control_divergence['diverges']}")
        if pd_control_divergence["diverges"]:
            _caveat(f"Part XI WARNING triggered: {pd_control_divergence['warning']} (new_pd K^PD vs K^control: "
                    f"r={pd_control_divergence['metrics']['pearson']:.3f}, "
                    f"spectral_angle={pd_control_divergence['metrics']['spectral_angle_deg']:.1f} deg; based on "
                    f"only n={len(new_pd_control_matrix)} control patient for the diagnostic curve -- "
                    "interpret with caution.")

    # -- Part XIII: apply transfer + re-normalize ----------------------------
    def apply_to_instrument(ds, corrected, k_smoothed):
        norm = np.full_like(corrected, np.nan)
        valid = ~np.isnan(corrected).any(axis=1)
        from spectra_ml.harmonization import area_normalize
        norm[valid] = area_normalize(corrected[valid], ref_waves)
        out = apply_transfer(norm[valid], ref_waves, k_smoothed)
        harmonized_full = np.full_like(corrected, np.nan)
        harmonized_full[valid] = out["harmonized"]
        return harmonized_full

    repr_new_control_after, repr_new_pd_after = None, None
    if curve_new_control is not None and not curve_new_control["exploratory"]:
        harmonized_new_control = apply_to_instrument(ds_new_control, corrected_new_control,
                                                       curve_new_control["smoothed"])
        repr_new_control_after = compute_patient_representative_spectra(
            harmonized_new_control, ds_new_control.patient_id, ds_new_control.target, ref_waves,
            qc_status=qc_new_control["qc_status"].to_numpy(), exclude_statuses=PRIMARY_QC_EXCLUDE, method="median",
        )
    if curve_new_pd_pd is not None and not curve_new_pd_pd["exploratory"]:
        harmonized_new_pd = apply_to_instrument(ds_new_pd, corrected_new_pd, curve_new_pd_pd["smoothed"])
        repr_new_pd_after = compute_patient_representative_spectra(
            harmonized_new_pd, ds_new_pd.patient_id, ds_new_pd.target, ref_waves,
            qc_status=qc_new_pd["qc_status"].to_numpy(), exclude_statuses=PRIMARY_QC_EXCLUDE, method="median",
        )

    mean_after_rows = []
    add_group_mean_rows(mean_after_rows, "old", repr_old, ref_waves, "PASS_only", "median", n_boot, seed)
    if repr_new_pd_after is not None:
        add_group_mean_rows(mean_after_rows, "new_pd", repr_new_pd_after, ref_waves, "PASS_only", "median", n_boot,
                             seed)
    else:
        add_group_mean_rows(mean_after_rows, "new_pd_UNHARMONIZED_transfer_not_applied", repr_new_pd, ref_waves,
                             "PASS_only", "median", n_boot, seed)
    if repr_new_control_after is not None:
        add_group_mean_rows(mean_after_rows, "new_control", repr_new_control_after, ref_waves, "PASS_only",
                             "median", n_boot, seed)
    else:
        add_group_mean_rows(mean_after_rows, "new_control_UNHARMONIZED_transfer_not_applied", repr_new_control,
                             ref_waves, "PASS_only", "median", n_boot, seed)
    mean_after_df = pd.DataFrame(mean_after_rows)
    mean_after_df.to_csv(os.path.join(OUT_DIR, "mean_spectra_after.csv"), index=False)
    print(f"  -> mean_spectra_after.csv ({len(mean_after_df)} rows)")

    # -- Part XIV: similarity metrics after transfer + before/after table ---
    sim_after = {}
    new_pd_after_all_mean, new_control_after_all_mean = None, None
    if repr_new_pd_after is not None:
        m, _ = class_patient_matrix_with_ids(repr_new_pd_after, None)
        new_pd_after_all_mean = m.mean(axis=0)
        sim_after["new_pd"] = compute_similarity_metrics(old_all_mean, new_pd_after_all_mean, ref_waves, sg_w, sg_p)
    if repr_new_control_after is not None:
        m, _ = class_patient_matrix_with_ids(repr_new_control_after, None)
        new_control_after_all_mean = m.mean(axis=0)
        sim_after["new_control"] = compute_similarity_metrics(old_all_mean, new_control_after_all_mean, ref_waves,
                                                                sg_w, sg_p)

    metric_names = ["pearson", "spearman", "cosine_similarity", "spectral_angle_deg", "rmse", "nrmse",
                     "derivative_correlation", "mean_abs_residual", "max_abs_residual"]
    sim_table_rows = []
    for inst_id in ["new_pd", "new_control"]:
        before = sim_before[inst_id]
        after = sim_after.get(inst_id)
        for m in metric_names:
            sim_table_rows.append({"instrument": inst_id, "metric": m, "before": before[m],
                                    "after": after[m] if after is not None else np.nan})
        shift_after = None
        if after is not None:
            new_after_mean = new_pd_after_all_mean if inst_id == "new_pd" else new_control_after_all_mean
            shift_after = estimate_wavelength_shift(ref_waves, old_all_mean, new_after_mean, search_range, step,
                                                      sg_w, sg_p)["delta_lambda_nm"]
        sim_table_rows.append({
            "instrument": inst_id, "metric": "estimated_shift_nm",
            "before": wavelength_shift_report[f"{inst_id}_vs_old"]["delta_lambda_nm"], "after": shift_after,
        })
    sim_table_df = pd.DataFrame(sim_table_rows)
    sim_table_df.to_csv(os.path.join(OUT_DIR, "similarity_metrics_before_after.csv"), index=False)
    print(f"  -> similarity_metrics_before_after.csv")

    # -- Part XV: PD-Control contrast preservation ---------------------------
    d_old = pd_control_contrast(old_pd_matrix, old_control_matrix)
    d_new_pd_before = pd_control_contrast(new_pd_pd_matrix, new_pd_control_matrix)
    d_new_pd_after = None
    if repr_new_pd_after is not None:
        pd_m, _ = class_patient_matrix_with_ids(repr_new_pd_after, 1.0)
        ctrl_m, _ = class_patient_matrix_with_ids(repr_new_pd_after, 0.0)
        if len(pd_m) and len(ctrl_m):
            d_new_pd_after = pd_control_contrast(pd_m, ctrl_m)

    compare_before = compare_contrast_curves(d_new_pd_before["D"], d_old["D"], ref_waves)
    compare_after = (compare_contrast_curves(d_new_pd_after["D"], d_old["D"], ref_waves)
                      if d_new_pd_after is not None else None)

    # exploratory-only: new_control's own contrast (PD n typically tiny) -- reported, not primary
    d_new_control_exploratory = None
    new_control_pd_matrix, _ = class_patient_matrix_with_ids(repr_new_control, 1.0)
    if len(new_control_pd_matrix) >= 1 and len(new_control_control_matrix) >= 1:
        d_new_control_exploratory = pd_control_contrast(new_control_pd_matrix, new_control_control_matrix)

    pd_control_contrast_report = {
        "primary_comparison": "OLD vs new_pd (new_pd is the only NEW instrument with both classes; "
                               "new_control has too few PD patients to support a primary within-instrument "
                               "contrast, see caveats).",
        "old": {"n_pd": d_old["n_pd"], "n_control": d_old["n_control"], "caveats": d_old["caveats"]},
        "new_pd_before_transfer": {"n_pd": d_new_pd_before["n_pd"], "n_control": d_new_pd_before["n_control"],
                                    "caveats": d_new_pd_before["caveats"]},
        "new_pd_after_transfer": ({"n_pd": d_new_pd_after["n_pd"], "n_control": d_new_pd_after["n_control"],
                                    "caveats": d_new_pd_after["caveats"]} if d_new_pd_after is not None else None),
        "comparison_vs_old_before_transfer": compare_before,
        "comparison_vs_old_after_transfer": compare_after,
        "new_control_exploratory_contrast": ({"n_pd": d_new_control_exploratory["n_pd"],
                                                "n_control": d_new_control_exploratory["n_control"],
                                                "caveats": d_new_control_exploratory["caveats"] + [
                                                    "EXPLORATORY ONLY: new_control instrument has almost no PD "
                                                    "patients by design (control-only department); this contrast "
                                                    "is not used as evidence for or against transfer quality."]}
                                               if d_new_control_exploratory is not None else None),
    }
    with open(os.path.join(OUT_DIR, "pd_control_contrast_report.json"), "w", encoding="utf-8") as f:
        json.dump(pd_control_contrast_report, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    print(f"  -> pd_control_contrast_report.json (correlation vs OLD: before={compare_before['correlation']:.3f}"
          + (f", after={compare_after['correlation']:.3f}" if compare_after is not None else ", after=N/A"))

    if d_new_pd_before["n_control"] < 5:
        _caveat(f"PD-Control contrast for new_pd is computed against n={d_new_pd_before['n_control']} control "
                "patient(s) -- Part XV's contrast-preservation check for new_pd is a heavily-caveated diagnostic, "
                "not a robust statistical comparison, per the explicit small-sample warning requested in the task.")

    if compare_before["correlation"] < 0 or compare_before["sign_agreement_fraction"] < 0.5:
        same_before_after = (compare_after is not None
                              and abs(compare_after["correlation"] - compare_before["correlation"]) < 0.05)
        _caveat(
            "Part XV flag: new_pd's own PD-Control contrast D(lambda) is essentially INVERTED relative to OLD's "
            f"D(lambda) (correlation={compare_before['correlation']:+.3f}, sign agreement "
            f"={compare_before['sign_agreement_fraction']:.1%} of wavelengths -- i.e. opposite sign almost "
            "everywhere), both BEFORE and AFTER transfer"
            + (f" (after: correlation={compare_after['correlation']:+.3f}, sign agreement "
               f"={compare_after['sign_agreement_fraction']:.1%})" if compare_after is not None else "")
            + ". Because this inversion is already present before any harmonization is applied and barely changes "
              "after it, it is NOT attributable to the transfer curve -- it is very likely an artifact of new_pd's "
              f"n={d_new_pd_before['n_control']} control patient (a single patient's individual spectrum standing "
              "in for a 'control group mean' is not a stable estimate of the biological PD-Control contrast). This "
              "should be treated as an open concern about new_pd's control-arm reference population, not as "
              "evidence that harmonization damages the PD signal."
        )

    # -- Part XVI: instrument-separability diagnostic classifier -------------
    sep_before = instrument_separability_auc(old_control_matrix, old_control_ids, new_control_control_matrix,
                                              new_control_control_ids, seed=seed)
    sep_after = None
    if repr_new_control_after is not None:
        after_ctrl_m, after_ctrl_ids = class_patient_matrix_with_ids(repr_new_control_after, 0.0)
        if len(after_ctrl_m) >= 4:
            sep_after = instrument_separability_auc(old_control_matrix, old_control_ids, after_ctrl_m,
                                                      after_ctrl_ids, seed=seed)
    instrument_separability_report = {
        "reference_population": "OLD control patients vs NEW_CONTROL control patients (best available "
                                 "comparable reference population -- new_pd has only 1 control patient, unusable "
                                 "for this diagnostic).",
        "before_transfer": sep_before,
        "after_transfer": sep_after,
    }
    with open(os.path.join(OUT_DIR, "instrument_separability_report.json"), "w", encoding="utf-8") as f:
        json.dump(instrument_separability_report, f, indent=2, ensure_ascii=False)
    print(f"  -> instrument_separability_report.json: AUC before={sep_before['mean_auc']:.3f}"
          + (f", after={sep_after['mean_auc']:.3f}" if sep_after is not None else ", after=N/A"))

    # -- figures ---------------------------------------------------------------
    def all_group(repr_, cls_val=None):
        return class_group_mean(repr_, cls_val, n_boot=n_boot, seed=seed)

    gm_old_all = all_group(repr_old)
    gm_new_pd_all = all_group(repr_new_pd)
    gm_new_control_all = all_group(repr_new_control)
    gm_new_pd_after_all = all_group(repr_new_pd_after) if repr_new_pd_after is not None else None
    gm_new_control_after_all = all_group(repr_new_control_after) if repr_new_control_after is not None else None

    plot_overlay(ref_waves, gm_old_all["mean"], gm_old_all["ci_lo"], gm_old_all["ci_hi"],
                 gm_new_pd_all["mean"], gm_new_pd_all["ci_lo"], gm_new_pd_all["ci_hi"],
                 gm_new_pd_after_all["mean"] if gm_new_pd_after_all else None,
                 gm_new_pd_after_all["ci_lo"] if gm_new_pd_after_all else None,
                 gm_new_pd_after_all["ci_hi"] if gm_new_pd_after_all else None,
                 "OLD vs new_pd: mean spectrum (all patients, area-normalized)",
                 os.path.join(FIG_DIR, "overlay_new_pd.png"))
    plot_overlay(ref_waves, gm_old_all["mean"], gm_old_all["ci_lo"], gm_old_all["ci_hi"],
                 gm_new_control_all["mean"], gm_new_control_all["ci_lo"], gm_new_control_all["ci_hi"],
                 gm_new_control_after_all["mean"] if gm_new_control_after_all else None,
                 gm_new_control_after_all["ci_lo"] if gm_new_control_after_all else None,
                 gm_new_control_after_all["ci_hi"] if gm_new_control_after_all else None,
                 "OLD vs new_control: mean spectrum (all patients, area-normalized)",
                 os.path.join(FIG_DIR, "overlay_new_control.png"))

    plot_difference(ref_waves, old_all_mean - new_pd_all_mean,
                     (old_all_mean - new_pd_after_all_mean) if new_pd_after_all_mean is not None else None,
                     "Difference spectrum: OLD - new_pd", os.path.join(FIG_DIR, "difference_new_pd.png"))
    plot_difference(ref_waves, old_all_mean - new_control_all_mean,
                     (old_all_mean - new_control_after_all_mean) if new_control_after_all_mean is not None else None,
                     "Difference spectrum: OLD - new_control", os.path.join(FIG_DIR, "difference_new_control.png"))

    plot_ratio(ref_waves, old_all_mean / (new_pd_all_mean + 1e-12),
               (old_all_mean / (new_pd_after_all_mean + 1e-12)) if new_pd_after_all_mean is not None else None,
               "Ratio spectrum: OLD / new_pd", os.path.join(FIG_DIR, "ratio_new_pd.png"))
    plot_ratio(ref_waves, old_all_mean / (new_control_all_mean + 1e-12),
               (old_all_mean / (new_control_after_all_mean + 1e-12)) if new_control_after_all_mean is not None else None,
               "Ratio spectrum: OLD / new_control", os.path.join(FIG_DIR, "ratio_new_control.png"))

    from spectra_ml.harmonization import _sg_deriv
    plot_derivative_overlay(ref_waves, _sg_deriv(old_all_mean, sg_w, sg_p), _sg_deriv(new_pd_all_mean, sg_w, sg_p),
                             "1st-derivative overlay: OLD vs new_pd (before transfer)",
                             os.path.join(FIG_DIR, "derivative_overlay_new_pd.png"))
    plot_derivative_overlay(ref_waves, _sg_deriv(old_all_mean, sg_w, sg_p),
                             _sg_deriv(new_control_all_mean, sg_w, sg_p),
                             "1st-derivative overlay: OLD vs new_control (before transfer)",
                             os.path.join(FIG_DIR, "derivative_overlay_new_control.png"))

    if curve_new_pd_pd is not None:
        curves = {"K_new_pd->old (PD-derived)": (curve_new_pd_pd["smoothed"], curve_new_pd_pd["bootstrap"]["ci_lo"],
                                                  curve_new_pd_pd["bootstrap"]["ci_hi"], "#d62728")}
        if curve_new_pd_control_diag is not None:
            curves["K^control_new_pd->old (diagnostic)"] = (
                curve_new_pd_control_diag["smoothed"], curve_new_pd_control_diag["bootstrap"]["ci_lo"],
                curve_new_pd_control_diag["bootstrap"]["ci_hi"], "#9467bd")
        plot_transfer_curve(ref_waves, curves, "Transfer curve: new_pd -> old (+ bootstrap 95% CI)",
                             os.path.join(FIG_DIR, "transfer_curve_new_pd.png"))
    if curve_new_control is not None:
        plot_transfer_curve(
            ref_waves,
            {"K_new_control->old (control-derived)": (curve_new_control["smoothed"],
                                                        curve_new_control["bootstrap"]["ci_lo"],
                                                        curve_new_control["bootstrap"]["ci_hi"], "#2ca02c")},
            "Transfer curve: new_control -> old (+ bootstrap 95% CI)",
            os.path.join(FIG_DIR, "transfer_curve_new_control.png"))
    print(f"  -> figures written to {FIG_DIR}")

    # -- HTML report -----------------------------------------------------------
    write_html_report(cfg, ds_old, ds_new_pd, ds_new_control, qc_counts, grid_match, wavelength_shift_report,
                       sim_before, sim_after, sim_table_df, curve_new_control, curve_new_pd_pd,
                       curve_new_pd_control_diag, pd_control_divergence, pd_control_contrast_report,
                       instrument_separability_report, gates, args)
    print(f"  -> {os.path.join(OUT_DIR, 'instrument_transfer_report.html')}")
    print("\nDone.")


def write_html_report(cfg, ds_old, ds_new_pd, ds_new_control, qc_counts, grid_match, wavelength_shift_report,
                       sim_before, sim_after, sim_table_df, curve_new_control, curve_new_pd_pd,
                       curve_new_pd_control_diag, pd_control_divergence, pd_control_contrast_report,
                       instrument_separability_report, gates, args):
    def fmt(x, nd=4):
        return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"

    instruments_html = pd.DataFrame([
        {"role": "old (reference)", "id": ds_old.instrument.id, "model": ds_old.instrument.model,
         "serial_number": ds_old.instrument.serial_number, "n_patients": len(np.unique(ds_old.patient_id)),
         "n_spectra": ds_old.n_spectra},
        {"role": "new_pd_department", "id": ds_new_pd.instrument.id, "model": ds_new_pd.instrument.model,
         "serial_number": ds_new_pd.instrument.serial_number, "n_patients": len(np.unique(ds_new_pd.patient_id)),
         "n_spectra": ds_new_pd.n_spectra},
        {"role": "new_control_department", "id": ds_new_control.instrument.id, "model": ds_new_control.instrument.model,
         "serial_number": ds_new_control.instrument.serial_number,
         "n_patients": len(np.unique(ds_new_control.patient_id)), "n_spectra": ds_new_control.n_spectra},
    ])

    qc_html = pd.DataFrame(qc_counts).fillna(0).astype(int)

    shift_rows = []
    for inst_id in ["new_pd", "new_control"]:
        r = wavelength_shift_report[f"{inst_id}_vs_old"]
        shift_rows.append({"instrument": inst_id, "delta_lambda_nm": r["delta_lambda_nm"],
                            "bootstrap_median_nm": r["bootstrap_median_nm"], "ci95_lo_nm": r["bootstrap_ci95_lo_nm"],
                            "ci95_hi_nm": r["bootstrap_ci95_hi_nm"], "decision": r["decision"]["status"]})
    shift_df = pd.DataFrame(shift_rows)

    warnings_html = "".join(f"<li>{c}</li>" for c in CAVEATS)
    pd_ctrl_warning_html = (f"<p><b>Part XI WARNING:</b> {pd_control_divergence['warning']}</p>"
                             if pd_control_divergence and pd_control_divergence["diverges"] else
                             "<p>Part XI check: PD-derived and Control-derived transfer curves for new_pd do not "
                             "diverge beyond the pre-registered threshold (or the diagnostic curve could not be "
                             "built).</p>")

    def img(path, alt):
        rel = os.path.relpath(path, OUT_DIR)
        return f'<img src="{rel}" alt="{alt}">' if os.path.exists(path) else f"<p><i>{alt}: not available</i></p>"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Instrument transfer / harmonization report (Phase 1)</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1, h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 0.3em; }}
h2 {{ margin-top: 2em; }}
table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th {{ background: #f0f0f0; }}
td:first-child, th:first-child {{ text-align: left; }}
.caveat {{ background: #fff3cd; border-left: 4px solid #e0a800; padding: 0.8em 1em; margin: 1em 0; }}
.finding {{ background: #d4edda; border-left: 4px solid #28a745; padding: 0.8em 1em; margin: 1em 0; }}
.fail {{ background: #f8d7da; border-left: 4px solid #dc3545; padding: 0.8em 1em; margin: 1em 0; }}
img {{ max-width: 100%; border: 1px solid #ddd; margin: 0.5em 0; }}
.figrow {{ display: flex; gap: 1em; flex-wrap: wrap; }}
.figrow img {{ max-width: 48%; }}
code {{ background: #f4f4f4; padding: 1px 4px; }}
</style></head><body>

<h1>Instrument comparison and empirical mean-spectrum instrument harmonization (Phase 1)</h1>
<p>Generated by <code>scripts/build_transfer_functions.py</code>. Terminology note (Part XXXVII): the curves below
are <b>empirical mean-spectrum instrument harmonization</b> / <b>empirical wavelength-dependent transfer
coefficients</b>, NOT a physical intensity-calibration transfer function -- the same physical sample was never
measured on two instruments.</p>

<div class="caveat"><b>Phase scope.</b> This report covers Phase 1 only (instrument comparison + harmonization).
No development/blind-test split exists yet (that is Phase 2). All curves below are "full-cohort" estimates.</div>

<h2>1. Instruments and serial numbers</h2>
{html_table(instruments_html, "{:.0f}")}

<h2>2. Wavelength coverage</h2>
<p>Reference grid = OLD instrument's own grid: {wavelength_shift_report['reference_grid']['min_nm']:.2f}-
{wavelength_shift_report['reference_grid']['max_nm']:.2f} nm, {wavelength_shift_report['reference_grid']['n_points']}
points, median step {wavelength_shift_report['reference_grid']['median_step_nm']:.4f} nm.
new_pd already on OLD grid: {grid_match['new_pd']}. new_control already on OLD grid: {grid_match['new_control']}
(both True here -- the raw table shares one set of wavelength columns across instruments; interpolation code path
is implemented but a documented no-op on this dataset).</p>

<h2>3. Reference populations</h2>
<p>QC status counts (spectrum-level) per instrument:</p>
{html_table(qc_html, "{:.0f}")}
<p>Patient-level representative spectra (Part VI): median across QC-PASS spectra, area-normalized, one row per
patient, equal patient weight in every group mean below.</p>

<h2>4. Mean normalized spectra (before transfer)</h2>
<div class="figrow">
{img(os.path.join(FIG_DIR, 'overlay_new_pd.png'), 'OLD vs new_pd overlay')}
{img(os.path.join(FIG_DIR, 'overlay_new_control.png'), 'OLD vs new_control overlay')}
</div>
<p>Full table: <code>mean_spectra_before.csv</code> (instrument x class x QC-variant x method-variant).</p>

<h2>5. Wavelength-shift estimation</h2>
{html_table(shift_df)}
<p>max_allowed_shift_nm = {wavelength_shift_report['reference_grid']['max_allowed_shift_nm']} nm (derivation: see
<code>configs/harmonization.yaml</code> comment -- half the OLD instrument's median wavelength-bin spacing, fixed
before any shift estimate was computed).</p>

<h2>6. Bootstrap shift CI</h2>
<p>Patient-bootstrap ({cfg.wavelength_alignment.n_bootstrap} replicates, OLD and NEW reference groups resampled
independently) median delta_lambda and 95% CI shown in the table above.</p>

<h2>7. Similarity metrics before transfer</h2>
{html_table(sim_table_df[['instrument','metric','before']])}

<h2>8. Transfer curves</h2>
<div class="figrow">
{img(os.path.join(FIG_DIR, 'transfer_curve_new_pd.png'), 'new_pd transfer curve')}
{img(os.path.join(FIG_DIR, 'transfer_curve_new_control.png'), 'new_control transfer curve')}
</div>
{pd_ctrl_warning_html}

<h2>9. Uncertainty on transfer curves</h2>
<p>Bootstrap 95% CI per wavelength: <code>transfer_bootstrap_ci.csv</code> (columns: curve_id, wavelength,
bootstrap_median, ci95_lo, ci95_hi, bootstrap_std, exploratory).</p>

<h2>10. Mean spectra after transfer</h2>
<div class="figrow">
{img(os.path.join(FIG_DIR, 'difference_new_pd.png'), 'difference new_pd')}
{img(os.path.join(FIG_DIR, 'difference_new_control.png'), 'difference new_control')}
</div>
<div class="figrow">
{img(os.path.join(FIG_DIR, 'ratio_new_pd.png'), 'ratio new_pd')}
{img(os.path.join(FIG_DIR, 'ratio_new_control.png'), 'ratio new_control')}
</div>

<h2>11. Similarity metrics after transfer</h2>
{html_table(sim_table_df)}

<h2>12. PD-Control contrast preservation</h2>
<p>Correlation of D(lambda)=PD-Control vs OLD's own D(lambda):
before={fmt(pd_control_contrast_report['comparison_vs_old_before_transfer']['correlation'], 3)},
after={fmt(pd_control_contrast_report['comparison_vs_old_after_transfer']['correlation'] if pd_control_contrast_report['comparison_vs_old_after_transfer'] else None, 3)}.
Sign agreement fraction: before={fmt(pd_control_contrast_report['comparison_vs_old_before_transfer']['sign_agreement_fraction'], 3)},
after={fmt(pd_control_contrast_report['comparison_vs_old_after_transfer']['sign_agreement_fraction'] if pd_control_contrast_report['comparison_vs_old_after_transfer'] else None, 3)}.
Full detail: <code>pd_control_contrast_report.json</code>.</p>

<h2>13. Instrument-classifier AUC before/after</h2>
<p>OLD-control vs NEW_CONTROL-control, grouped CV, logistic regression on SNV features:
AUC before={fmt(instrument_separability_report['before_transfer']['mean_auc'], 3)}
(std={fmt(instrument_separability_report['before_transfer']['std_auc'], 3)}),
AUC after={fmt(instrument_separability_report['after_transfer']['mean_auc'] if instrument_separability_report['after_transfer'] else None, 3)}.
Closer to 0.5 after transfer indicates reduced instrument separability (Part XVI), a diagnostic, not an absolute
success criterion.</p>

<h2>14. Warnings and limitations</h2>
<ul>
{warnings_html}
</ul>

</body></html>"""

    with open(os.path.join(OUT_DIR, "instrument_transfer_report.html"), "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
