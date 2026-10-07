"""Phase 2 step 2 (current_tasks/double_blind_exp.md Part X constraint, and the
general Part XXXIV prohibition on ever touching blinded patients/labels before
freeze): rebuild the empirical mean-spectrum instrument-harmonization transfer
curves using ONLY development patients (outputs/final_study/development_manifest.csv).

This is a thin, filtered wrapper around the same spectra_ml.harmonization
functions Phase 1's scripts/build_transfer_functions.py used -- no shift/
transfer/similarity logic is reimplemented here. The one behavioral
difference from Phase 1: every dataset is restricted to development-manifest
patient_ids BEFORE any statistic (QC aggregate, mean spectrum, shift
estimate, transfer curve, similarity metric, separability classifier) is
computed, so blind-test spectra/labels never enter any computation in this
script.

Known gap vs. Phase 1 (documented, not hidden -- see prepare_blinded_study.py's
module docstring for why): the development split leaves ZERO new_pd control
patients (the sole labeled new_pd control patient was assigned to the blind
test side so the primary endpoint is computable). Consequences, both
unavoidable given that split:
  * Part XI's new_pd PD-derived-vs-Control-derived transfer curve comparison
    cannot be built for the real study (no control-derived curve exists).
  * Part XV's new_pd within-instrument PD-Control contrast-preservation check
    cannot be built either (no new_pd control arm).
Phase 1's full-cohort exploratory report already covers both diagnostics
(with the appropriate small-n caveats) and is referenced here instead.

Outputs: outputs/final_study/transfer/ (separate from Phase 1's
outputs/final_study/instrument_transfer/, which is left untouched).

Usage: python scripts/rebuild_transfer_functions_development.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectra_ml.data import build_dataset, load_raw_table
from spectra_ml.harmonization import (
    FAIL, PASS, WARNING,
    HarmonizationConfig,
    apply_transfer,
    area_normalize,
    bootstrap_transfer_curve,
    build_patient_area_normalized_repr,
    build_transfer_curve_raw,
    class_group_mean,
    compare_contrast_curves,
    compute_similarity_metrics,
    estimate_wavelength_shift,
    evaluate_shift_gate,
    instrument_separability_auc,
    patient_bootstrap_shift,
    pd_control_contrast,
    reindex_to_reference_grid,
    smooth_transfer_curve,
)
from spectra_ml.integrity import write_manifest
from spectra_ml.qc import HARD_FAIL, QCConfig, SOFT_WARNING, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig
from spectra_ml.statistics import compute_patient_representative_spectra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY_DIR = os.path.join(ROOT, "outputs", "final_study")
OUT_DIR = os.path.join(STUDY_DIR, "transfer")
FIG_DIR = os.path.join(OUT_DIR, "figures")

PRIMARY_QC_EXCLUDE = (HARD_FAIL, SOFT_WARNING)  # PASS only
CAVEATS: list[str] = []


def _caveat(msg: str) -> None:
    print(f"  [CAVEAT] {msg}")
    CAVEATS.append(msg)


def _restrict_to_development(ds, dev_patient_ids: set):
    keep = np.isin(ds.patient_id, list(dev_patient_ids))
    if not keep.any():
        raise ValueError(f"no development patients found for instrument {ds.instrument.id!r} "
                          "-- check development_manifest.csv / instrument_id mapping.")
    ds.patient_id = ds.patient_id[keep]
    ds.legacy_mg_id = ds.legacy_mg_id[keep]
    ds.target = ds.target[keep]
    ds.spectra = ds.spectra[keep]
    return ds


def class_patient_matrix_with_ids(repr_, target_value):
    if target_value is None:
        mask = np.ones(len(repr_.patient_id), dtype=bool)
    else:
        mask = repr_.target == target_value
    sub = repr_.spectra[mask]
    valid = ~np.isnan(sub).any(axis=1)
    return sub[valid], repr_.patient_id[mask][valid]


def plot_overlay(waves, old_mean, new_before_mean, new_after_mean, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(waves, old_mean, color="#1f77b4", label="OLD (development)", linewidth=1.5)
    ax.plot(waves, new_before_mean, color="#d62728", label="NEW (development, before transfer)", linewidth=1.2,
            linestyle="--")
    if new_after_mean is not None:
        ax.plot(waves, new_after_mean, color="#2ca02c", label="NEW (development, after transfer)", linewidth=1.5)
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Area-normalized intensity")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_transfer_curve(waves, k_smoothed, ci_lo, ci_hi, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(1, color="grey", linewidth=0.8)
    ax.plot(waves, k_smoothed, color="#9467bd", linewidth=1.4, label="K(lambda), development-only")
    ax.fill_between(waves, ci_lo, ci_hi, color="#9467bd", alpha=0.15)
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("K(lambda) (empirical harmonization coefficient)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmonization-config", default=os.path.join(ROOT, "configs", "harmonization.yaml"))
    parser.add_argument("--qc-config", default=os.path.join(ROOT, "configs", "qc.yaml"))
    parser.add_argument("--development-manifest", default=os.path.join(STUDY_DIR, "development_manifest.csv"))
    args = parser.parse_args()

    if not os.path.exists(args.development_manifest):
        raise FileNotFoundError(
            f"{args.development_manifest} not found -- run scripts/prepare_blinded_study.py first (Part II must "
            "happen before any transfer-curve estimation, per Part X)."
        )

    os.makedirs(FIG_DIR, exist_ok=True)
    cfg = HarmonizationConfig.from_yaml(args.harmonization_config)
    qc_config = QCConfig.from_yaml(args.qc_config)

    dev_manifest = pd.read_csv(args.development_manifest, dtype={"patient_id": str})
    dev_by_role = {role: set(g["patient_id"]) for role, g in dev_manifest.groupby("instrument_role")}
    print(f"[dev manifest] old={len(dev_by_role.get('old', []))}, new_pd={len(dev_by_role.get('new_pd', []))}, "
          f"new_control={len(dev_by_role.get('new_control', []))} development patients")

    old_inst = InstrumentConfig.from_yaml(os.path.join(ROOT, cfg.instruments["old"].config))
    new_pd_inst = InstrumentConfig.from_yaml(os.path.join(ROOT, cfg.instruments["new_pd"].config))
    new_control_inst = InstrumentConfig.from_yaml(os.path.join(ROOT, cfg.instruments["new_control"].config))

    df_full = load_raw_table(old_inst.data)
    ds_old = _restrict_to_development(build_dataset(old_inst, df_full=df_full), dev_by_role.get("old", set()))
    ds_new_pd = _restrict_to_development(build_dataset(new_pd_inst, df_full=df_full), dev_by_role.get("new_pd", set()))
    ds_new_control = _restrict_to_development(build_dataset(new_control_inst, df_full=df_full),
                                               dev_by_role.get("new_control", set()))

    ref_waves = ds_old.waves
    print(f"[data] DEVELOPMENT-ONLY: old={ds_old.n_spectra} spectra/{len(np.unique(ds_old.patient_id))} patients, "
          f"new_pd={ds_new_pd.n_spectra}/{len(np.unique(ds_new_pd.patient_id))}, "
          f"new_control={ds_new_control.n_spectra}/{len(np.unique(ds_new_control.patient_id))}")

    n_new_pd_control = int((pd.Series(ds_new_pd.target) == 0.0).sum())
    if n_new_pd_control == 0:
        _caveat("new_pd development subset has ZERO control patients (by design of the dev/blind split, see "
                "prepare_blinded_study.py docstring) -- Part XI's PD-derived-vs-Control-derived comparison and "
                "Part XV's within-instrument PD-Control contrast check are NOT AVAILABLE for new_pd in this "
                "real-study rebuild. See outputs/final_study/instrument_transfer/ (Phase 1, full-cohort) for the "
                "exploratory version of these diagnostics.")

    # -- Part IV: reindex (no-op on this dataset; grids are identical) -------
    spectra_new_pd = reindex_to_reference_grid(ds_new_pd.spectra, ds_new_pd.waves, ref_waves)
    spectra_new_control = reindex_to_reference_grid(ds_new_control.spectra, ds_new_control.waves, ref_waves)

    # -- Part V: QC (development-only) ---------------------------------------
    def run_qc_and_correct(ds, spectra, waves):
        spectrum_id = [f"{pid}_{i}" for i, pid in enumerate(ds.patient_id)]
        qc_df, hp, _ = build_qc_table(spectra, waves, ds.patient_id, spectrum_id, qc_config)
        corrected, _ = correct_hot_pixels(spectra, hp, qc_config)
        return qc_df, corrected

    qc_old, corrected_old = run_qc_and_correct(ds_old, ds_old.spectra, ref_waves)
    qc_new_pd, corrected_new_pd = run_qc_and_correct(ds_new_pd, spectra_new_pd, ref_waves)
    qc_new_control, corrected_new_control = run_qc_and_correct(ds_new_control, spectra_new_control, ref_waves)
    print(f"[qc] old={qc_old['qc_status'].value_counts().to_dict()}, "
          f"new_pd={qc_new_pd['qc_status'].value_counts().to_dict()}, "
          f"new_control={qc_new_control['qc_status'].value_counts().to_dict()}")

    # -- Part VI: patient-level area-normalized representative spectra -------
    def build_repr(ds, corrected, qc_df):
        return build_patient_area_normalized_repr(corrected, ds.patient_id, ds.target, ref_waves,
                                                    qc_status=qc_df["qc_status"].to_numpy(),
                                                    exclude_statuses=PRIMARY_QC_EXCLUDE, method="median")

    repr_old = build_repr(ds_old, corrected_old, qc_old)
    repr_new_pd = build_repr(ds_new_pd, corrected_new_pd, qc_new_pd)
    repr_new_control = build_repr(ds_new_control, corrected_new_control, qc_new_control)

    n_boot = cfg.wavelength_alignment.n_bootstrap
    seed = cfg.random_seed
    sg_w, sg_p = cfg.savgol.window, cfg.savgol.polyorder
    search_range = tuple(cfg.wavelength_alignment.shift_search_range_nm)
    step = cfg.wavelength_alignment.shift_search_step_nm

    # -- Part VII/VIII: wavelength-shift diagnostics + PASS/WARNING/FAIL gate,
    #    development-only, target-blind (all development patients of each
    #    instrument, regardless of class) --------------------------------------
    old_all_matrix, _ = class_patient_matrix_with_ids(repr_old, None)
    new_pd_all_matrix, _ = class_patient_matrix_with_ids(repr_new_pd, None)
    new_control_all_matrix, _ = class_patient_matrix_with_ids(repr_new_control, None)
    old_all_mean = old_all_matrix.mean(axis=0)
    new_pd_all_mean = new_pd_all_matrix.mean(axis=0)
    new_control_all_mean = new_control_all_matrix.mean(axis=0)

    gates = {}
    shift_report = {}
    for inst_id, new_mean, new_matrix in [("new_pd", new_pd_all_mean, new_pd_all_matrix),
                                           ("new_control", new_control_all_mean, new_control_all_matrix)]:
        point = estimate_wavelength_shift(ref_waves, old_all_mean, new_mean, search_range, step, sg_w, sg_p)
        boot = patient_bootstrap_shift(old_all_matrix, new_matrix, ref_waves, n_boot=n_boot,
                                        seed=cfg.wavelength_alignment.bootstrap_seed, search_range_nm=search_range,
                                        step_nm=step, savgol_window=sg_w, savgol_polyorder=sg_p)
        gate = evaluate_shift_gate(point["delta_lambda_nm"], boot["ci_lo"], boot["ci_hi"],
                                    cfg.wavelength_alignment.max_allowed_shift_nm)
        gates[inst_id] = gate
        shift_report[inst_id] = {"point_estimate": point["delta_lambda_nm"], "best_correlation": point["best_correlation"],
                                  "bootstrap_median_nm": boot["median_delta_lambda_nm"], "ci_lo_nm": boot["ci_lo"],
                                  "ci_hi_nm": boot["ci_hi"], "gate": gate.to_dict()}
        print(f"[shift] {inst_id} (development-only): delta_lambda={point['delta_lambda_nm']:+.4f} nm, "
              f"bootstrap median={boot['median_delta_lambda_nm']:+.4f} [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}] "
              f"-> {gate.status}")
        if gate.status != PASS:
            _caveat(f"{inst_id} development-only wavelength-shift gate = {gate.status}: {gate.reason}")

    with open(os.path.join(OUT_DIR, "wavelength_shift_report.json"), "w", encoding="utf-8") as f:
        json.dump(shift_report, f, indent=2, ensure_ascii=False)

    # -- Part IX: similarity before transfer ----------------------------------
    sim_before = {
        "new_pd": compute_similarity_metrics(old_all_mean, new_pd_all_mean, ref_waves, sg_w, sg_p),
        "new_control": compute_similarity_metrics(old_all_mean, new_control_all_mean, ref_waves, sg_w, sg_p),
    }

    # -- Part X: transfer curve construction, DEVELOPMENT PATIENTS ONLY ------
    old_control_matrix, _ = class_patient_matrix_with_ids(repr_old, 0.0)
    old_pd_matrix, _ = class_patient_matrix_with_ids(repr_old, 1.0)
    new_control_control_matrix, _ = class_patient_matrix_with_ids(repr_new_control, 0.0)
    new_pd_pd_matrix, _ = class_patient_matrix_with_ids(repr_new_pd, 1.0)

    old_control_mean = old_control_matrix.mean(axis=0)
    old_pd_mean = old_pd_matrix.mean(axis=0)
    new_control_control_mean = new_control_control_matrix.mean(axis=0)
    new_pd_pd_mean = new_pd_pd_matrix.mean(axis=0)

    denom_thr = cfg.transfer_smoothing.denom_near_zero_rel_threshold
    edge_n = cfg.transfer_smoothing.edge_exclude_n_points
    sm_w, sm_p = cfg.transfer_smoothing.window, cfg.transfer_smoothing.polyorder

    def build_curve(curve_id, numerator_mean, denominator_mean, numerator_matrix, denominator_matrix, gate_status):
        if gate_status == FAIL:
            print(f"[transfer] {curve_id}: STOP TRANSFER (gate=FAIL, development-only) -- not built.")
            return None
        raw = build_transfer_curve_raw(numerator_mean, denominator_mean, denom_thr, edge_n)
        smoothed = smooth_transfer_curve(raw["k_raw"], sm_w, sm_p)
        boot_df = bootstrap_transfer_curve(numerator_matrix, denominator_matrix, denom_thr, edge_n, sm_w, sm_p,
                                            n_boot=n_boot, seed=seed)
        print(f"[transfer] {curve_id} (development-only): built. mean K={smoothed.mean():.3f}, std={smoothed.std():.3f}")
        return {"raw": raw["k_raw"], "smoothed": smoothed, "bootstrap": boot_df}

    curve_new_control = build_curve("new_control_to_old_control", old_control_mean, new_control_control_mean,
                                     old_control_matrix, new_control_control_matrix, gates["new_control"].status)
    curve_new_pd_pd = build_curve("new_pd_to_old_pd", old_pd_mean, new_pd_pd_mean, old_pd_matrix, new_pd_pd_matrix,
                                   gates["new_pd"].status)

    raw_rows, smoothed_rows, ci_rows = [], [], []
    for curve_id, curve in [("new_control_to_old_control", curve_new_control), ("new_pd_to_old_pd", curve_new_pd_pd)]:
        if curve is None:
            continue
        for w, kr, ks in zip(ref_waves, curve["raw"], curve["smoothed"]):
            raw_rows.append({"curve_id": curve_id, "wavelength": float(w), "k_raw": float(kr)})
            smoothed_rows.append({"curve_id": curve_id, "wavelength": float(w), "k_smoothed": float(ks)})
        for w, m, lo, hi, sd in zip(ref_waves, curve["bootstrap"]["bootstrap_median"], curve["bootstrap"]["ci_lo"],
                                     curve["bootstrap"]["ci_hi"], curve["bootstrap"]["bootstrap_std"]):
            ci_rows.append({"curve_id": curve_id, "wavelength": float(w), "bootstrap_median": float(m),
                             "ci95_lo": float(lo), "ci95_hi": float(hi), "bootstrap_std": float(sd)})
    pd.DataFrame(raw_rows).to_csv(os.path.join(OUT_DIR, "transfer_raw.csv"), index=False)
    pd.DataFrame(smoothed_rows).to_csv(os.path.join(OUT_DIR, "transfer_smoothed.csv"), index=False)
    pd.DataFrame(ci_rows).to_csv(os.path.join(OUT_DIR, "transfer_bootstrap_ci.csv"), index=False)

    # persist the actual K(lambda) arrays for downstream (freeze_final_model.py,
    # predict_blinded.py) consumption -- these are the FROZEN, fixed-per-instrument
    # functions (Part X: "фиксированная instrument-specific function").
    np.save(os.path.join(OUT_DIR, "wavelength_grid.npy"), ref_waves)
    transfer_functions_meta = {
        "reference_grid": {"min_nm": float(ref_waves.min()), "max_nm": float(ref_waves.max()),
                            "n_points": int(len(ref_waves))},
        "curves": {},
    }
    if curve_new_pd_pd is not None:
        np.save(os.path.join(OUT_DIR, "k_new_pd_to_old.npy"), curve_new_pd_pd["smoothed"])
        transfer_functions_meta["curves"]["new_pd"] = {
            "curve_id": "new_pd_to_old_pd", "built_from": "development PD patients only (Part X.B)",
            "n_old_pd_patients": int(len(old_pd_matrix)), "n_new_pd_pd_patients": int(len(new_pd_pd_matrix)),
            "gate_status": gates["new_pd"].status, "file": "k_new_pd_to_old.npy",
        }
    if curve_new_control is not None:
        np.save(os.path.join(OUT_DIR, "k_new_control_to_old.npy"), curve_new_control["smoothed"])
        transfer_functions_meta["curves"]["new_control"] = {
            "curve_id": "new_control_to_old_control", "built_from": "development control patients only (Part X.A)",
            "n_old_control_patients": int(len(old_control_matrix)),
            "n_new_control_control_patients": int(len(new_control_control_matrix)),
            "gate_status": gates["new_control"].status, "file": "k_new_control_to_old.npy",
        }
    with open(os.path.join(OUT_DIR, "transfer_functions_meta.json"), "w", encoding="utf-8") as f:
        json.dump(transfer_functions_meta, f, indent=2, ensure_ascii=False)
    print("  -> transfer_raw.csv, transfer_smoothed.csv, transfer_bootstrap_ci.csv, "
          "k_new_pd_to_old.npy, k_new_control_to_old.npy, transfer_functions_meta.json, wavelength_grid.npy")

    # -- Part XIII: apply transfer + re-normalize, development spectra -------
    def apply_to_instrument(ds, corrected, k_smoothed):
        out = np.full_like(corrected, np.nan)
        valid = ~np.isnan(corrected).any(axis=1)
        norm = np.full_like(corrected, np.nan)
        norm[valid] = area_normalize(corrected[valid], ref_waves)
        applied = apply_transfer(norm[valid], ref_waves, k_smoothed)
        out[valid] = applied["harmonized"]
        return out

    repr_new_pd_after, repr_new_control_after = None, None
    if curve_new_pd_pd is not None:
        harmonized_new_pd = apply_to_instrument(ds_new_pd, corrected_new_pd, curve_new_pd_pd["smoothed"])
        repr_new_pd_after = compute_patient_representative_spectra(
            harmonized_new_pd, ds_new_pd.patient_id, ds_new_pd.target, ref_waves,
            qc_status=qc_new_pd["qc_status"].to_numpy(), exclude_statuses=PRIMARY_QC_EXCLUDE, method="median")
    if curve_new_control is not None:
        harmonized_new_control = apply_to_instrument(ds_new_control, corrected_new_control,
                                                       curve_new_control["smoothed"])
        repr_new_control_after = compute_patient_representative_spectra(
            harmonized_new_control, ds_new_control.patient_id, ds_new_control.target, ref_waves,
            qc_status=qc_new_control["qc_status"].to_numpy(), exclude_statuses=PRIMARY_QC_EXCLUDE, method="median")

    # -- Part XIV: similarity after transfer ----------------------------------
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
                     "derivative_correlation"]
    sim_rows = []
    for inst_id in ["new_pd", "new_control"]:
        for m in metric_names:
            sim_rows.append({"instrument": inst_id, "metric": m, "before": sim_before[inst_id][m],
                              "after": sim_after.get(inst_id, {}).get(m, np.nan)})
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(os.path.join(OUT_DIR, "similarity_metrics_before_after.csv"), index=False)
    print(f"  -> similarity_metrics_before_after.csv")

    # -- Part XV: PD-Control contrast preservation (new_control only; new_pd
    #    has zero development control patients, see caveat above) -----------
    pd_control_contrast_report = {"new_pd": "NOT AVAILABLE (zero development control patients, see caveats)"}
    new_control_pd_matrix, _ = class_patient_matrix_with_ids(repr_new_control, 1.0)
    if len(new_control_pd_matrix) >= 1 and len(new_control_control_matrix) >= 1:
        d_old = pd_control_contrast(old_pd_matrix, old_control_matrix)
        d_new_control_before = pd_control_contrast(new_control_pd_matrix, new_control_control_matrix)
        compare_before = compare_contrast_curves(d_new_control_before["D"], d_old["D"], ref_waves)
        compare_after = None
        if repr_new_control_after is not None:
            pd_m, _ = class_patient_matrix_with_ids(repr_new_control_after, 1.0)
            ctrl_m, _ = class_patient_matrix_with_ids(repr_new_control_after, 0.0)
            if len(pd_m) and len(ctrl_m):
                d_new_control_after = pd_control_contrast(pd_m, ctrl_m)
                compare_after = compare_contrast_curves(d_new_control_after["D"], d_old["D"], ref_waves)
        pd_control_contrast_report["new_control"] = {
            "n_pd": d_new_control_before["n_pd"], "n_control": d_new_control_before["n_control"],
            "caveats": d_new_control_before["caveats"] + [
                "new_control has only n=2 development PD patients -- this contrast is exploratory/illustrative "
                "only (same caveat as Phase 1's full-cohort report), not a primary transfer-quality criterion."],
            "correlation_vs_old_before": compare_before["correlation"],
            "correlation_vs_old_after": compare_after["correlation"] if compare_after else None,
            "sign_agreement_before": compare_before["sign_agreement_fraction"],
            "sign_agreement_after": compare_after["sign_agreement_fraction"] if compare_after else None,
        }
    with open(os.path.join(OUT_DIR, "pd_control_contrast_report.json"), "w", encoding="utf-8") as f:
        json.dump(pd_control_contrast_report, f, indent=2, ensure_ascii=False)
    print(f"  -> pd_control_contrast_report.json")

    # -- Part XVI: instrument-separability diagnostic -------------------------
    sep_before = instrument_separability_auc(old_control_matrix, np.arange(len(old_control_matrix)).astype(str),
                                              new_control_control_matrix,
                                              np.arange(len(new_control_control_matrix)).astype(str), seed=seed)
    sep_after = None
    if repr_new_control_after is not None:
        after_ctrl_m, _ = class_patient_matrix_with_ids(repr_new_control_after, 0.0)
        if len(after_ctrl_m) >= 4:
            sep_after = instrument_separability_auc(old_control_matrix, np.arange(len(old_control_matrix)).astype(str),
                                                      after_ctrl_m, np.arange(len(after_ctrl_m)).astype(str), seed=seed)
    instrument_separability_report = {"before_transfer": sep_before, "after_transfer": sep_after}
    with open(os.path.join(OUT_DIR, "instrument_separability_report.json"), "w", encoding="utf-8") as f:
        json.dump(instrument_separability_report, f, indent=2, ensure_ascii=False)
    print(f"  -> instrument_separability_report.json: AUC before={sep_before['mean_auc']:.3f}"
          + (f", after={sep_after['mean_auc']:.3f}" if sep_after is not None else ", after=N/A"))

    # -- figures (subset -- full figure set already in Phase 1's report) -----
    plot_overlay(ref_waves, old_all_mean, new_pd_all_mean, new_pd_after_all_mean,
                 "OLD vs new_pd (development-only): mean spectrum", os.path.join(FIG_DIR, "overlay_new_pd_dev.png"))
    plot_overlay(ref_waves, old_all_mean, new_control_all_mean, new_control_after_all_mean,
                 "OLD vs new_control (development-only): mean spectrum",
                 os.path.join(FIG_DIR, "overlay_new_control_dev.png"))
    if curve_new_pd_pd is not None:
        plot_transfer_curve(ref_waves, curve_new_pd_pd["smoothed"], curve_new_pd_pd["bootstrap"]["ci_lo"],
                             curve_new_pd_pd["bootstrap"]["ci_hi"],
                             "K_new_pd->old (development-only, PD-derived)",
                             os.path.join(FIG_DIR, "transfer_curve_new_pd_dev.png"))
    if curve_new_control is not None:
        plot_transfer_curve(ref_waves, curve_new_control["smoothed"], curve_new_control["bootstrap"]["ci_lo"],
                             curve_new_control["bootstrap"]["ci_hi"],
                             "K_new_control->old (development-only, control-derived)",
                             os.path.join(FIG_DIR, "transfer_curve_new_control_dev.png"))

    # -- HTML report -----------------------------------------------------------
    write_html_report(shift_report, sim_df, pd_control_contrast_report, instrument_separability_report, gates)

    manifest_files = [
        os.path.join(OUT_DIR, "transfer_raw.csv"), os.path.join(OUT_DIR, "transfer_smoothed.csv"),
        os.path.join(OUT_DIR, "transfer_bootstrap_ci.csv"), os.path.join(OUT_DIR, "wavelength_grid.npy"),
        os.path.join(OUT_DIR, "transfer_functions_meta.json"),
    ]
    if curve_new_pd_pd is not None:
        manifest_files.append(os.path.join(OUT_DIR, "k_new_pd_to_old.npy"))
    if curve_new_control is not None:
        manifest_files.append(os.path.join(OUT_DIR, "k_new_control_to_old.npy"))
    write_manifest(manifest_files, os.path.join(OUT_DIR, "transfer_manifest.json"), base_dir=OUT_DIR,
                    extra={"caveats": CAVEATS})
    print(f"  -> transfer_manifest.json (SHA256 of all development-only transfer artifacts)")
    print("\nDone.")


def write_html_report(shift_report, sim_df, pd_control_contrast_report, instrument_separability_report, gates):
    def html_table(df):
        return df.to_html(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "")

    warnings_html = "".join(f"<li>{c}</li>" for c in CAVEATS) or "<li>none</li>"
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Development-only transfer rebuild (Phase 2, Part X)</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; }}
table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
.caveat {{ background: #fff3cd; border-left: 4px solid #e0a800; padding: 0.8em 1em; margin: 1em 0; }}
img {{ max-width: 48%; border: 1px solid #ddd; margin: 0.5em; }}
</style></head><body>
<h1>Development-only empirical instrument-harmonization rebuild (Phase 2)</h1>
<p>Generated by <code>scripts/rebuild_transfer_functions_development.py</code>. Terminology (Part XXXVII): these are
<b>empirical mean-spectrum instrument harmonization</b> coefficients, built using ONLY
<code>outputs/final_study/development_manifest.csv</code> patients (no blind-test spectra/labels touched anywhere in
this script). See <code>outputs/final_study/instrument_transfer/instrument_transfer_report.html</code> for the fuller
Phase 1 full-cohort exploratory report (figures, additional diagnostics).</p>

<div class="caveat"><b>Warnings / limitations</b><ul>{warnings_html}</ul></div>

<h2>Wavelength-shift gate (development-only)</h2>
<table><tr><th>instrument</th><th>delta_lambda (nm)</th><th>bootstrap CI</th><th>gate</th></tr>
{"".join(f"<tr><td>{k}</td><td>{v['point_estimate']:+.4f}</td><td>[{v['ci_lo_nm']:+.4f}, {v['ci_hi_nm']:+.4f}]</td><td>{v['gate']['status']}</td></tr>" for k, v in shift_report.items())}
</table>

<h2>Shape-similarity metrics before/after (development-only)</h2>
{html_table(sim_df)}

<h2>PD-Control contrast preservation</h2>
<pre>{json.dumps(pd_control_contrast_report, indent=2)}</pre>

<h2>Instrument separability (before/after)</h2>
<pre>AUC before={instrument_separability_report['before_transfer']['mean_auc']:.3f}, after={instrument_separability_report['after_transfer']['mean_auc'] if instrument_separability_report['after_transfer'] else 'N/A'}</pre>

<h2>Figures</h2>
<img src="figures/overlay_new_pd_dev.png"><img src="figures/overlay_new_control_dev.png">
<img src="figures/transfer_curve_new_pd_dev.png"><img src="figures/transfer_curve_new_control_dev.png">
</body></html>"""
    with open(os.path.join(OUT_DIR, "transfer_report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> transfer_report.html")


if __name__ == "__main__":
    main()
