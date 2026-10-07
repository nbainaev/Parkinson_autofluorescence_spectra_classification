"""Build one diagnosis-independent empirical transfer per new instrument.

Updated design: QC/normalization and global instrument harmonization precede
the unified patient split. Targets are never used by this script.
"""
from __future__ import annotations

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from spectra_ml.data import build_dataset, load_raw_table
from spectra_ml.harmonization import (HarmonizationConfig, area_normalize,
    apply_transfer, bootstrap_transfer_curve, build_transfer_curve_raw, compute_similarity_metrics,
    estimate_wavelength_shift, evaluate_shift_gate, patient_bootstrap_shift,
    instrument_separability_auc, reindex_to_reference_grid, smooth_transfer_curve)
from spectra_ml.integrity import write_manifest
from spectra_ml.qc import HARD_FAIL, SOFT_WARNING, QCConfig, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig
from spectra_ml.statistics import compute_patient_representative_spectra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "outputs", "final_study_unified")
OUT = os.path.join(STUDY, "transfer")


def patient_repr(ds, spectra, waves, qc_cfg):
    ids = np.asarray([f"{ds.instrument.id}:{i}" for i in range(len(ds.patient_id))])
    qc, hp, _ = build_qc_table(spectra, waves, ds.patient_id, ids, qc_cfg)
    corrected, _ = correct_hot_pixels(spectra, hp, qc_cfg)
    normalized = np.full_like(corrected, np.nan)
    valid = ~np.isnan(corrected).any(axis=1)
    normalized[valid] = area_normalize(corrected[valid], waves)
    # Target values are carried by the generic container but never selected or
    # inspected: every patient contributes equally regardless of diagnosis.
    rep = compute_patient_representative_spectra(
        normalized, ds.patient_id, np.zeros(len(ds.patient_id)), waves,
        qc_status=qc.qc_status.to_numpy(), exclude_statuses=(HARD_FAIL, SOFT_WARNING), method="median")
    good = ~np.isnan(rep.spectra).any(axis=1)
    return rep.spectra[good], rep.patient_id[good], qc


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = HarmonizationConfig.from_yaml(os.path.join(ROOT, "configs", "harmonization.yaml"))
    qc_cfg = QCConfig.from_yaml(os.path.join(ROOT, "configs", "qc.yaml"))
    inst = {r: InstrumentConfig.from_yaml(os.path.join(ROOT, x.config)) for r, x in cfg.instruments.items()}
    raw = load_raw_table(inst["old"].data)
    ds = {r: build_dataset(v, df_full=raw) for r, v in inst.items()}
    waves = ds["old"].waves
    matrices, qc_counts = {}, {}
    for role in ("old", "new_pd", "new_control"):
        x = ds[role].spectra if role == "old" else reindex_to_reference_grid(ds[role].spectra, ds[role].waves, waves)
        matrices[role], _, qc = patient_repr(ds[role], x, waves, qc_cfg)
        qc_counts[role] = qc.qc_status.value_counts().to_dict()

    old_mean = matrices["old"].mean(axis=0)
    rows, meta, artifacts = [], {"method": "diagnosis-independent empirical mean-spectrum instrument harmonization",
        "constructed_before_patient_split": True, "targets_used": False, "qc_counts": qc_counts,
        "n_patient_representatives": {k: int(len(v)) for k, v in matrices.items()}, "curves": {}}, []
    np.save(os.path.join(OUT, "wavelength_grid.npy"), waves)
    artifacts.append(os.path.join(OUT, "wavelength_grid.npy"))
    for role in ("new_pd", "new_control"):
        new_mean = matrices[role].mean(axis=0)
        shift = estimate_wavelength_shift(waves, old_mean, new_mean,
            tuple(cfg.wavelength_alignment.shift_search_range_nm), cfg.wavelength_alignment.shift_search_step_nm,
            cfg.savgol.window, cfg.savgol.polyorder)
        boot_shift = patient_bootstrap_shift(matrices["old"], matrices[role], waves,
            cfg.wavelength_alignment.n_bootstrap, cfg.wavelength_alignment.bootstrap_seed,
            tuple(cfg.wavelength_alignment.shift_search_range_nm), cfg.wavelength_alignment.shift_search_step_nm,
            cfg.savgol.window, cfg.savgol.polyorder)
        gate = evaluate_shift_gate(shift["delta_lambda_nm"], boot_shift["ci_lo"], boot_shift["ci_hi"],
                                   cfg.wavelength_alignment.max_allowed_shift_nm)
        if gate.status == "FAIL":
            raise RuntimeError(f"{role} wavelength gate failed: {gate.reason}")
        raw_curve = build_transfer_curve_raw(old_mean, new_mean,
            cfg.transfer_smoothing.denom_near_zero_rel_threshold, cfg.transfer_smoothing.edge_exclude_n_points)
        curve = smooth_transfer_curve(raw_curve["k_raw"], cfg.transfer_smoothing.window,
                                      cfg.transfer_smoothing.polyorder)
        ci = bootstrap_transfer_curve(matrices["old"], matrices[role],
            cfg.transfer_smoothing.denom_near_zero_rel_threshold, cfg.transfer_smoothing.edge_exclude_n_points,
            cfg.transfer_smoothing.window, cfg.transfer_smoothing.polyorder,
            cfg.wavelength_alignment.n_bootstrap, cfg.random_seed)
        npy = os.path.join(OUT, f"k_{role}_to_old.npy"); np.save(npy, curve); artifacts.append(npy)
        after_matrix = apply_transfer(matrices[role], waves, curve)["harmonized"]
        before_sep = instrument_separability_auc(matrices["old"], np.arange(len(matrices["old"])).astype(str),
            matrices[role], np.arange(len(matrices[role])).astype(str), seed=cfg.random_seed)
        after_sep = instrument_separability_auc(matrices["old"], np.arange(len(matrices["old"])).astype(str),
            after_matrix, np.arange(len(after_matrix)).astype(str), seed=cfg.random_seed)
        for i, w in enumerate(waves):
            rows.append({"instrument_role": role, "wavelength": float(w), "k_raw": float(raw_curve["k_raw"][i]),
                         "k_smoothed": float(curve[i]), "bootstrap_median": float(ci.bootstrap_median.iloc[i]),
                         "ci95_lo": float(ci.ci_lo.iloc[i]), "ci95_hi": float(ci.ci_hi.iloc[i])})
        meta["curves"][role] = {"file": os.path.basename(npy), "n_old_patients": len(matrices["old"]),
            "n_new_patients": len(matrices[role]), "diagnosis_independent": True,
            "shift_nm": shift["delta_lambda_nm"], "shift_ci95": [boot_shift["ci_lo"], boot_shift["ci_hi"]],
            "gate": gate.to_dict(), "similarity_before": compute_similarity_metrics(old_mean, new_mean, waves,
                cfg.savgol.window, cfg.savgol.polyorder),
            "similarity_after": compute_similarity_metrics(old_mean, after_matrix.mean(axis=0), waves,
                cfg.savgol.window, cfg.savgol.polyorder),
            "instrument_separability_before": before_sep, "instrument_separability_after": after_sep}
    csv = os.path.join(OUT, "transfer_curves.csv"); pd.DataFrame(rows).to_csv(csv, index=False); artifacts.append(csv)
    smoothed_csv = os.path.join(OUT, "transfer_smoothed.csv")
    pd.DataFrame(rows).rename(columns={"instrument_role": "curve_id"})[
        ["curve_id", "wavelength", "k_smoothed"]].to_csv(smoothed_csv, index=False)
    artifacts.append(smoothed_csv)
    js = os.path.join(OUT, "transfer_functions_meta.json")
    with open(js, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2, ensure_ascii=False)
    artifacts.append(js)
    write_manifest(artifacts, os.path.join(OUT, "transfer_manifest.json"), base_dir=STUDY,
                   extra={"state": "GLOBAL_TRANSFER_FROZEN_BEFORE_UNIFIED_SPLIT"})
    print(json.dumps(meta["n_patient_representatives"], indent=2))
    print(f"Wrote diagnosis-independent transfer artifacts to {OUT}")


if __name__ == "__main__": main()
