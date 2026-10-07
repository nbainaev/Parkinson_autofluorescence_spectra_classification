"""Blind-safe data preparation and prediction helpers for the final study."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from spectra_ml.data import build_dataset, load_raw_table
from spectra_ml.harmonization import HarmonizationConfig, apply_transfer, area_normalize, reindex_to_reference_grid
from spectra_ml.qc import HARD_FAIL, QCConfig, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig


def load_instruments(root: str, harmonization_config: str):
    cfg = HarmonizationConfig.from_yaml(harmonization_config)
    instruments = {role: InstrumentConfig.from_yaml(os.path.join(root, item.config))
                   for role, item in cfg.instruments.items()}
    return cfg, instruments


def load_transfer_artifacts(transfer_dir: str):
    waves = np.load(os.path.join(transfer_dir, "wavelength_grid.npy"))
    curves = {
        "new_pd": np.load(os.path.join(transfer_dir, "k_new_pd_to_old.npy")),
        "new_control": np.load(os.path.join(transfer_dir, "k_new_control_to_old.npy")),
    }
    with open(os.path.join(transfer_dir, "transfer_functions_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return waves, curves, meta


def prepare_role_dataset(role: str, patient_ids: set[str], instruments: dict, df_full: pd.DataFrame,
                         ref_waves: np.ndarray, curves: dict[str, np.ndarray], qc_config: QCConfig,
                         use_transfer: bool = True):
    """QC, interpolate once, area-normalize, optionally transfer and re-normalize."""
    ds = build_dataset(instruments[role], df_full=df_full)
    keep_patient = np.isin(ds.patient_id.astype(str), list(patient_ids))
    if not keep_patient.any():
        raise ValueError(f"No requested patients found for role {role!r}")
    spectra = reindex_to_reference_grid(ds.spectra[keep_patient], ds.waves, ref_waves)
    pid = ds.patient_id[keep_patient].astype(str)
    target = ds.target[keep_patient]
    spectrum_id = np.asarray([f"{role}:{p}:{i}" for i, p in enumerate(pid)])
    qc, hp, _ = build_qc_table(spectra, ref_waves, pid, spectrum_id, qc_config)
    corrected, _ = correct_hot_pixels(spectra, hp, qc_config)
    usable = (qc["qc_status"].to_numpy() != HARD_FAIL) & ~np.isnan(corrected).any(axis=1)
    x = area_normalize(corrected[usable], ref_waves)
    if use_transfer and role != "old":
        x = apply_transfer(x, ref_waves, curves[role])["harmonized"]
    return x, target[usable], pid[usable], qc.loc[usable].reset_index(drop=True)


def prepare_development(root: str, harmonization_config: str, qc_config_path: str,
                        development_manifest_path: str, transfer_dir: str, use_transfer: bool = True,
                        roles=("old", "new_pd", "new_control")):
    cfg, instruments = load_instruments(root, harmonization_config)
    ref_waves, curves, _ = load_transfer_artifacts(transfer_dir)
    qc_config = QCConfig.from_yaml(qc_config_path)
    manifest = pd.read_csv(development_manifest_path, dtype={"patient_id": str})
    df_full = load_raw_table(instruments["old"].data)
    parts = []
    for role in roles:
        ids = set(manifest.loc[manifest.instrument_role == role, "patient_id"])
        x, y, pid, qc = prepare_role_dataset(role, ids, instruments, df_full, ref_waves, curves, qc_config,
                                               use_transfer=use_transfer)
        parts.append((x, y, pid, np.repeat(role, len(y)), qc))
    return (np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]).astype(int),
            np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]), ref_waves, cfg)


def raw_model_probability(bundle: dict, x: np.ndarray) -> np.ndarray:
    xt = bundle["preprocessing"].transform(x)
    model = bundle["model"]
    if hasattr(model, "decision_function"):
        score = model.decision_function(xt)
        raw = 1.0 / (1.0 + np.exp(-np.asarray(score)))
    else:
        raw = model.predict_proba(xt)[:, 1]
    calibrator = bundle.get("calibrator")
    return calibrator.predict_proba(raw) if calibrator is not None else raw
