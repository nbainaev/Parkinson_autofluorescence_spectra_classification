"""Stage 11-12 entry point: nested-CV + OOF Platt calibration for MLP and 1D-CNN,
reusing the exact same experiment infrastructure as the classical models
(sections 14-18). Writes to a SEPARATE predictions file so it doesn't clobber
the classical-model patient_predictions.csv from Stage 6-9.

Usage: python scripts/run_neural.py configs/instrument_old.yaml
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.data import build_dataset
from spectra_ml.experiment import run_all_families_nested_cv_calibrated
from spectra_ml.models.cnn1d import build_cnn1d_candidates
from spectra_ml.models.mlp import build_mlp_candidates
from spectra_ml.qc import QCConfig, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    instrument = InstrumentConfig.from_yaml(os.path.join(ROOT, "configs", "instrument_old.yaml"))
    qc_config = QCConfig.from_yaml(os.path.join(ROOT, "configs", "qc.yaml"))
    seed = instrument.data.random_seed

    dataset = build_dataset(instrument)
    spectrum_id = [f"{pid}_{i}" for i, pid in enumerate(dataset.patient_id)]
    qc_df, hp, _ = build_qc_table(dataset.spectra, dataset.waves, dataset.patient_id, spectrum_id, qc_config)
    corrected, _ = correct_hot_pixels(dataset.spectra, hp, qc_config)
    keep = (qc_df["qc_status"].to_numpy() != "HARD_FAIL") & ~np.isnan(dataset.target)
    spectra, target, patient_id = corrected[keep], dataset.target[keep], dataset.patient_id[keep]
    print(f"modeling set: {keep.sum()} spectra, {len(np.unique(patient_id))} patients")

    pool = build_mlp_candidates(seed=seed) + build_cnn1d_candidates(seed=seed)
    print(f"candidates: {[c.name for c in pool]}")

    t0 = time.time()
    predictions, fold_metrics = run_all_families_nested_cv_calibrated(pool, spectra, target, patient_id,
                                                                        dataset.waves, seed=seed)
    print(f"nested CV + calibration: {time.time() - t0:.1f}s")

    out_dir = os.path.join(ROOT, "outputs", instrument.id)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    predictions.to_csv(os.path.join(out_dir, "predictions", "neural_patient_predictions.csv"), index=False)
    fold_metrics.drop(columns=["chosen_params"]).to_csv(
        os.path.join(out_dir, "models", "neural_fold_metrics.csv"), index=False)

    for family_name, fam_df in fold_metrics.groupby("family"):
        print(f"  {family_name}: raw AUC {fam_df['raw_roc_auc'].mean():.3f}+/-{fam_df['raw_roc_auc'].std():.3f}  "
              f"calibrated AUC {fam_df['calibrated_roc_auc'].mean():.3f}+/-{fam_df['calibrated_roc_auc'].std():.3f}")

    print(f"\n  -> {out_dir}/predictions/neural_patient_predictions.csv")
    print(f"  -> {out_dir}/models/neural_fold_metrics.csv")


if __name__ == "__main__":
    main()
