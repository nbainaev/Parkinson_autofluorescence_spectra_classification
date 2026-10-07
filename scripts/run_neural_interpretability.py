"""Stage 13 entry point: Integrated Gradients + GradientSHAP cross-check for
CNN1D (mlp skipped -- raw AUC 0.593+/-0.110 too close to chance to interpret).

Reuses the per-outer-fold model selection already done in Stage 11-12
(outputs/old/models/neural_fold_metrics.csv: chosen_model/preprocessing per
fold) instead of re-running inner CV -- that selection doesn't change, so
redoing it here was pure wasted compute (the first version of this script took
>45 min and was killed for exactly that reason). Only refits (5 outer folds)
plus attribution computation happen here.

Scope, reduced from the original 3-baseline-IG plan to keep runtime sane:
IG uses one baseline (train-mean spectrum, the standard choice) at steps=50;
GradientSHAP (baselines = a training-fold subsample, n_samples=30) as the
independent cross-check requested instead of the other two IG baselines.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from spectra_ml.data import build_dataset
from spectra_ml.evaluation import patient_balanced_sample_weight
from spectra_ml.interpretability import fold_stability
from spectra_ml.models.cnn1d import build_cnn1d_candidates
from spectra_ml.neural_interpretability import (aggregate_attribution_to_patient, compute_baselines,
                                                  global_attribution_summary, gradient_shap_attribution,
                                                  integrated_gradients)
from spectra_ml.preprocessing import build_preprocessing_pipeline
from spectra_ml.qc import QCConfig, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig
from spectra_ml.splitting import nested_group_stratified_splits

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
    spectra, target, patient_id, waves = corrected[keep], dataset.target[keep], dataset.patient_id[keep], dataset.waves

    chosen = pd.read_csv(os.path.join(ROOT, "outputs", "old", "models", "neural_fold_metrics.csv"))
    chosen = chosen[chosen["family"] == "cnn1d"].set_index("outer_fold")
    candidates_by_name = {c.name: c for c in build_cnn1d_candidates(seed=seed)}

    nested = nested_group_stratified_splits(patient_id, target, seed=seed)

    out_dir = os.path.join(ROOT, "outputs", "old", "interpretability")
    os.makedirs(out_dir, exist_ok=True)

    ig_mean_parts, ig_abs_parts = [], []
    gs_mean_parts, gs_abs_parts = [], []

    t0 = time.time()
    for outer in nested:
        row = chosen.loc[outer.outer_fold_index]
        best = candidates_by_name[row["chosen_model"]]

        X_tr, y_tr, pid_tr = spectra[outer.outer_train_idx], target[outer.outer_train_idx], patient_id[outer.outer_train_idx]
        X_te, y_te, pid_te = spectra[outer.outer_test_idx], target[outer.outer_test_idx], patient_id[outer.outer_test_idx]

        prep = build_preprocessing_pipeline(best.preprocessing_variant, wavelengths=waves)
        Xt_tr = prep.fit_transform(X_tr)
        Xt_te = prep.transform(X_te)

        model = best.build_fn()
        sw = patient_balanced_sample_weight(pid_tr)
        model.fit(Xt_tr, y_tr, sample_weight=sw, patient_id=pid_tr)

        train_mean = compute_baselines(Xt_tr, n_wave=len(waves))["train_mean"]
        ig_attr = integrated_gradients(model.module_, Xt_te, train_mean, steps=50, reshape_for_conv=True)
        ig_mean_df, ig_abs_df = aggregate_attribution_to_patient(ig_attr, pid_te)
        ig_mean_parts.append(ig_mean_df)
        ig_abs_parts.append(ig_abs_df)

        rng = np.random.default_rng(seed + outer.outer_fold_index)
        baseline_samples = Xt_tr[rng.choice(len(Xt_tr), min(30, len(Xt_tr)), replace=False)]
        gs_attr = gradient_shap_attribution(model.module_, Xt_te, baseline_samples, n_samples=30,
                                             seed=seed + outer.outer_fold_index, reshape_for_conv=True)
        gs_mean_df, gs_abs_df = aggregate_attribution_to_patient(gs_attr, pid_te)
        gs_mean_parts.append(gs_mean_df)
        gs_abs_parts.append(gs_abs_df)

        print(f"  outer_fold {outer.outer_fold_index}: reused chosen={best.name} "
              f"(t={time.time() - t0:.1f}s)")

    print(f"IG + GradientSHAP (5 folds): {time.time() - t0:.1f}s")

    def save(mean_parts, abs_parts, tag):
        patient_mean = pd.concat(mean_parts).groupby(level=0).mean().to_numpy()
        patient_abs = pd.concat(abs_parts).groupby(level=0).mean().to_numpy()
        summary = global_attribution_summary(patient_mean, patient_abs, waves)
        summary.to_csv(os.path.join(out_dir, f"{tag}_cnn1d.csv"), index=False)
        per_fold_abs = np.stack([df.mean(axis=0).to_numpy() for df in abs_parts])
        stability = fold_stability(per_fold_abs, waves, top_k=20)
        stability.to_csv(os.path.join(out_dir, f"{tag}_stability_cnn1d.csv"), index=False)
        return summary

    ig_summary = save(ig_mean_parts, ig_abs_parts, "integrated_gradients_train_mean")
    gs_summary = save(gs_mean_parts, gs_abs_parts, "gradient_shap")

    corr = np.corrcoef(ig_summary["mean_abs_attribution"], gs_summary["mean_abs_attribution"])[0, 1]
    print(f"IG vs GradientSHAP |attribution| correlation across wavelengths: {corr:.3f}")

    print(f"\n  -> {out_dir}/integrated_gradients_train_mean_cnn1d.csv")
    print(f"  -> {out_dir}/gradient_shap_cnn1d.csv")


if __name__ == "__main__":
    main()
