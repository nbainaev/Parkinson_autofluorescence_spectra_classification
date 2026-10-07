"""Ad hoc: nested CV for just the tree/GBM families (hgb fixed, random_forest,
xgboost, lightgbm, catboost) on instrument='old', skipping the already-validated
linear families to save time.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.data import build_dataset
from spectra_ml.experiment import run_all_families_nested_cv
from spectra_ml.models.classical import (build_extra_trees_candidates, build_gradient_boosting_candidates,
                                          build_random_forest_candidates, gbm_candidate_pool)
from spectra_ml.qc import QCConfig, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

instrument = InstrumentConfig.from_yaml(os.path.join(ROOT, "configs", "instrument_old.yaml"))
qc_config = QCConfig.from_yaml(os.path.join(ROOT, "configs", "qc.yaml"))
seed = instrument.data.random_seed

dataset = build_dataset(instrument)
spectrum_id = [f"{pid}_{i}" for i, pid in enumerate(dataset.patient_id)]
qc_df, hp, _ = build_qc_table(dataset.spectra, dataset.waves, dataset.patient_id, spectrum_id, qc_config)
corrected, _ = correct_hot_pixels(dataset.spectra, hp, qc_config)

keep = (qc_df["qc_status"].to_numpy() != "HARD_FAIL") & ~np.isnan(dataset.target)
spectra, target, patient_id = corrected[keep], dataset.target[keep], dataset.patient_id[keep]

pool = (
    build_gradient_boosting_candidates(seed=seed)     # now with class_weight='balanced' fix
    + build_random_forest_candidates(seed=seed)        # new
    + gbm_candidate_pool(seed=seed)                     # xgboost + lightgbm + catboost
)
print(f"families: {sorted({c.family for c in pool})} ({len(pool)} candidates)")

t0 = time.time()
predictions, fold_metrics = run_all_families_nested_cv(pool, spectra, target, patient_id, dataset.waves, seed=seed)
print(f"nested CV: {time.time() - t0:.1f}s")

summary = fold_metrics.groupby("family")[["roc_auc", "pr_auc", "sensitivity", "specificity"]].agg(["mean", "std"])
print(summary)

out_dir = os.path.join(ROOT, "outputs", "old", "models")
fold_metrics.drop(columns=["chosen_params"]).to_csv(os.path.join(out_dir, "gbm_check_fold_metrics.csv"), index=False)
print(f"chosen configs:\n{fold_metrics[['family', 'outer_fold', 'chosen_model', 'preprocessing', 'roc_auc']]}")
