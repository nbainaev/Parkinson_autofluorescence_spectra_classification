"""Freeze all development-only model artifacts before blinded prediction."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import yaml

from spectra_ml.evaluation import youden_threshold
from spectra_ml.experiment import select_calibrate_and_refit
from spectra_ml.final_study import prepare_development
from spectra_ml.integrity import write_manifest
from spectra_ml.models.classical import (build_linear_svm_candidates, build_logreg_candidates,
                                          build_pls_logreg_candidates)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.environ.get("SPECTRA_STUDY_DIR", os.path.join(ROOT, "outputs", "final_study"))


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNAVAILABLE"


def _freeze_set(out_dir, x, y, pid, waves, pools, protocol, seed):
    os.makedirs(out_dir, exist_ok=True)
    artifacts = []
    summaries = {}
    for family, (candidates, calibrate) in pools.items():
        result = select_calibrate_and_refit(family, candidates, x, y, pid, waves, seed=seed,
                                             calibrate=calibrate)
        oof_col = "calibrated_probability" if calibrate else "raw_probability"
        threshold = youden_threshold(result["oof_patient_predictions"]["target"].to_numpy(),
                                     result["oof_patient_predictions"][oof_col].to_numpy())
        bundle = {k: result[k] for k in ("family", "preprocessing", "model", "calibrator")}
        bundle["threshold"] = threshold
        bundle["chosen_name"] = result["chosen_candidate"].name
        path = os.path.join(out_dir, f"final_model_{family}.joblib")
        joblib.dump(bundle, path)
        artifacts.append(path)
        oof_path = os.path.join(out_dir, f"development_oof_{family}.csv")
        result["oof_patient_predictions"].to_csv(oof_path, index=False)
        artifacts.append(oof_path)
        summaries[family] = {"chosen_candidate": result["chosen_candidate"].name,
                             "chosen_params": result["chosen_candidate"].params,
                             "calibrated": calibrate, "threshold": threshold,
                             "selection_mean_auc": result["selection_mean_auc"],
                             "selection_fold_aucs": result["selection_fold_aucs"]}
    np.save(os.path.join(out_dir, "wavelength_grid.npy"), waves)
    artifacts.append(os.path.join(out_dir, "wavelength_grid.npy"))
    with open(os.path.join(out_dir, "final_protocol.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({**protocol, "selected_models": summaries}, f, sort_keys=False)
    artifacts.append(os.path.join(out_dir, "final_protocol.yaml"))
    return artifacts, summaries


def main():
    config_path = os.path.join(ROOT, "configs", "final_study.yaml")
    with open(config_path, encoding="utf-8") as f:
        protocol = yaml.safe_load(f)
    seed = int(protocol["random_seed"])
    common = (ROOT, os.path.join(ROOT, "configs", "harmonization.yaml"), os.path.join(ROOT, "configs", "qc.yaml"),
              os.path.join(STUDY, "development_manifest.csv"), os.path.join(STUDY, "transfer"))
    x, y, pid, role, waves, _ = prepare_development(*common, use_transfer=True)
    x_no, y_no, pid_no, _, _, _ = prepare_development(*common, use_transfer=False)
    old = role == "old"
    pools = {
        "linear_svm": (build_linear_svm_candidates(("E",), (0.1, 1.0, 10.0), seed), True),
        "pls_logreg": (build_pls_logreg_candidates(("E",), (3, 8), (1.0,), seed), False),
        "logreg": (build_logreg_candidates(("E",), (0.1, 1.0, 10.0), seed), False),
    }
    all_artifacts = []
    frozen = os.path.join(STUDY, "frozen")
    a, summaries = _freeze_set(frozen, x, y, pid, waves, pools, protocol, seed); all_artifacts += a
    svm_pool = {"linear_svm": pools["linear_svm"]}
    a, _ = _freeze_set(os.path.join(STUDY, "frozen_no_transfer"), x_no, y_no, pid_no, waves,
                       svm_pool, {**protocol, "instrument_transfer": False}, seed); all_artifacts += a
    a, _ = _freeze_set(os.path.join(STUDY, "frozen_old_only"), x[old], y[old], pid[old], waves,
                       svm_pool, {**protocol, "training_roles": ["old"]}, seed); all_artifacts += a

    # Candidate regions are preregistered hypotheses plus development-only coefficient regions.
    svm = joblib.load(os.path.join(frozen, "final_model_linear_svm.joblib"))
    coef = np.abs(np.asarray(svm["model"].coef_).ravel())
    cutoff = np.percentile(coef, protocol["feature_regions"]["importance_percentile"])
    mask = coef >= cutoff
    runs, i = [], 0
    while i < len(mask):
        if not mask[i]: i += 1; continue
        j = i + 1
        while j < len(mask) and mask[j]: j += 1
        runs.append({"wavelength_start": float(waves[i]), "wavelength_end": float(waves[j-1]),
                     "mean_importance": float(coef[i:j].mean()), "source": "development_linear_svm"})
        i = j
    regions = {"locked_before_blind_prediction": True, "settings": protocol["feature_regions"],
               "pre_existing_hypotheses": [{"wavelength_start": a, "wavelength_end": b,
                                              "source": "old_exploratory_hypothesis"}
                                             for a, b in protocol["feature_regions"]["pre_existing_hypotheses_nm"]],
               "development_candidates": runs}
    feature_path = os.path.join(frozen, "feature_regions.json")
    with open(feature_path, "w", encoding="utf-8") as f: json.dump(regions, f, indent=2)
    all_artifacts.append(feature_path)
    for source, name in [(config_path, "final_study_config.yaml"),
                         (os.path.join(STUDY, "transfer", "transfer_smoothed.csv"), "transfer_functions.csv"),
                         (os.path.join(STUDY, "transfer", "k_new_pd_to_old.npy"), "k_new_pd_to_old.npy"),
                         (os.path.join(STUDY, "transfer", "k_new_control_to_old.npy"), "k_new_control_to_old.npy"),
                         (os.path.join(STUDY, "transfer", "transfer_functions_meta.json"), "transfer_functions_meta.json")]:
        dest = os.path.join(frozen, name)
        with open(source, "rb") as src, open(dest, "wb") as dst: dst.write(src.read())
        all_artifacts.append(dest)
    for name, content in [("git_commit.txt", _git_commit() + "\n"),
                          ("environment.txt", f"python={sys.version}\nplatform={platform.platform()}\n")]:
        path = os.path.join(frozen, name)
        with open(path, "w", encoding="utf-8") as f: f.write(content)
        all_artifacts.append(path)
    write_manifest(all_artifacts, os.path.join(frozen, "frozen_manifest.json"), base_dir=STUDY,
                   extra={"state": "FROZEN_BEFORE_BLIND_PREDICTION", "models": summaries})
    print(f"Frozen {len(all_artifacts)} artifacts; manifest: {os.path.join(frozen, 'frozen_manifest.json')}")


if __name__ == "__main__": main()
