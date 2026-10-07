"""Verify frozen artifacts/predictions, then unblind once and report endpoints."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from spectra_ml.blinding import load_blinding_key
from spectra_ml.evaluation import compute_patient_metrics
from spectra_ml.integrity import sha256_file, verify_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "outputs", "final_study")


def bootstrap_auc(y, p, n_boot=2000, seed=42):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    rng, values = np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2: values.append(roc_auc_score(y[idx], p[idx]))
    return {"median": float(np.median(values)), "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)), "valid_replicates": len(values)} if values else None


def main():
    frozen_manifest = os.path.join(STUDY, "frozen", "frozen_manifest.json")
    prediction_manifest = os.path.join(STUDY, "blinded_predictions_manifest.json")
    verify_manifest(frozen_manifest, base_dir=STUDY)
    verify_manifest(prediction_manifest, base_dir=STUDY)
    with open(prediction_manifest, encoding="utf-8") as f: pred_meta = json.load(f)
    if pred_meta["frozen_manifest_sha256"] != sha256_file(frozen_manifest):
        raise RuntimeError("Frozen manifest itself changed after prediction.")

    pred = pd.read_csv(os.path.join(STUDY, "blinded_predictions.csv"), dtype={"study_id": str})
    key = load_blinding_key(os.path.join(STUDY, "private", "blinding_key.csv"))
    evaluated = pred.merge(key[["study_id", "target", "cohort"]], on="study_id", validate="many_to_one")
    evaluated.to_csv(os.path.join(STUDY, "unblinded_predictions.csv"), index=False)
    reports = []
    cohorts = {"primary_new_pd": "primary_new_pd_blind_test", "secondary_old": "secondary_old_holdout",
               "pooled": None}
    for (branch, model), model_df in evaluated.groupby(["analysis_branch", "model"]):
        for label, cohort in cohorts.items():
            sub = model_df if cohort is None else model_df[model_df.cohort == cohort]
            if sub.empty: continue
            metrics = compute_patient_metrics(sub.target.to_numpy(), sub.predicted_probability_PD.to_numpy(),
                                              threshold=float(sub.frozen_threshold.iloc[0]))
            reports.append({"analysis_branch": branch, "model": model, "cohort": label, **metrics,
                            "roc_auc_bootstrap": bootstrap_auc(sub.target, sub.predicted_probability_PD)})
    with open(os.path.join(STUDY, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, allow_nan=True)
    table = pd.json_normalize(reports)
    table.to_csv(os.path.join(STUDY, "evaluation_metrics.csv"), index=False)
    primary = table[(table.analysis_branch == "harmonized") & (table.cohort == "primary_new_pd")]
    html = """<!doctype html><html><head><meta charset='utf-8'><title>Final blinded PD study</title>
<style>body{font-family:sans-serif;max-width:1200px;margin:2em auto}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.35em}</style></head><body>
<h1>Final retrospective blinded validation study</h1>
<p>All frozen artifacts and predictions passed SHA256 verification before the private target key was read. Models, calibration, thresholds, transfer curves, and feature regions were fixed using development data only.</p>
<p><strong>Addition-compliant reproducibility rerun:</strong> empirical harmonization used every eligible patient in each full DEVELOPMENT clinical subgroup after predefined QC. No separate bridge, classifier-training, or calibration patient subsets were carved out. Patients contributing to the global harmonization artifact remained eligible for diagnostic-model training and development cross-validation. Because this dataset had already been unblinded in the earlier completed run, this rerun confirms computational reproducibility but is not a second independent blinded validation.</p>
<h2>Pipeline artifact scope</h2>
<ul>
<li><strong>Global DEVELOPMENT artifacts:</strong> fixed patient-balanced mean-spectrum transfer coefficients, wavelength grid, QC policy, and protocol configuration, all estimated without final-test patients.</li>
<li><strong>Fold-specific DEVELOPMENT artifacts:</strong> supervised model fits and preprocessing transforms used for hyperparameter scoring and OOF predictions; held-out fold patients were not used to fit those supervised components.</li>
<li><strong>Frozen final artifacts:</strong> refitted preprocessing/models, OOF Platt calibrator where specified, development-selected thresholds, transfer coefficients, feature regions, and integrity hashes used unchanged for test prediction.</li>
</ul>
<h2>Primary endpoint: new_pd blinded patients</h2>""" + primary.to_html(index=False) + "<h2>All preregistered analyses</h2>" + table.to_html(index=False) + """
<h2>Interpretation constraints</h2><p>The primary cohort contains only one control patient, so ROC-AUC is computable but its uncertainty is necessarily extreme. OLD hold-out is secondary because OLD data informed prior exploration. Transfer is empirical mean-spectrum shape harmonization, not physical intensity calibration.</p>
</body></html>"""
    with open(os.path.join(STUDY, "final_blinded_study_report.html"), "w", encoding="utf-8") as f: f.write(html)
    print(f"Unblinded {evaluated.study_id.nunique()} patients after integrity verification; report written.")


if __name__ == "__main__": main()
