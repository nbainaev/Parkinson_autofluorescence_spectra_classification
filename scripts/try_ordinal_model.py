"""Exploratory, single-shot test: does an ORDINAL model (proportional-odds /
cumulative-link ordinal logistic regression, respecting control<stage1<stage2
<stage3 as an ordered scale rather than 4 unordered nominal classes) do better
than the nominal multinomial-logistic head already frozen as `pls_mnlogreg`
(blind-test macro-OvR AUC 0.685)?

Reuses the EXACT same development/blind split, preprocessing (variant D area
normalization), and PLS(n_components=8) reduction already frozen for
`pls_mnlogreg` (loaded from outputs/blinded_multiclass/frozen/final_model_
pls_mnlogreg.joblib), so the only thing that differs is the classifier head:
ordinal proportional-odds instead of nominal multinomial. Not wired into the
full nested-CV/freeze/predict/unblind machinery -- this is a single fixed-
config exploratory fit to answer "is this worth pursuing further" before
investing in full pipeline integration.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from run_experiment_united_stage_v2 import CLASS_LABELS
from spectra_ml.evaluation import macro_ovr_auc, compute_patient_metrics_multiclass
from spectra_ml.qc import HARD_FAIL
from spectra_ml.united import build_united_dataset
from spectra_ml.united_stage import build_united_stage_dataset
from spectra_ml.blinding import load_blinded_test_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "outputs", "blinded_multiclass")
FROZEN = os.path.join(STUDY, "frozen")


def _load_dev_and_blind_spectra():
    ud = build_united_dataset(transfer_dir=os.path.join(FROZEN, "harmonization"))
    usd = build_united_stage_dataset(ud)
    keep = (usd.qc_status != HARD_FAIL) & ~np.isnan(usd.stage_class)
    spectra, target, patient_id, waves = usd.spectra[keep], usd.stage_class[keep], usd.patient_id[keep], usd.waves

    dev = pd.read_csv(os.path.join(STUDY, "development_manifest.csv"), dtype={"patient_id": str})
    dev_ids = set(dev["patient_id"])
    dev_mask = np.isin(patient_id, list(dev_ids))

    blind = load_blinded_test_manifest(os.path.join(STUDY, "blinded_test_manifest.csv"))
    identity = pd.read_csv(os.path.join(STUDY, "private", "prediction_identity_lookup.csv"),
                            dtype={"study_id": str, "real_patient_id": str})
    mapping = blind.merge(identity[["study_id", "real_patient_id"]], on="study_id", validate="one_to_one")
    blind_ids = set(mapping["real_patient_id"])
    blind_mask = np.isin(patient_id, list(blind_ids))

    key = pd.read_csv(os.path.join(STUDY, "private", "blinding_key.csv"), dtype={"real_patient_id": str})
    target_lookup = dict(zip(key["real_patient_id"].astype(str), key["target"]))

    return (spectra[dev_mask], target[dev_mask], patient_id[dev_mask],
            spectra[blind_mask], patient_id[blind_mask], target_lookup, waves)


def main():
    (X_dev, y_dev, pid_dev, X_blind, pid_blind, target_lookup, waves) = _load_dev_and_blind_spectra()

    bundle = joblib.load(os.path.join(FROZEN, "final_model_pls_mnlogreg.joblib"))
    prep, pls_step = bundle["preprocessing"], bundle["model"].named_steps["pls"]

    Xt_dev = prep.transform(X_dev)
    Xt_blind = prep.transform(X_blind)

    # Refit PLS(n=8) on development only (same component count as the frozen
    # nominal model, for a fair head-to-head; PLS itself is already somewhat
    # ordinal-aware since it's fit against the raw integer stage value as a
    # continuous regression target -- see build_pls_multinomial_logreg_candidates).
    from spectra_ml.reduction import PLSFeatureExtractor
    pls = PLSFeatureExtractor(n_components=pls_step.n_components)
    Zdev = pls.fit_transform(Xt_dev, y_dev)
    Zblind = pls.transform(Xt_blind)

    # Patient-level: aggregate spectra to one representative feature vector per
    # patient (mean of that patient's spectra in PLS space) so the ordinal
    # model is fit/evaluated at the same patient-level granularity as every
    # other model in this comparison (avoids pseudo-replication across a
    # patient's repeat spectra).
    def to_patient_level(Z, pid, y=None):
        df = pd.DataFrame(Z, columns=[f"z{i}" for i in range(Z.shape[1])])
        df["pid"] = pid
        agg = df.groupby("pid", sort=False).mean()
        if y is not None:
            yy = pd.Series(y, index=pid).groupby(level=0).first()
            return agg, yy.loc[agg.index].to_numpy()
        return agg

    Zdev_p, y_dev_p = to_patient_level(Zdev, pid_dev, y_dev)
    Zblind_p = to_patient_level(Zblind, pid_blind)
    blind_pids = Zblind_p.index.to_numpy()
    y_blind_p = np.array([target_lookup[str(p)] for p in blind_pids], dtype=float)

    print(f"development patients: {len(Zdev_p)}, blind patients: {len(Zblind_p)}")
    print(f"dev class counts: {pd.Series(y_dev_p).value_counts().sort_index().to_dict()}")
    print(f"blind class counts: {pd.Series(y_blind_p).value_counts().sort_index().to_dict()}")

    # Proportional-odds ordinal logistic regression (cumulative link model).
    mod = OrderedModel(y_dev_p, Zdev_p.to_numpy(), distr="logit")
    res = mod.fit(method="bfgs", disp=False, maxiter=500)

    probs_blind = res.model.predict(res.params, exog=Zblind_p.to_numpy())  # (n, 4), columns in sorted-class order
    probs_blind = np.asarray(probs_blind)

    class_labels = sorted(np.unique(y_dev_p))
    pred_class = np.array(class_labels)[np.argmax(probs_blind, axis=1)]

    auc = macro_ovr_auc(y_blind_p, probs_blind, class_labels)
    metrics = compute_patient_metrics_multiclass(y_blind_p, probs_blind, class_labels)

    print("\n=== ORDINAL (proportional-odds) model, blind test ===")
    print(f"macro-OvR AUC: {auc:.3f}")
    print(f"balanced accuracy: {metrics.get('balanced_accuracy', float('nan')):.3f}")
    cm = pd.crosstab(pd.Series(y_blind_p, name="true"), pd.Series(pred_class, name="pred"))
    print(cm)
    print("\nFor comparison, frozen pls_mnlogreg (nominal multinomial) blind AUC = 0.685 (balanced_acc known from prior report)")

    out = {
        "model": "ordinal_proportional_odds", "macro_ovr_auc": float(auc),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", float("nan"))),
        "n_dev": len(Zdev_p), "n_blind": len(Zblind_p),
        "confusion_matrix": cm.to_dict(),
    }
    import json
    with open(os.path.join(STUDY, "ordinal_model_exploratory_result.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {os.path.join(STUDY, 'ordinal_model_exploratory_result.json')}")


if __name__ == "__main__":
    main()
