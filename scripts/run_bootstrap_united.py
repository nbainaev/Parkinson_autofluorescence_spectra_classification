"""United-dataset bootstrap entry point: patient-level bootstrap CI + paired
comparisons on the OOF patient_predictions table produced by
run_calibration_united.py. Mirrors scripts/run_bootstrap.py.

Usage: python scripts/run_bootstrap_united.py
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from spectra_ml.bootstrap import bootstrap_patient_metrics, paired_bootstrap_compare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united")
RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--prob-col", default="calibrated_probability")
    parser.add_argument("--compare", nargs="*", default=["logreg", "linear_svm", "pls_logreg"],
                         help="families to run pairwise paired-bootstrap comparisons on")
    args = parser.parse_args()

    pred_path = os.path.join(OUT, "predictions", "patient_predictions.csv")
    predictions = pd.read_csv(pred_path)
    print(f"[united] loaded {pred_path}: {predictions['family'].nunique()} families, "
          f"{predictions['patient_id'].nunique()} patients")

    out_dir = os.path.join(OUT, "bootstrap")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    all_summaries, by_family = [], {}
    for family_name, fam_df in predictions.groupby("family"):
        replicates, summary = bootstrap_patient_metrics(fam_df, prob_col=args.prob_col,
                                                          n_boot=args.n_boot, seed=RANDOM_SEED)
        summary.insert(0, "family", family_name)
        all_summaries.append(summary)
        by_family[family_name] = fam_df
        replicates.to_csv(os.path.join(out_dir, f"replicates_{family_name}.csv"), index=False)

    summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_all.to_csv(os.path.join(out_dir, "bootstrap_summary.csv"), index=False)
    print(f"  bootstrap ({args.n_boot} iters x {predictions['family'].nunique()} families): "
          f"{time.time() - t0:.1f}s")

    print("\nROC-AUC 95% CI (patient-level, stratified bootstrap):")
    for _, row in summary_all[summary_all["metric"] == "roc_auc"].iterrows():
        print(f"  {row['family']:>12s}: {row['mean']:.3f}  [{row['ci95_lo']:.3f}, {row['ci95_hi']:.3f}]")

    compare_families = [f for f in args.compare if f in by_family]
    comparisons = []
    for fam_a, fam_b in itertools.combinations(compare_families, 2):
        result = paired_bootstrap_compare(by_family[fam_a], by_family[fam_b], prob_col=args.prob_col,
                                           metric="roc_auc", n_boot=args.n_boot, seed=RANDOM_SEED)
        result = {"family_a": fam_a, "family_b": fam_b, **result}
        comparisons.append(result)

    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        comp_df.to_csv(os.path.join(out_dir, "paired_comparisons.csv"), index=False)
        print("\nPaired bootstrap comparisons (ROC-AUC, family_a - family_b):")
        for r in comparisons:
            sig = "significant" if r["significant_at_0.05"] else "not significant"
            print(f"  {r['family_a']:>12s} vs {r['family_b']:<12s}: "
                  f"delta={r['mean_delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]  {sig}")

    print(f"\n  -> {out_dir}/bootstrap_summary.csv")
    print(f"  -> {out_dir}/paired_comparisons.csv")


if __name__ == "__main__":
    main()
