"""Stage 9 entry point: patient-level bootstrap uncertainty on the OOF
patient_predictions table (no refitting -- cheap regardless of model cost).

Usage: python scripts/run_bootstrap.py configs/instrument_old.yaml
"""
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from spectra_ml.bootstrap import bootstrap_patient_metrics, paired_bootstrap_compare
from spectra_ml.schema import InstrumentConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instrument_config")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--prob-col", default="calibrated_probability")
    parser.add_argument("--compare", nargs="*", default=["logreg", "linear_svm", "pls_logreg"],
                         help="families to run pairwise paired-bootstrap comparisons on")
    args = parser.parse_args()

    instrument = InstrumentConfig.from_yaml(args.instrument_config)
    seed = instrument.data.random_seed

    pred_path = os.path.join(ROOT, "outputs", instrument.id, "predictions", "patient_predictions.csv")
    predictions = pd.read_csv(pred_path)
    print(f"[{instrument.id}] loaded {pred_path}: {predictions['family'].nunique()} families, "
          f"{predictions['patient_id'].nunique()} patients")

    out_dir = os.path.join(ROOT, "outputs", instrument.id, "bootstrap")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    all_summaries, by_family = [], {}
    for family_name, fam_df in predictions.groupby("family"):
        replicates, summary = bootstrap_patient_metrics(fam_df, prob_col=args.prob_col,
                                                          n_boot=args.n_boot, seed=seed)
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
                                           metric="roc_auc", n_boot=args.n_boot, seed=seed)
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
