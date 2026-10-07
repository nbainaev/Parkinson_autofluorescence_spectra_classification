"""Task 2 (v2): multiclass (4-class Hoehn-Yahr PD stage) pooled statistics --
Kruskal-Wallis omnibus test (unchanged per-wavelength FDR-BH convention) +
pairwise Mann-Whitney (tie-corrected z-score, Hedges' g), on the CORRECTED
(post-identity-fix, re-harmonized via outputs/harmonization_v2/) dataset.

Pairwise Mann-Whitney is reported under BOTH conventions for comparison:
  - per-pair FDR (pairwise_mannwhitney_multiclass): kept as-is (matches the
    stale outputs/united_multiclass run's convention), for a direct before/
    after comparison in the report.
  - JOINT FDR across all 6 pairs x all wavelengths pooled into one family
    (pairwise_mannwhitney_multiclass_joint_fdr) -- the convention the user
    explicitly asked to fix for this task; THIS is the primary result.

Usage: python scripts/run_statistics_united_stage_v2.py [--normalize none|snv|area] [--method median|mean]

`--normalize area` (added for the area-normalization redo of this task, additive: `none`/`snv`
still work unchanged) applies `spectra_ml.preprocessing.AreaNormalizer` per-spectrum -- the SAME
per-spectrum-area normalization already used as the final classification preprocessing step
(`build_preprocessing_pipeline` variant "D"), just applied here directly for the statistics-only
pooled comparison, on top of the already-area-normalized-once-upstream QC'd/harmonized spectra
(re-applying area normalization per spectrum is idempotent up to the already-integrated-over-band
scale, exactly mirroring how `snv` here is a second, statistics-only SNV pass on top of whatever
upstream normalization already happened). `area` is the variant actually reported for this task's
deliverable statistics; `snv` remains available for direct before/after comparison.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.preprocessing import AreaNormalizer, SNVTransformer
from spectra_ml.qc import HARD_FAIL
from spectra_ml.statistics import (
    compute_patient_representative_spectra,
    kruskal_wallis_per_wavelength,
    pairwise_mannwhitney_multiclass,
    pairwise_mannwhitney_multiclass_joint_fdr,
)
from spectra_ml.united import build_united_dataset
from spectra_ml.united_stage import CLASS_NAMES, build_united_stage_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSFER_DIR_V2 = os.path.join(ROOT, "outputs", "harmonization_v2")
RANDOM_SEED = 42
CLASS_LABELS = (0, 1, 2, 3)


def run_variant(usd, normalize: str, method: str):
    spectra = usd.spectra.copy()
    if normalize == "snv":
        valid_rows = ~np.isnan(spectra).any(axis=1)
        spectra[valid_rows] = SNVTransformer().fit_transform(spectra[valid_rows])
    elif normalize == "area":
        valid_rows = ~np.isnan(spectra).any(axis=1)
        spectra[valid_rows] = AreaNormalizer(wavelengths=usd.waves).fit_transform(spectra[valid_rows])

    repr_ = compute_patient_representative_spectra(
        spectra, usd.patient_id, usd.stage_class, usd.waves,
        qc_status=usd.qc_status, exclude_statuses=(HARD_FAIL,), method=method,
    )
    counts = {CLASS_NAMES[c]: int((repr_.target == c).sum()) for c in CLASS_LABELS}
    print(f"[united_multiclass_v2/{normalize}] patients per class: {counts}")

    t0 = time.time()
    kw_df = kruskal_wallis_per_wavelength(repr_, class_labels=CLASS_LABELS)
    print(f"  kruskal_wallis_per_wavelength: {time.time() - t0:.1f}s for {len(kw_df)} wavelengths")

    t0 = time.time()
    pw_df = pairwise_mannwhitney_multiclass(repr_, class_labels=CLASS_LABELS)
    print(f"  pairwise_mannwhitney_multiclass (per-pair FDR): {time.time() - t0:.1f}s for {len(pw_df)} rows")

    t0 = time.time()
    pw_joint_df = pairwise_mannwhitney_multiclass_joint_fdr(repr_, class_labels=CLASS_LABELS)
    print(f"  pairwise_mannwhitney_multiclass_joint_fdr: {time.time() - t0:.1f}s for {len(pw_joint_df)} rows "
          f"({len(pw_joint_df)} = 6 pairs x {len(kw_df)} wavelengths, one joint BH-FDR family)")

    suffix = "" if normalize == "none" else f"_{normalize}"
    out_dir = os.path.join(ROOT, "outputs", "united_multiclass_v2", f"statistics{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    kw_df.to_csv(os.path.join(out_dir, "kruskal_wallis_omnibus.csv"), index=False)
    pw_df.to_csv(os.path.join(out_dir, "pairwise_mannwhitney_per_pair_fdr.csv"), index=False)
    pw_joint_df.to_csv(os.path.join(out_dir, "pairwise_mannwhitney_joint_fdr.csv"), index=False)

    n_sig_kw = int((kw_df["kruskal_p_fdr"] < 0.05).sum())
    print(f"  omnibus significant wavelengths (FDR<0.05): {n_sig_kw}/{len(kw_df)}")
    print("  pairwise per-pair-FDR vs joint-FDR significant-wavelength counts:")
    for (a, b), sub in pw_df.groupby(["class_a", "class_b"]):
        sub_joint = pw_joint_df[(pw_joint_df["class_a"] == a) & (pw_joint_df["class_b"] == b)]
        n_sig_pp = int((sub["mwu_p_fdr"] < 0.05).sum())
        n_sig_j = int((sub_joint["mwu_p_fdr_joint"] < 0.05).sum())
        print(f"    {CLASS_NAMES[a]} vs {CLASS_NAMES[b]}: per-pair {n_sig_pp}/{len(sub)}  "
              f"joint {n_sig_j}/{len(sub_joint)}")
    print(f"  -> {out_dir}/kruskal_wallis_omnibus.csv")
    print(f"  -> {out_dir}/pairwise_mannwhitney_per_pair_fdr.csv")
    print(f"  -> {out_dir}/pairwise_mannwhitney_joint_fdr.csv")
    return kw_df, pw_df, pw_joint_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="median", choices=["median", "mean"])
    parser.add_argument("--normalize", nargs="+", default=["none", "area"], choices=["none", "snv", "area"])
    args = parser.parse_args()

    ud = build_united_dataset(transfer_dir=TRANSFER_DIR_V2)
    usd = build_united_stage_dataset(ud)
    for normalize in args.normalize:
        run_variant(usd, normalize, args.method)


if __name__ == "__main__":
    main()
