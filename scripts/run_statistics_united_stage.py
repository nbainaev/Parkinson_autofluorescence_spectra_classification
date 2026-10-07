"""Task 2: multiclass (4-class Hoehn-Yahr PD stage) pooled statistics -- Kruskal-
Wallis omnibus test + pairwise Mann-Whitney (tie-corrected z-score, Hedges' g,
FDR) per wavelength, on the pooled, harmonized, 3-instrument dataset restricted
to the 4 included classes (control/stage1/stage2/stage3).

Mirrors scripts/run_statistics_united.py's structure/output convention. Area-
normalized variant is skipped for the same reason as the binary script: the
united "raw" base spectra are already area-normalized (+ frozen K-transfer) as
part of harmonization.

Usage: python scripts/run_statistics_united_stage.py [--normalize none|snv] [--method median|mean]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.preprocessing import SNVTransformer
from spectra_ml.qc import HARD_FAIL
from spectra_ml.statistics import (
    compute_patient_representative_spectra,
    kruskal_wallis_per_wavelength,
    pairwise_mannwhitney_multiclass,
)
from spectra_ml.united import build_united_dataset
from spectra_ml.united_stage import CLASS_NAMES, build_united_stage_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_SEED = 42
CLASS_LABELS = (0, 1, 2, 3)


def run_variant(usd, normalize: str, method: str):
    spectra = usd.spectra.copy()
    if normalize == "snv":
        valid_rows = ~np.isnan(spectra).any(axis=1)
        spectra[valid_rows] = SNVTransformer().fit_transform(spectra[valid_rows])

    repr_ = compute_patient_representative_spectra(
        spectra, usd.patient_id, usd.stage_class, usd.waves,
        qc_status=usd.qc_status, exclude_statuses=(HARD_FAIL,), method=method,
    )
    # restrict to the 4 modeled classes -- patients with NaN stage_class (excluded:
    # stage4 or unknown) simply have no class mask below and are ignored, matching
    # how compare_groups_per_wavelength already ignores unknown-target patients.
    counts = {CLASS_NAMES[c]: int((repr_.target == c).sum()) for c in CLASS_LABELS}
    print(f"[united_multiclass/{normalize}] patients per class: {counts}")

    t0 = time.time()
    kw_df = kruskal_wallis_per_wavelength(repr_, class_labels=CLASS_LABELS)
    print(f"  kruskal_wallis_per_wavelength: {time.time() - t0:.1f}s for {len(kw_df)} wavelengths")

    t0 = time.time()
    pw_df = pairwise_mannwhitney_multiclass(repr_, class_labels=CLASS_LABELS)
    print(f"  pairwise_mannwhitney_multiclass: {time.time() - t0:.1f}s for {len(pw_df)} rows "
          f"({len(pw_df) // len(kw_df)} pairs x {len(kw_df)} wavelengths)")

    suffix = "" if normalize == "none" else f"_{normalize}"
    out_dir = os.path.join(ROOT, "outputs", "united_multiclass", f"statistics{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    kw_df.to_csv(os.path.join(out_dir, "kruskal_wallis_omnibus.csv"), index=False)
    pw_df.to_csv(os.path.join(out_dir, "pairwise_mannwhitney.csv"), index=False)

    n_sig_kw = int((kw_df["kruskal_p_fdr"] < 0.05).sum())
    print(f"  omnibus significant wavelengths (FDR<0.05): {n_sig_kw}/{len(kw_df)}")
    for (a, b), sub in pw_df.groupby(["class_a", "class_b"]):
        n_sig = int((sub["mwu_p_fdr"] < 0.05).sum())
        print(f"    pair {CLASS_NAMES[a]} vs {CLASS_NAMES[b]}: {n_sig}/{len(sub)} significant (FDR<0.05)")
    print(f"  -> {out_dir}/kruskal_wallis_omnibus.csv")
    print(f"  -> {out_dir}/pairwise_mannwhitney.csv")
    return kw_df, pw_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="median", choices=["median", "mean"])
    parser.add_argument("--normalize", nargs="+", default=["none", "snv"], choices=["none", "snv"])
    args = parser.parse_args()

    ud = build_united_dataset()
    usd = build_united_stage_dataset(ud)
    for normalize in args.normalize:
        run_variant(usd, normalize, args.method)


if __name__ == "__main__":
    main()
