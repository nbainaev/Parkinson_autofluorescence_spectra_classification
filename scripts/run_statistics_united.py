"""United-dataset statistics entry point: patient-level PD-vs-control
comparison (Mann-Whitney incl. tie-corrected z-score, Welch, Hedges' g, FDR)
on the pooled, harmonized, 3-instrument dataset (spectra_ml.united).

Mirrors scripts/run_statistics.py's structure/output-file convention so
outputs/united/statistics{,_snv}/ is directly comparable to
outputs/old/statistics{,_snv}/. Area-normalized variant is intentionally
skipped here: the united "raw" base spectra are already area-normalized (+
frozen K-transfer for new1/new2) as part of harmonization -- see
spectra_ml.united module docstring.

Usage: python scripts/run_statistics_united.py [--normalize none|snv] [--method median|mean]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.plotting import (plot_difference_spectrum, plot_effect_size_spectrum,
                                  plot_fdr_significance_spectrum, plot_group_spectra_with_ci)
from spectra_ml.preprocessing import SNVTransformer
from spectra_ml.qc import HARD_FAIL
from spectra_ml.statistics import compare_groups_per_wavelength, compute_patient_representative_spectra
from spectra_ml.united import build_united_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_SEED = 42


def run_variant(ud, normalize: str, method: str, n_boot: int):
    spectra = ud.spectra.copy()
    if normalize == "snv":
        valid_rows = ~np.isnan(spectra).any(axis=1)
        spectra[valid_rows] = SNVTransformer().fit_transform(spectra[valid_rows])

    repr_ = compute_patient_representative_spectra(
        spectra, ud.patient_id, ud.target, ud.waves,
        qc_status=ud.qc_status, exclude_statuses=(HARD_FAIL,), method=method,
    )
    n_unknown = int(np.isnan(repr_.target).sum())
    print(f"[united/{normalize}] patients: park={int((repr_.target == 1).sum())} "
          f"control={int((repr_.target == 0).sum())} unknown_target={n_unknown}")

    t0 = time.time()
    stats_df = compare_groups_per_wavelength(repr_, n_boot=n_boot, ci=95, seed=RANDOM_SEED)
    print(f"  compare_groups_per_wavelength: {time.time() - t0:.1f}s for {len(stats_df)} wavelengths")

    suffix = "" if normalize == "none" else f"_{normalize}"
    out_dir = os.path.join(ROOT, "outputs", "united", f"statistics{suffix}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(out_dir, "wavelength_statistics.csv"), index=False)

    plot_group_spectra_with_ci(stats_df, os.path.join(fig_dir, "group_mean_median_spectra.png"))
    plot_difference_spectrum(stats_df, os.path.join(fig_dir, "difference_spectrum.png"))
    plot_effect_size_spectrum(stats_df, os.path.join(fig_dir, "effect_size_spectrum.png"))
    plot_fdr_significance_spectrum(stats_df, os.path.join(fig_dir, "fdr_significance_spectrum.png"))

    n_sig_welch = int((stats_df["welch_p_fdr"] < 0.05).sum())
    n_sig_mwu = int((stats_df["mwu_p_fdr"] < 0.05).sum())
    n_sig_mwu_z = int((stats_df["mwu_z"].abs() > 1.959964).sum())  # ~alpha=0.05, uncorrected, cross-check only
    print(f"  wavelengths significant at FDR<0.05: welch={n_sig_welch}/{len(stats_df)} mwu={n_sig_mwu}/{len(stats_df)}"
          f"  (uncorrected |mwu_z|>1.96: {n_sig_mwu_z}/{len(stats_df)})")
    print(f"  -> {out_dir}/wavelength_statistics.csv")
    print(f"  -> {fig_dir}/")
    return stats_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="median", choices=["median", "mean"])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--normalize", nargs="+", default=["none", "snv"], choices=["none", "snv"])
    args = parser.parse_args()

    ud = build_united_dataset()
    for normalize in args.normalize:
        run_variant(ud, normalize, args.method, args.n_boot)


if __name__ == "__main__":
    main()
