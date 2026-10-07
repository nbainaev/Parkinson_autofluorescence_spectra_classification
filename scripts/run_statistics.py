"""Stage 4 entry point: patient-level statistics + EDA figures.

Usage: python scripts/run_statistics.py configs/instrument_old.yaml
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spectra_ml.data import build_dataset
from spectra_ml.plotting import (plot_difference_spectrum, plot_effect_size_spectrum,
                                  plot_fdr_significance_spectrum, plot_group_spectra_with_ci)
from spectra_ml.preprocessing import AreaNormalizer, SNVTransformer
from spectra_ml.qc import QCConfig, build_qc_table, correct_hot_pixels
from spectra_ml.schema import InstrumentConfig
from spectra_ml.statistics import compare_groups_per_wavelength, compute_patient_representative_spectra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instrument_config")
    parser.add_argument("--qc-config", default=os.path.join(ROOT, "configs", "qc.yaml"))
    parser.add_argument("--method", default="median", choices=["median", "mean"])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--normalize", default="none", choices=["none", "snv", "area"],
                         help="Per-spectrum normalization applied BEFORE patient-level aggregation and "
                              "group comparison -- diagnostic for whether a group difference is a spectral "
                              "SHAPE effect or just an overall intensity/baseline confound. Row-wise and "
                              "stateless (no fit), so applying it to the full dataset here is not a leakage "
                              "concern -- unlike SpectralStandardScaler, which would need a train-only fit.")
    args = parser.parse_args()

    instrument = InstrumentConfig.from_yaml(args.instrument_config)
    qc_config = QCConfig.from_yaml(args.qc_config)

    dataset = build_dataset(instrument)
    spectrum_id = [f"{pid}_{i}" for i, pid in enumerate(dataset.patient_id)]
    qc_df, hp, _ = build_qc_table(dataset.spectra, dataset.waves, dataset.patient_id, spectrum_id, qc_config)
    corrected, _ = correct_hot_pixels(dataset.spectra, hp, qc_config)

    if args.normalize in ("snv", "area"):
        normalizer = SNVTransformer() if args.normalize == "snv" else AreaNormalizer(wavelengths=dataset.waves)
        valid_rows = ~np.isnan(corrected).any(axis=1)
        corrected = corrected.copy()
        corrected[valid_rows] = normalizer.fit_transform(corrected[valid_rows])

    repr_ = compute_patient_representative_spectra(
        corrected, dataset.patient_id, dataset.target, dataset.waves,
        qc_status=qc_df["qc_status"].to_numpy(), method=args.method,
    )
    n_unknown = int(np.isnan(repr_.target).sum())
    print(f"[{instrument.id}] patients: park={int((repr_.target == 1).sum())} "
          f"control={int((repr_.target == 0).sum())} unknown_target={n_unknown}")

    t0 = time.time()
    stats_df = compare_groups_per_wavelength(repr_, n_boot=args.n_boot, ci=95, seed=instrument.data.random_seed)
    print(f"  compare_groups_per_wavelength: {time.time() - t0:.1f}s for {len(stats_df)} wavelengths")

    suffix = "" if args.normalize == "none" else f"_{args.normalize}"
    out_dir = os.path.join(ROOT, "outputs", instrument.id, f"statistics{suffix}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(out_dir, "wavelength_statistics.csv"), index=False)

    plot_group_spectra_with_ci(stats_df, os.path.join(fig_dir, "group_mean_median_spectra.png"))
    plot_difference_spectrum(stats_df, os.path.join(fig_dir, "difference_spectrum.png"))
    plot_effect_size_spectrum(stats_df, os.path.join(fig_dir, "effect_size_spectrum.png"))
    plot_fdr_significance_spectrum(stats_df, os.path.join(fig_dir, "fdr_significance_spectrum.png"))

    n_sig_welch = int((stats_df["welch_p_fdr"] < 0.05).sum())
    n_sig_mwu = int((stats_df["mwu_p_fdr"] < 0.05).sum())
    print(f"  wavelengths significant at FDR<0.05: welch={n_sig_welch}/{len(stats_df)} mwu={n_sig_mwu}/{len(stats_df)}")
    print(f"  -> {out_dir}/wavelength_statistics.csv")
    print(f"  -> {fig_dir}/")


if __name__ == "__main__":
    main()
