"""Stage 2 entry point: full-dataset QC audit (EDA only, per section 5 -- this
pass's thresholds are all fixed rules, so it is also exactly what a
train-fold QC pass inside CV would compute; the difference in the ML pipeline
is only that ShapeOutlierDetector, if enabled, must be refit per outer fold).

Usage: python scripts/run_qc.py configs/instrument_old.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectra_ml.data import build_dataset
from spectra_ml.plotting import (plot_hot_pixel_examples, plot_qc_reason_histogram,
                                  plot_raw_vs_qc_spectra, plot_spectra_per_patient_hist)
from spectra_ml.qc import QCConfig, build_qc_table
from spectra_ml.schema import InstrumentConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instrument_config")
    parser.add_argument("--qc-config", default=os.path.join(ROOT, "configs", "qc.yaml"))
    args = parser.parse_args()

    instrument = InstrumentConfig.from_yaml(args.instrument_config)
    qc_config = QCConfig.from_yaml(args.qc_config)

    dataset = build_dataset(instrument)
    spectrum_id = [f"{pid}_{i}" for i, pid in enumerate(dataset.patient_id)]

    qc_df, hp, correction_log = build_qc_table(dataset.spectra, dataset.waves, dataset.patient_id,
                                                 spectrum_id, qc_config)

    out_dir = os.path.join(ROOT, "outputs", instrument.id, "qc")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    qc_df.to_csv(os.path.join(out_dir, "qc_report.csv"), index=False)

    import json
    with open(os.path.join(out_dir, "hot_pixel_correction_log.json"), "w", encoding="utf-8") as f:
        json.dump(correction_log, f, indent=2)
    with open(os.path.join(out_dir, "systematic_defective_channels.json"), "w", encoding="utf-8") as f:
        bad_idx = [int(i) for i in hp.systematic_channel.nonzero()[0]]
        json.dump({"wave_index": bad_idx, "wavelength_nm": [float(dataset.waves[i]) for i in bad_idx]}, f, indent=2)

    corrected, _ = (dataset.spectra, None)
    from spectra_ml.qc import correct_hot_pixels
    corrected, _ = correct_hot_pixels(dataset.spectra, hp, qc_config)

    plot_raw_vs_qc_spectra(dataset.spectra, dataset.waves, qc_df["qc_status"].to_numpy(),
                            os.path.join(fig_dir, "raw_vs_qc_spectra.png"))
    plot_hot_pixel_examples(dataset.spectra, corrected, dataset.waves, correction_log,
                             os.path.join(fig_dir, "hot_pixel_examples.png"))
    plot_spectra_per_patient_hist(dataset.patient_table(), os.path.join(fig_dir, "spectra_per_patient.png"))
    plot_qc_reason_histogram(qc_df, os.path.join(fig_dir, "qc_reason_histogram.png"))

    counts = qc_df["qc_status"].value_counts()
    print(f"[{instrument.id}] QC: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  systematic defective channels: {int(hp.systematic_channel.sum())}")
    print(f"  hot pixels corrected: {len(correction_log)}")
    print(f"  -> {out_dir}/qc_report.csv")
    print(f"  -> {fig_dir}/")


if __name__ == "__main__":
    main()
