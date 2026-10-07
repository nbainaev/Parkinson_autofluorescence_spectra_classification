"""Reusable plotting helpers. Scripts/notebooks call these; no analysis logic here."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectra_ml.qc import HARD_FAIL, PASS, SOFT_WARNING


def plot_raw_vs_qc_spectra(spectra, waves, qc_status, out_path, n_examples=40, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    for ax, status, title in zip(axes, [PASS, SOFT_WARNING, HARD_FAIL], ["PASS", "SOFT_WARNING", "HARD_FAIL"]):
        idx = np.where(qc_status == status)[0]
        if len(idx) == 0:
            ax.set_title(f"{title} (n=0)")
            continue
        sample = rng.choice(idx, size=min(n_examples, len(idx)), replace=False)
        for i in sample:
            ax.plot(waves, spectra[i], linewidth=0.5, alpha=0.5)
        ax.set_title(f"{title} (n={len(idx)})")
        ax.set_xlabel("Wavelength, nm")
    axes[0].set_ylabel("Intensity, a.u.")
    fig.suptitle("Raw spectra by QC status (sample)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hot_pixel_examples(spectra_raw, spectra_corrected, waves, correction_log, out_path, n_examples=6):
    spectra_idx = sorted({e["spectrum_index"] for e in correction_log})[:n_examples]
    if not spectra_idx:
        return
    n = len(spectra_idx)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, i in zip(axes[0], spectra_idx):
        ax.plot(waves, spectra_raw[i], label="raw", linewidth=1, alpha=0.7)
        ax.plot(waves, spectra_corrected[i], label="corrected", linewidth=1, alpha=0.7)
        idx_here = [e["wave_index"] for e in correction_log if e["spectrum_index"] == i]
        ax.scatter(waves[idx_here], spectra_raw[i][idx_here], color="red", s=20, zorder=5, label="hot pixel")
        ax.set_title(f"spectrum {i}")
        ax.legend(fontsize=7)
    fig.suptitle("Hot-pixel correction examples")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectra_per_patient_hist(patient_table: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(patient_table["n_spectra"], bins=range(1, int(patient_table["n_spectra"].max()) + 2))
    ax.set_xlabel("Spectra per patient")
    ax.set_ylabel("Number of patients")
    ax.set_title("Distribution of spectra per patient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_qc_reason_histogram(qc_df: pd.DataFrame, out_path):
    reasons = qc_df["reason_codes"].str.split(";").explode()
    reasons = reasons[reasons.notna() & (reasons != "")]
    counts = reasons.value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(counts):
        ax.barh(counts.index[::-1], counts.values[::-1])
    ax.set_xlabel("Count")
    ax.set_title("QC reason codes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_group_spectra_with_ci(stats_df: pd.DataFrame, out_path, ci=95,
                                title="Patient-level median spectra: Parkinson vs Control"):
    fig, ax = plt.subplots(figsize=(10, 5))
    w = stats_df["wavelength"]
    ax.plot(w, stats_df["park_median"], color="#d62728", label="Parkinson (median)", linewidth=1.5)
    ax.fill_between(w, stats_df[f"park_bootstrap_ci{ci}_lo"], stats_df[f"park_bootstrap_ci{ci}_hi"],
                     color="#d62728", alpha=0.15)
    ax.plot(w, stats_df["control_median"], color="#1f77b4", label="Control (median)", linewidth=1.5)
    ax.fill_between(w, stats_df[f"control_bootstrap_ci{ci}_lo"], stats_df[f"control_bootstrap_ci{ci}_hi"],
                     color="#1f77b4", alpha=0.15)
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Intensity, a.u.")
    ax.set_title(f"{title} ({ci}% bootstrap CI, patient-level)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_difference_spectrum(stats_df: pd.DataFrame, out_path, ci=95):
    fig, ax = plt.subplots(figsize=(10, 4))
    w = stats_df["wavelength"]
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.plot(w, stats_df["difference"], color="black", linewidth=1.2)
    ax.fill_between(w, stats_df[f"diff_bootstrap_ci{ci}_lo"], stats_df[f"diff_bootstrap_ci{ci}_hi"],
                     alpha=0.2, color="black", label="bootstrap CI")
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Parkinson - Control (median)")
    ax.set_title("Difference spectrum")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_effect_size_spectrum(stats_df: pd.DataFrame, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    w = stats_df["wavelength"]
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].plot(w, stats_df["hedges_g"], color="#2ca02c")
    axes[0].set_ylabel("Hedges' g")
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].plot(w, stats_df["rank_biserial"], color="#9467bd")
    axes[1].set_ylabel("Rank-biserial r")
    axes[1].set_xlabel("Wavelength, nm")
    fig.suptitle("Effect size vs wavelength")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(y_true, raw_prob, calibrated_prob, out_path, n_bins=5, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, prob, label in zip(axes, [raw_prob, calibrated_prob], ["Before calibration", "After calibration (Platt)"]):
        edges = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
        bin_acc, bin_conf, bin_n = [], [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            bin_acc.append(np.mean(np.asarray(y_true)[mask]))
            bin_conf.append(np.mean(np.asarray(prob)[mask]))
            bin_n.append(mask.sum())
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
        ax.scatter(bin_conf, bin_acc, s=[20 + 4 * n for n in bin_n], color="#1f77b4")
        ax.plot(bin_conf, bin_acc, color="#1f77b4", alpha=0.5)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_probability_histograms(raw_prob, calibrated_prob, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    axes[0].hist(raw_prob, bins=20, range=(0, 1), color="#1f77b4", alpha=0.8)
    axes[0].set_title("Raw probability")
    axes[1].hist(calibrated_prob, bins=20, range=(0, 1), color="#2ca02c", alpha=0.8)
    axes[1].set_title("Calibrated probability")
    for ax in axes:
        ax.set_xlabel("P(Parkinson)")
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fdr_significance_spectrum(stats_df: pd.DataFrame, out_path, alpha=0.05):
    fig, ax = plt.subplots(figsize=(10, 4))
    w = stats_df["wavelength"]
    for col, color, label in [("welch_p_fdr", "#1f77b4", "Welch (FDR)"), ("mwu_p_fdr", "#d62728", "Mann-Whitney (FDR)")]:
        y = -np.log10(stats_df[col].clip(lower=1e-300))
        ax.plot(w, y, color=color, label=label, linewidth=1)
    ax.axhline(-np.log10(alpha), color="grey", linestyle="--", linewidth=0.8, label=f"alpha={alpha}")
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("-log10(FDR p)")
    ax.set_title("FDR significance vs wavelength")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coefficient_stability(stability_df: pd.DataFrame, out_path, family: str, backprojected: bool):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    w = stability_df["wavelength"]
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].plot(w, stability_df["mean"], color="#1f77b4")
    axes[0].fill_between(w, stability_df["ci95_lo"], stability_df["ci95_hi"], alpha=0.2, color="#1f77b4")
    units = "raw-intensity units" if backprojected else "SNV-normalized-feature units (not raw intensity)"
    axes[0].set_ylabel(f"Coefficient\n({units})")
    axes[0].set_title(f"{family}: coefficient stability across outer folds (mean, 95% range)")

    axes[1].bar(w, stability_df["fraction_folds_in_topK"], width=(w.max() - w.min()) / len(w), color="#2ca02c")
    axes[1].set_ylabel("Fraction of folds\nin top-20 |coef|")
    axes[1].set_xlabel("Wavelength, nm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_permutation_importance(perm_stability_df: pd.DataFrame, out_path, family: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    mid = (perm_stability_df["wavelength_start"] + perm_stability_df["wavelength_end"]) / 2
    width = perm_stability_df["wavelength_end"] - perm_stability_df["wavelength_start"]
    ax.bar(mid, perm_stability_df["mean"], width=width, yerr=perm_stability_df["std"],
           color="#9467bd", alpha=0.8, error_kw=dict(alpha=0.4))
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Grouped permutation importance\n(patient-level ROC-AUC drop)")
    ax.set_title(f"{family}: permutation importance by wavelength region (mean +/- std across 5 outer folds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
