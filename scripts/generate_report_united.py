"""United-dataset report generator: reads the CSV/JSON artifacts already
produced under outputs/united/ (does not recompute anything) and writes:
  - outputs/united/feature_importance.csv (unified table, same format as OLD's)
  - outputs/united/report.html

Structured to mirror scripts/generate_report.py's OLD report section-by-
section, so a reader can directly compare OLD-only vs united-harmonized
side by side.

Scope reminder (also stated in the HTML): this is a single non-blinded pooled
analysis (nested CV over ALL labeled patients from all 3 instruments after
harmonization) -- NOT the Phase-2 blinded validation protocol in
outputs/final_study_unified/.

Usage: python scripts/generate_report_united.py
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united")
OLD = os.path.join(ROOT, "outputs", "old")


def build_feature_importance_csv():
    rows = []
    for fam in ["linear_svm", "pls_logreg"]:
        cs_path = os.path.join(OUT, "interpretability", f"coefficient_stability_{fam}.csv")
        if not os.path.exists(cs_path):
            continue
        cs = pd.read_csv(cs_path)
        for _, r in cs.iterrows():
            rows.append({"model": fam, "wavelength_start": r["wavelength"], "wavelength_end": r["wavelength"],
                         "importance": r["mean"], "importance_std": r["std"], "signed_effect": r["mean"],
                         "stability": r["fraction_folds_in_topK"], "method": "linear_coefficient"})
        perm_path = os.path.join(OUT, "interpretability", f"permutation_importance_stability_{fam}.csv")
        if os.path.exists(perm_path):
            perm = pd.read_csv(perm_path)
            for _, r in perm.iterrows():
                rows.append({"model": fam, "wavelength_start": r["wavelength_start"],
                             "wavelength_end": r["wavelength_end"], "importance": r["mean"],
                             "importance_std": r["std"], "signed_effect": None, "stability": None,
                             "method": "grouped_permutation"})

    vip_path = os.path.join(OUT, "interpretability", "vip_stability_pls_logreg.csv")
    if os.path.exists(vip_path):
        vip = pd.read_csv(vip_path)
        for _, r in vip.iterrows():
            rows.append({"model": "pls_logreg", "wavelength_start": r["wavelength"],
                         "wavelength_end": r["wavelength"], "importance": r["mean"], "importance_std": r["std"],
                         "signed_effect": None, "stability": r["fraction_folds_in_topK"], "method": "pls_vip"})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
    return out


def top_regions(fi: pd.DataFrame, method: str, n: int = 8) -> pd.DataFrame:
    sub = fi[fi["method"] == method].copy()
    if sub.empty:
        return sub
    sub["abs_importance"] = sub["importance"].abs()
    return sub.sort_values("abs_importance", ascending=False).head(n)


def html_table(df: pd.DataFrame, cols=None, float_fmt="{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "<p><i>not available</i></p>"
    if cols:
        df = df[cols]
    return df.to_html(index=False, float_format=lambda x: float_fmt.format(x) if pd.notna(x) else "")


def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    audit = json.load(open(os.path.join(OUT, "audit", "data_audit.json")))
    qc = pd.read_csv(os.path.join(OUT, "qc", "qc_report.csv"))
    classical_metrics = pd.read_csv(os.path.join(OUT, "metrics.csv"))
    bootstrap = safe_read_csv(os.path.join(OUT, "bootstrap", "bootstrap_summary.csv"))
    paired = safe_read_csv(os.path.join(OUT, "bootstrap", "paired_comparisons.csv"))
    calibration = safe_read_csv(os.path.join(OUT, "calibration", "calibration_summary.csv"))
    stats_raw = safe_read_csv(os.path.join(OUT, "statistics", "wavelength_statistics.csv"))
    stats_snv = safe_read_csv(os.path.join(OUT, "statistics_snv", "wavelength_statistics.csv"))
    fi = build_feature_importance_csv()

    auc_bootstrap = bootstrap[bootstrap["metric"] == "roc_auc"].sort_values("mean", ascending=False) \
        if bootstrap is not None else None

    def n_sig(df, col):
        return int((df[col] < 0.05).sum()) if df is not None else None

    by_inst = audit["by_instrument"]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Parkinson's autofluorescence-spectra pipeline: UNITED (old+new1+new2, harmonized)</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1, h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 0.3em; }}
h2 {{ margin-top: 2em; }}
table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th {{ background: #f0f0f0; }}
td:first-child, th:first-child {{ text-align: left; }}
.caveat {{ background: #fff3cd; border-left: 4px solid #e0a800; padding: 0.8em 1em; margin: 1em 0; }}
.finding {{ background: #d4edda; border-left: 4px solid #28a745; padding: 0.8em 1em; margin: 1em 0; }}
.scope {{ background: #e2e3ff; border-left: 4px solid #4b3fdb; padding: 0.8em 1em; margin: 1em 0; }}
img {{ max-width: 100%; border: 1px solid #ddd; margin: 0.5em 0; }}
.figrow {{ display: flex; gap: 1em; flex-wrap: wrap; }}
.figrow img {{ max-width: 48%; }}
code {{ background: #f4f4f4; padding: 1px 4px; }}
</style></head><body>

<h1>Parkinson's disease classification from skin autofluorescence spectra -- UNITED (all instruments, harmonized)</h1>
<p>Instruments pooled: <b>old</b> (STS s/n S07279, reference), <b>new_pd</b> (Optosky, new-PD-department,
instrument id <code>new1</code>), <b>new_control</b> (Optosky, new-control-department, instrument id
<code>new2</code>). Generated by <code>scripts/generate_report_united.py</code> from artifacts already produced
by <code>scripts/build_united_dataset.py</code>, <code>scripts/run_statistics_united.py</code>,
<code>scripts/run_experiment_united.py</code>, <code>scripts/run_calibration_united.py</code>,
<code>scripts/run_bootstrap_united.py</code>, <code>scripts/run_interpretability_united.py</code> (no
computation happens in this script).</p>

<div class="scope">
<b>Scope of this report.</b> This is a single, non-blinded, all-data pooled descriptive + classification study
of the <b>harmonized</b> 3-instrument population (same depth of analysis as the original OLD-only exploratory
study in <code>outputs/old/report.html</code>, but on new1/new2 spectra transferred into OLD's space via the
FROZEN, diagnosis-independent transfer curves already built and frozen in
<code>outputs/final_study_unified/transfer/</code> during Phase 2). Every labeled patient is used for both
training and evaluation via nested cross-validation, exactly like OLD's original study. This is
<b>explicitly NOT</b> a re-run of the Phase-2 blinded validation protocol (<code>outputs/final_study_unified/</code>):
no new blind split was created, no transfer curves were rebuilt, and <code>outputs/final_study_unified/private/</code>
was not touched. The pooled blinded-test result from Phase 2 (AUC 0.876, linear_svm, 3 pre-registered families
only) remains the authoritative validated estimate; the numbers below are a broader exploratory comparison across
7 model families on the full harmonized cohort.
</div>

<div class="caveat">
<b>Read this before the numbers below.</b>
<ul>
<li><b>Instrument id is metadata only</b> -- never a predictor feature in any classifier; used only for QC/
statistics stratification and reporting.</li>
<li><b>Class composition differs sharply by instrument.</b> new_pd (new1) is almost entirely Parkinson patients
({by_inst['new_pd']['n_parkinson_patients']} park / {by_inst['new_pd']['n_control_patients']} control) and
new_control (new2) is almost entirely control ({by_inst['new_control']['n_parkinson_patients']} park /
{by_inst['new_control']['n_control_patients']} control) -- so instrument and diagnosis are heavily confounded
in the pooled dataset by construction (department-level recruitment, not random assignment). Any classifier
performance here should be read with this confound in mind: it cannot be ruled out that some signal reflects
residual inter-instrument differences that the frozen shape-harmonization did not fully remove, rather than
purely biological PD/control differences. The Phase-2 blinded study's pre-registered protocol is the more
defensible test of this; this report is exploratory.</li>
<li><b>Patient GUIDs were confirmed globally unique</b> across all 3 instruments (zero collisions) before
pooling -- see <code>outputs/united/audit/data_audit.json</code>.</li>
</ul>
</div>

<h2>1. Data</h2>
<p>{audit['n_patients_total']} patients (GUID-level, all instruments pooled), {audit['n_spectra_total']} spectra,
wavelength grid {audit['wavelength_range_nm'][0]:.1f}-{audit['wavelength_range_nm'][1]:.1f} nm
({audit['n_wavelength_features']} points, OLD's reference grid). Class balance (all loaded patients, before QC):
{audit['n_parkinson_patients']} Parkinson, {audit['n_control_patients']} control,
{audit['n_patients_unknown_target']} unknown target (excluded from statistics/modeling, kept in the audit table
for completeness).</p>
<p>Per-instrument breakdown:</p>
<table><tr><th>instrument</th><th>patients</th><th>spectra</th><th>parkinson</th><th>control</th><th>unknown</th>
<th>QC status counts</th></tr>
{"".join(f"<tr><td>{role}</td><td>{by_inst[role]['n_patients']}</td><td>{by_inst[role]['n_spectra']}</td>"
         f"<td>{by_inst[role]['n_parkinson_patients']}</td><td>{by_inst[role]['n_control_patients']}</td>"
         f"<td>{by_inst[role]['n_patients_unknown_target']}</td><td>{by_inst[role]['qc_status_counts']}</td></tr>"
         for role in ["old", "new_pd", "new_control"])}
</table>
<p>Harmonization: {audit['harmonization']['method']} Transfer curves reused from (frozen, not rebuilt):
<code>{audit['harmonization']['transfer_curve_source']}</code>.</p>

<h2>2. QC</h2>
<p>Spectrum-level QC status counts (pooled, computed once post-pooling since QC rules are per-spectrum,
relative-scale, and instrument-agnostic):</p>
{html_table(qc['qc_status'].value_counts().rename_axis('status').reset_index(name='n_spectra'))}

<h2>3. Statistics (patient-level, median spectrum, Mann-Whitney incl. tie-corrected z-score)</h2>
<p>Raw (harmonized, area-normalized) vs. SNV-normalized group comparison. "Raw" here means the harmonized,
area-normalized pooled spectra (a separate area-normalized variant was skipped: harmonization already puts every
instrument on a common area-normalized scale, see caveat in <code>outputs/united/statistics/</code> naming).
FDR-corrected significance count: welch={n_sig(stats_raw, 'welch_p_fdr')}/278 (raw),
{n_sig(stats_snv, 'welch_p_fdr')}/278 (SNV); mwu={n_sig(stats_raw, 'mwu_p_fdr')}/278 (raw),
{n_sig(stats_snv, 'mwu_p_fdr')}/278 (SNV). Figures:</p>
<div class="figrow">
<img src="statistics/figures/effect_size_spectrum.png" alt="raw effect size">
<img src="statistics_snv/figures/effect_size_spectrum.png" alt="SNV effect size">
</div>
<p>Full tables incl. <code>mwu_z</code> (tie-corrected normal-approximation z-score, verified against
<code>scipy.stats.mannwhitneyu(..., method="asymptotic")</code> in <code>tests/test_statistics.py</code>):
<code>outputs/united/statistics{{,_snv}}/wavelength_statistics.csv</code>.</p>

<h2>4. Model comparison (patient-level, nested CV, all 3 instruments pooled)</h2>
<p>Classical models (7 families: dummy, logreg, linear_svm, pls_logreg, rbf_svm, hgb, extratrees -- same family
set as OLD's original comparison), mean +/- std ROC-AUC across outer folds, then patient-level bootstrap 95% CI
(2000 iterations, on calibrated_probability):</p>
{html_table(classical_metrics[['family', 'roc_auc_mean', 'roc_auc_std', 'pr_auc_mean', 'sensitivity_mean', 'specificity_mean']])}
{html_table(auc_bootstrap[['family', 'mean', 'ci95_lo', 'ci95_hi']].rename(columns={'mean': 'bootstrap_mean_auc'})) if auc_bootstrap is not None else '<p><i>bootstrap not available</i></p>'}
<p><i>Neural models (cnn1d/mlp) were not run for this united analysis (runtime cost on the larger pooled cohort
was not justified given the classical-family comparison already answers the primary "does the OLD finding
replicate" question; OLD's own neural results were not competitive with the linear models either -- see
<code>outputs/old/report.html</code> section 4).</i></p>

<h2>5. Calibration</h2>
<p>Platt scaling per outer fold, fit on group-aware OOF patient-level scores from outer_train only. Brier score
raw -> calibrated:</p>
{html_table(calibration[['family', 'raw_brier_mean', 'calibrated_brier_mean', 'raw_roc_auc_mean', 'calibrated_roc_auc_mean']]) if calibration is not None else '<p><i>not available</i></p>'}
<div class="figrow">
<img src="calibration/figures/reliability_linear_svm.png" alt="linear_svm reliability">
<img src="calibration/figures/reliability_pls_logreg.png" alt="pls_logreg reliability">
</div>

<h2>6. Bootstrap comparisons</h2>
{html_table(paired[['family_a', 'family_b', 'mean_delta', 'ci95_lo', 'ci95_hi', 'significant_at_0.05']]) if paired is not None else '<p><i>not available</i></p>'}

<h2>7. Interpretability</h2>
<p>Top wavelength points/regions by |linear coefficient| (mean across outer folds):</p>
{html_table(top_regions(fi, 'linear_coefficient')[['model', 'wavelength_start', 'importance', 'stability']]) if len(fi) else '<p><i>not available</i></p>'}
<p>Top wavelength regions by grouped permutation importance:</p>
{html_table(top_regions(fi, 'grouped_permutation')[['model', 'wavelength_start', 'wavelength_end', 'importance', 'importance_std']]) if len(fi) else '<p><i>not available</i></p>'}
<div class="figrow">
<img src="interpretability/figures/coefficient_stability_linear_svm.png" alt="linear_svm coefficients">
<img src="interpretability/figures/permutation_importance_linear_svm.png" alt="linear_svm permutation">
</div>
<p>Compare against OLD-only's converged regions (~467-500 nm dominant, ~416-425 nm and ~537-561 nm secondary,
<code>outputs/old/report.html</code> section 6) to check replication on the harmonized pooled cohort.</p>

<h2>Bottom line</h2>
<div class="finding">
See sections 4-7 above for the united-cohort headline numbers, and the top-level task summary for a direct
OLD-only vs united-harmonized comparison. This report is exploratory (see scope note at the top) and should be
read alongside, not instead of, the Phase-2 pre-registered blinded validation result.
</div>

</body></html>"""

    with open(os.path.join(OUT, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> {os.path.join(OUT, 'feature_importance.csv')} ({len(fi)} rows)")
    print(f"-> {os.path.join(OUT, 'report.html')}")


if __name__ == "__main__":
    main()
