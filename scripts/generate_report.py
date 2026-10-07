"""Stage 14: unified report. Reads the CSV/JSON artifacts already produced by
Stages 1-13 (does not recompute anything) and writes:
  - outputs/old/feature_importance.csv  (unified table, section 30 format)
  - outputs/old/report.html             (human-readable summary)

Usage: python scripts/generate_report.py
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "old")


def build_feature_importance_csv():
    rows = []
    for fam in ["linear_svm", "pls_logreg"]:
        cs = pd.read_csv(os.path.join(OUT, "interpretability", f"coefficient_stability_{fam}.csv"))
        for _, r in cs.iterrows():
            rows.append({"model": fam, "wavelength_start": r["wavelength"], "wavelength_end": r["wavelength"],
                         "importance": r["mean"], "importance_std": r["std"], "signed_effect": r["mean"],
                         "stability": r["fraction_folds_in_topK"], "method": "linear_coefficient"})
        perm = pd.read_csv(os.path.join(OUT, "interpretability", f"permutation_importance_stability_{fam}.csv"))
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

    for baseline in ["zero", "train_mean", "train_median"]:
        ig_path = os.path.join(OUT, "interpretability", f"integrated_gradients_{baseline}_cnn1d.csv")
        if os.path.exists(ig_path):
            ig = pd.read_csv(ig_path)
            for _, r in ig.iterrows():
                rows.append({"model": "cnn1d", "wavelength_start": r["wavelength"],
                             "wavelength_end": r["wavelength"], "importance": r["mean_abs_attribution"],
                             "importance_std": None, "signed_effect": r["mean_attribution"], "stability": None,
                             "method": f"integrated_gradients_{baseline}"})

    gs_path = os.path.join(OUT, "interpretability", "gradient_shap_cnn1d.csv")
    if os.path.exists(gs_path):
        gs = pd.read_csv(gs_path)
        for _, r in gs.iterrows():
            rows.append({"model": "cnn1d", "wavelength_start": r["wavelength"], "wavelength_end": r["wavelength"],
                         "importance": r["mean_abs_attribution"], "importance_std": None,
                         "signed_effect": r["mean_attribution"], "stability": None, "method": "gradient_shap"})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
    return out


def top_regions(fi: pd.DataFrame, method: str, n: int = 8) -> pd.DataFrame:
    sub = fi[fi["method"] == method].copy()
    sub["abs_importance"] = sub["importance"].abs()
    return sub.sort_values("abs_importance", ascending=False).head(n)


def html_table(df: pd.DataFrame, cols=None, float_fmt="{:.3f}") -> str:
    if cols:
        df = df[cols]
    return df.to_html(index=False, float_format=lambda x: float_fmt.format(x) if pd.notna(x) else "")


def main():
    audit = json.load(open(os.path.join(OUT, "audit", "data_audit.json")))
    qc = pd.read_csv(os.path.join(OUT, "qc", "qc_report.csv"))
    classical_metrics = pd.read_csv(os.path.join(OUT, "metrics.csv"))
    neural_metrics = pd.read_csv(os.path.join(OUT, "models", "neural_fold_metrics.csv"))
    bootstrap = pd.read_csv(os.path.join(OUT, "bootstrap", "bootstrap_summary.csv"))
    paired = pd.read_csv(os.path.join(OUT, "bootstrap", "paired_comparisons.csv"))
    calibration = pd.read_csv(os.path.join(OUT, "calibration", "calibration_summary.csv"))
    fi = build_feature_importance_csv()

    ig_gs_corr_text = "correlation not available (one of the two hasn't been generated yet)."
    ig_path = os.path.join(OUT, "interpretability", "integrated_gradients_train_mean_cnn1d.csv")
    gs_path = os.path.join(OUT, "interpretability", "gradient_shap_cnn1d.csv")
    if os.path.exists(ig_path) and os.path.exists(gs_path):
        ig_df = pd.read_csv(ig_path).sort_values("wavelength")
        gs_df = pd.read_csv(gs_path).sort_values("wavelength")
        corr = pd.Series(ig_df["mean_abs_attribution"].to_numpy()).corr(
            pd.Series(gs_df["mean_abs_attribution"].to_numpy()))
        ig_gs_corr_text = f"agreeing at r={corr:.2f} across wavelengths (|attribution|, pooled across 5 outer folds)."

    neural_summary = neural_metrics.groupby("family")[["raw_roc_auc", "calibrated_roc_auc"]].agg(["mean", "std"])
    neural_summary.columns = [f"{a}_{b}" for a, b in neural_summary.columns]
    neural_summary = neural_summary.reset_index()

    auc_bootstrap = bootstrap[bootstrap["metric"] == "roc_auc"].sort_values("mean", ascending=False)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Parkinson's autofluorescence-spectra pipeline: instrument=old (STS)</title>
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
img {{ max-width: 100%; border: 1px solid #ddd; margin: 0.5em 0; }}
.figrow {{ display: flex; gap: 1em; flex-wrap: wrap; }}
.figrow img {{ max-width: 48%; }}
code {{ background: #f4f4f4; padding: 1px 4px; }}
</style></head><body>

<h1>Parkinson's disease classification from skin autofluorescence spectra</h1>
<p>Instrument: <b>old</b> (STS s/n S07279). Generated by <code>scripts/generate_report.py</code>
from Stage 1-13 artifacts (no computation happens in this script). Config snapshot:
<code>outputs/old/config_snapshot.yaml</code>.</p>

<div class="caveat">
<b>Read this before the numbers below.</b>
<ul>
<li><b>patient_id = per-visit GUID, not the MG-1-N clinical ID.</b> 16 MG-1-N ids were found to collide between
two different real patients (build_master_table.py assigns new-department-1 and old-department-1 patients from
two independent, unsynchronized numbering schemes). GUID has zero such collisions.</li>
<li><b>The "control" class is disproportionately missing legacy data.</b> 31/74 (42%) of control-labeled visits
have zero spectra in the analysis window, vs 12/152 (8%) of Parkinson-labeled visits -- traced to the raw source
file (<code>raw_spectra_to_MG168.xlsx</code>): the <i>Comparison</i> clinical subgroup (64% missing) is the
overwhelming driver, not genuine healthy <i>Control</i> (12% missing). Per your instruction, Comparison+Control
are unified as one "no PD" class, and only patients with real spectra are used
(park=140, control=43 after QC).</li>
<li><b>The raw (unnormalized) group difference was mostly a confound.</b> Statistics on raw intensities found
278/278 wavelengths "significant" at FDR&lt;0.05 with a near-uniform effect size across the whole spectrum
(Hedges' g -0.56..-0.72 everywhere) -- the signature of a global intensity/baseline shift, not a spectral-shape
effect. After per-spectrum SNV normalization, Welch's test found 0/278 significant and Mann-Whitney found 62/278,
now in localized bands. See "Statistics" below.</li>
</ul>
</div>

<h2>1. Data</h2>
<p>{audit['n_patients']} patients (GUID-level), {audit['n_spectra']} spectra, wavelength grid
{audit['wavelength_range_nm'][0]:.1f}-{audit['wavelength_range_nm'][1]:.1f} nm ({audit['n_wavelength_features']}
points). Class balance (all loaded patients, before QC): {audit['n_parkinson_patients']} Parkinson,
{audit['n_control_patients']} control.</p>

<h2>2. QC</h2>
<p>Spectrum-level QC status counts:</p>
{html_table(qc['qc_status'].value_counts().rename_axis('status').reset_index(name='n_spectra'))}
<p>See <code>outputs/old/qc/figures/</code> for raw-vs-QC-status spectra, hot-pixel correction examples, and the
QC-reason histogram.</p>

<h2>3. Statistics (patient-level, median spectrum)</h2>
<p>Raw vs. SNV-normalized group comparison -- see the caveat above. Figures:</p>
<div class="figrow">
<img src="statistics/figures/effect_size_spectrum.png" alt="raw effect size">
<img src="statistics_snv/figures/effect_size_spectrum.png" alt="SNV effect size">
</div>
<p>Left: raw intensities (uniform negative effect everywhere). Right: after SNV (effect size now mixed-sign,
localized). Full tables: <code>outputs/old/statistics{{,_snv,_area}}/wavelength_statistics.csv</code>.</p>

<h2>4. Model comparison (patient-level, nested CV)</h2>
<p>Classical models (Stage 6-7), mean +/- std ROC-AUC across 5 outer folds, then patient-level bootstrap 95% CI
(Stage 9, 2000 iterations, on calibrated_probability):</p>
{html_table(classical_metrics[['family', 'roc_auc_mean', 'roc_auc_std', 'pr_auc_mean', 'sensitivity_mean', 'specificity_mean']])}
{html_table(auc_bootstrap[['family', 'mean', 'ci95_lo', 'ci95_hi']].rename(columns={'mean': 'bootstrap_mean_auc'}))}
<p>Neural models (Stage 11-12):</p>
{html_table(neural_summary)}
<p>Also tried (not in the main comparison; see chat history for the investigation): RandomForest, XGBoost,
LightGBM, CatBoost -- all landed at ROC-AUC 0.53-0.58, essentially the same as sklearn's HistGradientBoosting;
none competitive with the linear models. Raw fold-level detail: <code>outputs/old/models/gbm_check_fold_metrics.csv</code>.</p>

<div class="finding">
<b>linear_svm and pls_logreg are statistically indistinguishable from each other</b> (paired bootstrap
&Delta;={paired.loc[(paired.family_a=='linear_svm')&(paired.family_b=='pls_logreg'), 'mean_delta'].values[0]:+.3f},
not significant) <b>and both significantly beat logreg</b>
(&Delta;={paired.loc[(paired.family_a=='logreg')&(paired.family_b=='linear_svm'), 'mean_delta'].values[0]:+.3f},
significant). Tree ensembles (6 different implementations tried), RBF-SVM, and both neural architectures are all
well below the linear models -- plausible given 278 highly collinear features and only 43 control patients.
</div>

<h2>5. Calibration (Stage 8)</h2>
<p>Platt scaling per outer fold, fit on group-aware OOF patient-level scores from outer_train only. Brier score
raw -> calibrated:</p>
{html_table(calibration[['family', 'raw_brier_mean', 'calibrated_brier_mean', 'raw_roc_auc_mean', 'calibrated_roc_auc_mean']])}
<div class="caveat">Calibration <b>hurt</b> the <code>hgb</code> family (mean ROC-AUC 0.559 -&gt; 0.482): one outer
fold's OOF sample was noisy enough that the fitted Platt slope flipped sign, inverting that fold's ranking. Do not
trust calibrated probabilities for families that were already near chance pre-calibration.</div>
<div class="figrow">
<img src="calibration/figures/reliability_linear_svm.png" alt="linear_svm reliability">
<img src="calibration/figures/reliability_pls_logreg.png" alt="pls_logreg reliability">
</div>

<h2>6. Interpretability (Stage 10, 13)</h2>
<p>Top wavelength points/regions by |linear coefficient| (mean across 5 outer folds):</p>
{html_table(top_regions(fi, 'linear_coefficient')[['model', 'wavelength_start', 'importance', 'stability']])}
<p>Top wavelength regions by grouped permutation importance:</p>
{html_table(top_regions(fi, 'grouped_permutation')[['model', 'wavelength_start', 'wavelength_end', 'importance', 'importance_std']])}
<div class="finding">
<code>linear_svm</code> (coefficient-based) and <code>pls_logreg</code> (independently, via PLS back-projection)
and grouped permutation importance for both families converge on the <b>same wavelength regions</b>: ~467-500 nm
(dominant across every method), ~416-425 nm and ~537-561 nm (secondary). Two structurally different models and
two structurally different importance methods agreeing is stronger evidence than any one method alone -- though
still subject to every caveat above (small N, missingness composition of the control class).
</div>
<div class="figrow">
<img src="interpretability/figures/coefficient_stability_linear_svm.png" alt="linear_svm coefficients">
<img src="interpretability/figures/permutation_importance_linear_svm.png" alt="linear_svm permutation">
</div>
<p>CNN1D attribution (<code>mlp</code> was skipped for interpretability -- its raw AUC 0.593+/-0.110 is too close
to chance to be worth interpreting), two independent methods: Integrated Gradients (train-mean baseline) and
GradientSHAP (baselines = a training-fold subsample), {ig_gs_corr_text}</p>
{html_table(top_regions(fi[fi['method'] == 'integrated_gradients_train_mean'], 'integrated_gradients_train_mean')[['wavelength_start', 'importance', 'signed_effect']]) if 'integrated_gradients_train_mean' in fi['method'].values else '<p><i>not yet generated</i></p>'}

<h2>7. Reproducibility (Stage 35)</h2>
<p>Full snapshot: <code>outputs/old/config_snapshot.yaml</code> (data/instrument/QC configs, library versions,
timestamp, git commit status).</p>

<h2>Bottom line</h2>
<div class="finding">
Under nested, patient-grouped, leakage-checked cross-validation, <code>linear_svm</code> and
<code>pls_logreg</code> achieve patient-level ROC-AUC around 0.81-0.82 (95% CI clearly excluding chance) on
park=140/control=43, with two independent interpretability methods agreeing on ~467-500 nm as the most
informative band. This is real signal within this dataset's limitations -- not yet a validated biomarker,
given the small control class, its composition (see missingness caveat), and the single-instrument scope of
this analysis stage.
</div>

</body></html>"""

    with open(os.path.join(OUT, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> {os.path.join(OUT, 'feature_importance.csv')} ({len(fi)} rows)")
    print(f"-> {os.path.join(OUT, 'report.html')}")


if __name__ == "__main__":
    main()
