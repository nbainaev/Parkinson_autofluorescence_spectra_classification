"""Task 3 (report): reads the CSV/JSON artifacts already produced under
outputs/united_multiclass/ (does not recompute anything) and writes
outputs/united_multiclass/report.html. Mirrors scripts/generate_report_united.py's
structure, generalized to the 4-class (Hoehn-Yahr PD stage) task.

Usage: python scripts/generate_report_united_stage.py
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united_multiclass")
CLASS_NAMES_ORDER = ["control", "stage1", "stage2", "stage3"]


def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def html_table(df: pd.DataFrame, cols=None, float_fmt="{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "<p><i>not available</i></p>"
    if cols:
        df = df[cols]
    return df.to_html(index=False, float_format=lambda x: float_fmt.format(x) if pd.notna(x) else "")


def n_sig(df, col):
    return int((df[col] < 0.05).sum()) if df is not None else None


def main():
    audit = json.load(open(os.path.join(OUT, "audit", "data_audit.json")))
    classical_metrics = safe_read_csv(os.path.join(OUT, "metrics.csv"))
    bootstrap = safe_read_csv(os.path.join(OUT, "bootstrap", "bootstrap_summary.csv"))
    calibration = safe_read_csv(os.path.join(OUT, "calibration", "calibration_summary.csv"))
    kw_raw = safe_read_csv(os.path.join(OUT, "statistics", "kruskal_wallis_omnibus.csv"))
    kw_snv = safe_read_csv(os.path.join(OUT, "statistics_snv", "kruskal_wallis_omnibus.csv"))
    pw_raw = safe_read_csv(os.path.join(OUT, "statistics", "pairwise_mannwhitney.csv"))
    pw_snv = safe_read_csv(os.path.join(OUT, "statistics_snv", "pairwise_mannwhitney.csv"))
    band_summary = safe_read_csv(os.path.join(OUT, "interpretability", "hypothesis_band_replication_summary.csv"))

    auc_bootstrap = bootstrap[bootstrap["metric"] == "roc_auc_macro_ovr"].sort_values("mean", ascending=False) \
        if bootstrap is not None else None

    by_inst = audit["by_instrument"]
    included = audit["included_class_counts"]
    excluded = audit["excluded_counts"]

    def pairwise_row(df, a_name, b_name, a_idx, b_idx):
        if df is None:
            return "n/a"
        sub = df[(df["class_a"] == a_idx) & (df["class_b"] == b_idx)]
        return f"{n_sig(sub, 'mwu_p_fdr')}/{len(sub)}" if len(sub) else "n/a"

    pair_names = [(0, 1, "control", "stage1"), (0, 2, "control", "stage2"), (0, 3, "control", "stage3"),
                  (1, 2, "stage1", "stage2"), (1, 3, "stage1", "stage3"), (2, 3, "stage2", "stage3")]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Parkinson's disease stage (Hoehn-Yahr, 4-class) classification -- pooled harmonized cohort</title>
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
code {{ background: #f4f4f4; padding: 1px 4px; }}
</style></head><body>

<h1>Parkinson's disease severity (Hoehn-Yahr stage 0-3) classification from skin autofluorescence spectra --
pooled, harmonized, all-instrument cohort</h1>

<div class="scope">
<b>Scope of this report.</b> This is a 4-CLASS analysis (control / stage1 / stage2 / stage3 Hoehn-Yahr PD
severity), built on top of the UNCHANGED pooled/harmonized dataset from <code>spectra_ml.united.
build_united_dataset()</code> (same diagnosis-independent, per-instrument harmonization and QC as
<code>outputs/united/</code>'s binary PD-vs-control analysis). The Hoehn-Yahr stage column is joined
additively by patient GUID from the raw table -- see <code>spectra_ml.united_stage</code>. This is a separate,
additive analysis from <code>outputs/united/</code> (binary) and does not modify it. Model family scope was
deliberately kept small for runtime (dummy-stratified, multinomial logreg, linear SVM one-vs-rest (calibrated),
PLS+multinomial-logreg only -- no neural nets, no tree ensembles, no extended hyperparameter grids).
</div>

<div class="caveat">
<b>Read this before the numbers below.</b>
<ul>
<li><b>Instrument id is metadata only</b> -- never a predictor feature in any classifier.</li>
<li><b>Included classes are heavily 'old'-instrument-skewed</b>: {by_inst['old']['n_stage1']}/
{included['stage1']} stage1, {by_inst['old']['n_stage2']}/{included['stage2']} stage2 and
{by_inst['old']['n_stage3']}/{included['stage3']} stage3 patients come from the 'old' instrument alone; new_pd/
new_control contribute mostly control patients and a handful of stage3. This is a real property of the cohort
(department-level recruitment), not an artifact of this analysis -- stated plainly, not obscured.</li>
<li><b>Data audit verification note:</b> {audit['verification_note']}</li>
</ul>
</div>

<h2>1. Data (Task 1)</h2>
<p>{audit['n_patients_total_pooled']} patients pooled (GUID-level, all 3 instruments). Included in the 4-class
analysis: {audit['n_patients_included_4class']} ({included}). Excluded: {audit['n_patients_excluded']}
({excluded}) -- stage4 (n={excluded['stage4_excluded_small_n']}, too few to model) and unknown/missing stage
(n={excluded['unknown_missing_stage']}), both logged in
<code>outputs/united_multiclass/audit/patient_summary.csv</code>, never silently dropped.</p>
<p>Per-instrument breakdown of the 4 included classes:</p>
<table><tr><th>instrument</th><th>patients</th><th>control</th><th>stage1</th><th>stage2</th><th>stage3</th>
<th>stage4 excl.</th><th>unknown excl.</th></tr>
{"".join(f"<tr><td>{role}</td><td>{by_inst[role]['n_patients']}</td><td>{by_inst[role]['n_control']}</td>"
         f"<td>{by_inst[role]['n_stage1']}</td><td>{by_inst[role]['n_stage2']}</td>"
         f"<td>{by_inst[role]['n_stage3']}</td><td>{by_inst[role]['n_stage4_excluded']}</td>"
         f"<td>{by_inst[role]['n_unknown_excluded']}</td></tr>"
         for role in ["old", "new_pd", "new_control"])}
</table>

<h2>2. Statistics (Task 2): Kruskal-Wallis omnibus + pairwise Mann-Whitney (tie-corrected z-score)</h2>
<p>Patient-level, median representative spectrum per patient (QC HARD_FAIL spectra excluded). Omnibus
Kruskal-Wallis across all 4 classes, FDR-BH across wavelengths: raw
{n_sig(kw_raw, 'kruskal_p_fdr')}/{len(kw_raw) if kw_raw is not None else 'n/a'} significant wavelengths,
SNV {n_sig(kw_snv, 'kruskal_p_fdr')}/{len(kw_snv) if kw_snv is not None else 'n/a'}.</p>
<p>Pairwise post-hoc (Mann-Whitney, FDR-BH within each pair across wavelengths), significant wavelength counts:</p>
<table><tr><th>pair</th><th>raw</th><th>SNV</th></tr>
{"".join(f"<tr><td>{a_name} vs {b_name}</td><td>{pairwise_row(pw_raw, a_name, b_name, a, b)}</td>"
         f"<td>{pairwise_row(pw_snv, a_name, b_name, a, b)}</td></tr>"
         for a, b, a_name, b_name in pair_names)}
</table>
<p>Full tables: <code>outputs/united_multiclass/statistics{{,_snv}}/{{kruskal_wallis_omnibus,pairwise_mannwhitney}}.csv</code>.</p>

<h2>3. Classification (Task 3): 4-class nested patient-grouped CV</h2>
<p>Macro one-vs-rest ROC-AUC, balanced accuracy, macro F1 (mean +/- std across outer folds):</p>
{html_table(classical_metrics[['family', 'roc_auc_macro_ovr_mean', 'roc_auc_macro_ovr_std', 'balanced_accuracy_mean', 'macro_f1_mean']]) if classical_metrics is not None else '<p><i>not available</i></p>'}
<p>Patient-level, class-stratified bootstrap 95% CI for macro-OvR-AUC (on calibrated probabilities,
750 iterations by default -- see <code>scripts/run_bootstrap_united_stage.py</code>):</p>
{html_table(auc_bootstrap[['family', 'mean', 'ci95_lo', 'ci95_hi']].rename(columns={'mean': 'bootstrap_mean_macro_auc'})) if auc_bootstrap is not None else '<p><i>bootstrap not available</i></p>'}

<h2>4. Calibration</h2>
<p>Per-class one-vs-rest Platt scaling (fit on group-aware OOF patient-level scores from outer_train only, then
renormalized to a valid probability simplex -- see <code>spectra_ml.multiclass._calibrate_multiclass_ovr</code>).
macro-OvR-AUC / multiclass Brier, raw -> calibrated:</p>
{html_table(calibration[['family', 'raw_roc_auc_macro_ovr_mean', 'calibrated_roc_auc_macro_ovr_mean', 'raw_brier_multiclass_mean', 'calibrated_brier_multiclass_mean']]) if calibration is not None else '<p><i>not available</i></p>'}

<h2>5. Interpretability: do the OLD-derived hypothesis bands replicate per stage?</h2>
<p>Per-class-vs-rest linear coefficients (mn_logreg, pls_mnlogreg), back-projected to raw wavelength units where
possible, aggregated into 5-point regions, ranked by |coefficient| across folds. Rank (1 = most important region)
of the 3 OLD-study hypothesis bands (~467-500 nm, ~416-425 nm, ~537-561 nm) within each class-vs-rest
comparison:</p>
{html_table(band_summary) if band_summary is not None else '<p><i>not available</i></p>'}
<p>Full per-class tables: <code>outputs/united_multiclass/interpretability/coefficient_region_importance_*.csv</code>.</p>

<h2>Bottom line</h2>
<div class="finding">
See sections 2-5 above for the 4-class headline numbers. This is an exploratory, non-blinded, all-data pooled
analysis (every labeled patient used for both training and evaluation via nested CV) -- there is no blinded
holdout for the stage task (unlike <code>outputs/final_study_unified/</code>'s binary Phase-2 protocol), so
these numbers should be read as descriptive/hypothesis-generating, not as a validated generalization estimate.
</div>

</body></html>"""

    with open(os.path.join(OUT, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> {os.path.join(OUT, 'report.html')}")


if __name__ == "__main__":
    main()
