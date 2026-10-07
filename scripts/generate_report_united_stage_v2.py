"""Task 3 (report, v2): reads the CSV/JSON artifacts already produced under
outputs/united_multiclass_v2/ and outputs/harmonization_v2/ (does not
recompute anything) and writes outputs/united_multiclass_v2/report.html.
Mirrors scripts/generate_report_united_stage.py's structure, updated for:
  - the corrected (post-identity-fix) patient counts,
  - the new population-mean (diagnosis-blind) transfer curves,
  - the joint-FDR pairwise Mann-Whitney convention (primary result; per-pair
    convention kept alongside for a direct before/after comparison),
  - PLS/PCA-reduced classification (no raw-278-dim linear model),
  - no calibration section (skipped entirely this round).

Usage: python scripts/generate_report_united_stage_v2.py
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "united_multiclass_v2")
HARM = os.path.join(ROOT, "outputs", "harmonization_v2")
STALE = os.path.join(ROOT, "outputs", "united_multiclass")


def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def safe_read_json(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def html_table(df: pd.DataFrame, cols=None, float_fmt="{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "<p><i>not available</i></p>"
    if cols:
        df = df[cols]
    return df.to_html(index=False, float_format=lambda x: float_fmt.format(x) if pd.notna(x) else "")


def n_sig(df, col):
    return int((df[col] < 0.05).sum()) if df is not None else None


def main():
    audit = safe_read_json(os.path.join(OUT, "audit", "data_audit.json"))
    stale_audit = safe_read_json(os.path.join(STALE, "audit", "data_audit.json"))
    transfer_meta = safe_read_json(os.path.join(HARM, "transfer_functions_meta.json"))

    classical_metrics = safe_read_csv(os.path.join(OUT, "metrics.csv"))
    chosen_n_components = safe_read_csv(os.path.join(OUT, "chosen_n_components.csv"))
    predictions = safe_read_csv(os.path.join(OUT, "predictions", "patient_predictions.csv"))

    kw_raw = safe_read_csv(os.path.join(OUT, "statistics", "kruskal_wallis_omnibus.csv"))
    kw_snv = safe_read_csv(os.path.join(OUT, "statistics_snv", "kruskal_wallis_omnibus.csv"))
    pw_raw_pp = safe_read_csv(os.path.join(OUT, "statistics", "pairwise_mannwhitney_per_pair_fdr.csv"))
    pw_raw_joint = safe_read_csv(os.path.join(OUT, "statistics", "pairwise_mannwhitney_joint_fdr.csv"))
    pw_snv_pp = safe_read_csv(os.path.join(OUT, "statistics_snv", "pairwise_mannwhitney_per_pair_fdr.csv"))
    pw_snv_joint = safe_read_csv(os.path.join(OUT, "statistics_snv", "pairwise_mannwhitney_joint_fdr.csv"))
    stale_pw = safe_read_csv(os.path.join(STALE, "statistics", "pairwise_mannwhitney.csv"))

    band_summary = safe_read_csv(os.path.join(OUT, "interpretability", "hypothesis_band_replication_summary.csv"))

    by_inst = audit["by_instrument"] if audit else {}
    included = audit["included_class_counts"] if audit else {}
    excluded = audit["excluded_counts"] if audit else {}

    def pairwise_row(df, col, a_idx, b_idx):
        if df is None:
            return "n/a"
        sub = df[(df["class_a"] == a_idx) & (df["class_b"] == b_idx)]
        return f"{n_sig(sub, col)}/{len(sub)}" if len(sub) else "n/a"

    pair_names = [(0, 1, "control", "stage1"), (0, 2, "control", "stage2"), (0, 3, "control", "stage3"),
                  (1, 2, "stage1", "stage2"), (1, 3, "stage1", "stage3"), (2, 3, "stage2", "stage3")]

    # per-model macro-OvR-AUC patient-level summary (from raw OOF predictions,
    # no bootstrap script run in this pass -- classical_metrics.csv already has
    # outer-fold mean/std, which is the primary reported number here).
    n_comp_summary = None
    if chosen_n_components is not None and len(chosen_n_components):
        n_comp_summary = chosen_n_components.groupby("family")["n_components"].agg(["mean", "min", "max", "count"])
        n_comp_summary = n_comp_summary.reset_index()

    confusion_html = "<p><i>not available</i></p>"
    per_class_recall_html = "<p><i>not available</i></p>"
    fold_metrics_full = safe_read_json(os.path.join(OUT, "models", "fold_metrics_full.json"))
    if fold_metrics_full:
        rows = []
        for fam in sorted({r["family"] for r in fold_metrics_full}):
            fam_rows = [r for r in fold_metrics_full if r["family"] == fam]
            recalls = {}
            for r in fam_rows:
                for cls, val in r["per_class_recall"].items():
                    recalls.setdefault(cls, []).append(val)
            row = {"family": fam}
            for cls, vals in recalls.items():
                row[f"recall_class_{cls}"] = sum(vals) / len(vals)
            rows.append(row)
        recall_df = pd.DataFrame(rows)
        per_class_recall_html = html_table(recall_df)

    transfer_rows = ""
    if transfer_meta:
        for role, c in transfer_meta["curves"].items():
            b, a = c["similarity_before"], c["similarity_after"]
            transfer_rows += (f"<tr><td>{role}</td><td>{c['n_old_patients']}</td><td>{c['n_new_patients']}</td>"
                               f"<td>{b['pearson']:.4f}</td><td>{a['pearson']:.4f}</td>"
                               f"<td>{b['nrmse']:.4f}</td><td>{a['nrmse']:.4f}</td></tr>")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Parkinson's disease stage (Hoehn-Yahr, 4-class) classification v2 -- corrected identity, re-harmonized</title>
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
v2: corrected patient identity, re-harmonized, PLS/PCA-reduced, joint-FDR pairwise statistics</h1>

<div class="scope">
<b>Scope of this report.</b> This SUPERSEDES <code>outputs/united_multiclass/</code> (kept on disk,
visibly stale, for auditability), rebuilt after fixing a patient-identity bug (same-session two-hand
GUID pairs were double-counted as two separate patients: 88 pairs total across all 3 instruments) and
replacing the diagnosis-conditioned PD/control transfer-curve pair with a SINGLE, simpler,
diagnosis-BLIND population-mean transfer curve per new instrument (built from the area-normalized mean
spectrum over ALL patients of that instrument, not split by class -- see
<code>scripts/build_transfer_v2.py</code>, <code>outputs/harmonization_v2/</code>). Every classical model
head (except a no-feature dummy floor baseline) is now routed through a PLS or PCA dimensionality-reduction
step first (no raw-278-wavelength linear model this round) -- and calibration is skipped entirely per
task instruction. Pairwise Mann-Whitney multiple-comparison correction is now applied JOINTLY across all
6 class-pairs x 278 wavelengths (1668 p-values, one BH-FDR family) instead of independently per pair.
</div>

<div class="caveat">
<b>Read this before the numbers below.</b>
<ul>
<li><b>Instrument id is metadata only</b> -- never a predictor feature in any classifier.</li>
<li><b>Included classes are heavily 'old'-instrument-skewed</b>: {by_inst.get('old', {}).get('n_stage1', 'n/a')}/
{included.get('stage1', 'n/a')} stage1, {by_inst.get('old', {}).get('n_stage2', 'n/a')}/{included.get('stage2', 'n/a')}
stage2 and {by_inst.get('old', {}).get('n_stage3', 'n/a')}/{included.get('stage3', 'n/a')} stage3 patients come from
the 'old' instrument alone -- a real property of the cohort, not an artifact of this analysis.</li>
<li><b>Wavelength-shift diagnostic is inherited, not re-derived</b>: {transfer_meta['wavelength_shift_diagnostic'] if transfer_meta else 'n/a'}</li>
</ul>
</div>

<h2>1. Data (corrected identity + re-harmonized)</h2>
<p>Pooled total: {audit['n_patients_total_pooled'] if audit else 'n/a'} patients (canonical, real-person-level
id -- was {stale_audit['n_patients_total_pooled'] if stale_audit else 'n/a'} under the pre-fix raw-GUID
counting). Included in the 4-class analysis: {audit['n_patients_included_4class'] if audit else 'n/a'}
({included}). Excluded: {audit['n_patients_excluded'] if audit else 'n/a'} ({excluded}).</p>
<p>Per-instrument breakdown of the 4 included classes:</p>
<table><tr><th>instrument</th><th>patients</th><th>control</th><th>stage1</th><th>stage2</th><th>stage3</th>
<th>stage4 excl.</th><th>unknown excl.</th></tr>
{"".join(f"<tr><td>{role}</td><td>{by_inst[role]['n_patients']}</td><td>{by_inst[role]['n_control']}</td>"
         f"<td>{by_inst[role]['n_stage1']}</td><td>{by_inst[role]['n_stage2']}</td>"
         f"<td>{by_inst[role]['n_stage3']}</td><td>{by_inst[role]['n_stage4_excluded']}</td>"
         f"<td>{by_inst[role]['n_unknown_excluded']}</td></tr>"
         for role in ["old", "new_pd", "new_control"]) if by_inst else ""}
</table>

<h2>2. Harmonization v2: population-mean (diagnosis-blind) transfer curves</h2>
<p>One curve per new instrument, estimated from the area-normalized mean spectrum over ALL patients of that
instrument (not split by PD/control) -- see <code>outputs/harmonization_v2/</code>. Similarity of the new
instrument's own population mean to OLD's, before vs after applying the curve:</p>
<table><tr><th>instrument</th><th>n_old</th><th>n_new</th><th>pearson before</th><th>pearson after</th>
<th>nrmse before</th><th>nrmse after</th></tr>
{transfer_rows}
</table>

<h2>3. Statistics: Kruskal-Wallis omnibus + pairwise Mann-Whitney (joint vs per-pair FDR)</h2>
<p>Patient-level, median representative spectrum per patient (QC HARD_FAIL spectra excluded). Omnibus
Kruskal-Wallis across all 4 classes, FDR-BH across wavelengths (unchanged convention): raw
{n_sig(kw_raw, 'kruskal_p_fdr')}/{len(kw_raw) if kw_raw is not None else 'n/a'} significant wavelengths,
SNV {n_sig(kw_snv, 'kruskal_p_fdr')}/{len(kw_snv) if kw_snv is not None else 'n/a'}.</p>
<p><b>Pairwise Mann-Whitney, JOINT FDR (primary result: all 6 pairs x wavelengths corrected together)</b>
vs the old per-pair-independent convention, significant wavelength counts (raw / SNV):</p>
<table><tr><th>pair</th><th>joint FDR (raw)</th><th>per-pair FDR (raw)</th><th>stale (outputs/united_multiclass)</th>
<th>joint FDR (SNV)</th><th>per-pair FDR (SNV)</th></tr>
{"".join(f"<tr><td>{a_name} vs {b_name}</td>"
         f"<td>{pairwise_row(pw_raw_joint, 'mwu_p_fdr_joint', a, b)}</td>"
         f"<td>{pairwise_row(pw_raw_pp, 'mwu_p_fdr', a, b)}</td>"
         f"<td>{pairwise_row(stale_pw, 'mwu_p_fdr', a, b)}</td>"
         f"<td>{pairwise_row(pw_snv_joint, 'mwu_p_fdr_joint', a, b)}</td>"
         f"<td>{pairwise_row(pw_snv_pp, 'mwu_p_fdr', a, b)}</td></tr>"
         for a, b, a_name, b_name in pair_names)}
</table>
<p>Full tables: <code>outputs/united_multiclass_v2/statistics{{,_snv}}/*.csv</code>.</p>

<h2>4. Classification: 4-class nested patient-grouped CV, PLS/PCA-reduced (no calibration)</h2>
<p>Macro one-vs-rest ROC-AUC, balanced accuracy, macro F1 (mean +/- std across outer folds):</p>
{html_table(classical_metrics[['family', 'roc_auc_macro_ovr_mean', 'roc_auc_macro_ovr_std', 'balanced_accuracy_mean', 'macro_f1_mean']]) if classical_metrics is not None else '<p><i>not available</i></p>'}
<p>Per-class recall (raw OOF outer-test predictions, averaged across outer folds):</p>
{per_class_recall_html}
<p>Chosen PLS/PCA component count per family (across outer folds, nested-CV-selected):</p>
{html_table(n_comp_summary) if n_comp_summary is not None else '<p><i>not available</i></p>'}
<p>Full tables: <code>outputs/united_multiclass_v2/{{metrics.csv, chosen_n_components.csv,
predictions/patient_predictions.csv, models/fold_metrics_full.json}}</code>. No separate calibration step
this round (skipped per task instruction) -- probabilities above are raw model outputs.</p>

<h2>5. Interpretability: do the OLD-derived hypothesis bands replicate per stage?</h2>
<p>Per-class-vs-rest linear coefficients (pca_mnlogreg, pls_mnlogreg), back-projected through the
reduction step (PCA/PLS) and then through any fixed-scaling preprocessing step, into raw wavelength units,
aggregated into 5-point regions, ranked by |coefficient| across folds. Rank (1 = most important region)
of the 3 OLD-study hypothesis bands (~467-500 nm, ~416-425 nm, ~537-561 nm) within each class-vs-rest
comparison:</p>
{html_table(band_summary) if band_summary is not None else '<p><i>not available</i></p>'}
<p>Full per-class tables: <code>outputs/united_multiclass_v2/interpretability/coefficient_region_importance_*.csv</code>.</p>

<h2>Bottom line</h2>
<div class="finding">
See sections 1-5 above for the corrected, re-harmonized 4-class headline numbers. This remains an
exploratory, non-blinded, all-data pooled analysis (every labeled patient used for both training and
evaluation via nested CV) -- there is no blinded holdout for the stage task, so these numbers should be
read as descriptive/hypothesis-generating, not as a validated generalization estimate. The joint-FDR
pairwise convention (section 3) is intentionally more conservative than the stale per-pair convention used
in <code>outputs/united_multiclass/</code>; a shrinking significant-wavelength count between the two
conventions is EXPECTED, not a sign of a regression.
</div>

</body></html>"""

    with open(os.path.join(OUT, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> {os.path.join(OUT, 'report.html')}")


if __name__ == "__main__":
    main()
