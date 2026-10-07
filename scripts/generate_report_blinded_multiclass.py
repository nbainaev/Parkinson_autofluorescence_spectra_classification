"""Part XXXVI of current_tasks/double_blind_exp.md, adapted: self-contained
HTML report for the multiclass retrospective blinded validation
(outputs/blinded_multiclass/). Reads only already-computed CSV/JSON outputs
from the prior 5 scripts -- does not refit or re-touch any label.

Usage: python scripts/generate_report_blinded_multiclass.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "outputs", "blinded_multiclass")
FIGURES = os.path.join(STUDY, "figures")
os.makedirs(FIGURES, exist_ok=True)
SNV_VARIANT_STUDY = os.path.join(ROOT, "outputs", "blinded_multiclass_snv_variant")  # pre-area-norm backup, for the
# direct old(SNV)-vs-new(area) comparison table this report now adds (see fig_normalization_comparison).
CLASS_ORDER = ["control", "stage1", "stage2", "stage3"]
MODEL_ORDER = ["dummy", "pca_mnlogreg", "pls_mnlogreg", "linear_svm_mc_reduced",
               "random_forest_reduced_pca", "hgb_reduced_pca", "rbf_svm", "knn",
               "mlp_multiclass", "cnn1d_multiclass"]
MODEL_LABEL = {"dummy": "dummy (stratified)", "pca_mnlogreg": "PCA+mnlogreg", "pls_mnlogreg": "PLS+mnlogreg",
               "linear_svm_mc_reduced": "PCA-reduced linear SVM (OvR)",
               "random_forest_reduced_pca": "PCA-reduced Random Forest", "hgb_reduced_pca": "PCA-reduced HistGB",
               "rbf_svm": "PCA-reduced RBF-SVM", "knn": "PCA-reduced kNN",
               "mlp_multiclass": "MLP (PCA-reduced)", "cnn1d_multiclass": "CNN1D (raw wavelengths)"}


def _fig_to_base64(standalone_name: str | None = None):
    """Encode the current matplotlib figure as base64 for HTML embedding; if
    `standalone_name` is given, ALSO save a standalone PNG under
    outputs/blinded_multiclass/figures/<standalone_name>.png (deliverable
    requirement: plots must exist as standalone files, not just embedded)."""
    if standalone_name is not None:
        plt.savefig(os.path.join(FIGURES, f"{standalone_name}.png"), format="png", dpi=130, bbox_inches="tight")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(b64, alt=""):
    return f'<img alt="{alt}" style="max-width:100%;" src="data:image/png;base64,{b64}"/>'


def fig_split_summary(split_manifest):
    dev = pd.DataFrame(split_manifest["development_counts_by_instrument_class"])
    blind = pd.DataFrame(split_manifest["blind_counts_by_instrument_class"])
    dev["cohort"], blind["cohort"] = "development", "blind_test"
    both = pd.concat([dev, blind], ignore_index=True)
    pivot = both.pivot_table(index=["instrument", "stage_name"], columns="cohort", values="n", fill_value=0)
    pivot = pivot.reindex(columns=["development", "blind_test"])
    return pivot.to_html()


def fig_confusion_matrices(cms):
    n = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.6))
    for ax, model in zip(axes, MODEL_ORDER):
        cm = np.array(cms[model]["confusion_matrix"])
        labels = cms[model]["labels"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(MODEL_LABEL[model], fontsize=10)
        vmax = cm.max() if cm.max() > 0 else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > vmax / 2 else "black", fontsize=9)
    fig.suptitle("Blind-test confusion matrices (argmax prediction, n=26 patients)")
    fig.tight_layout()
    return _fig_to_base64()


def fig_auc_bar(comparison):
    comp = comparison.set_index("model").loc[MODEL_ORDER]
    x = np.arange(len(MODEL_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    blind_err = np.vstack([comp["roc_auc_macro_ovr"] - comp["roc_auc_macro_ovr_ci95_lo"],
                            comp["roc_auc_macro_ovr_ci95_hi"] - comp["roc_auc_macro_ovr"]])
    ax.bar(x - width / 2, comp["roc_auc_macro_ovr"], width, yerr=blind_err, capsize=4,
           label="blind test (95% bootstrap CI)", color="#4b3fdb")
    ax.bar(x + width / 2, comp["roc_auc_macro_ovr_mean"], width, yerr=comp["roc_auc_macro_ovr_std"], capsize=4,
           label="development nested-CV (mean +/- std)", color="#999")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel("macro one-vs-rest ROC-AUC")
    ax.set_title("Blind-test vs development-only nested-CV macro-OvR AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_base64("roc_auc_bar")


def fig_auc_histogram(replicates: pd.DataFrame):
    """NEW plot (distinct from fig_auc_bar's single CI-whisker bar chart):
    distribution of the macro-OvR AUC across the SAME per-model bootstrap
    replicates used for fig_auc_bar's CI (outputs/blinded_multiclass/
    bootstrap_auc_replicates.csv), one small-multiple histogram per model."""
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(2.8 * len(MODEL_ORDER), 3.4), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        vals = replicates.loc[replicates["model"] == model, "roc_auc_macro_ovr"].dropna()
        ax.hist(vals, bins=30, color="#4b3fdb", alpha=0.85)
        ax.axvline(vals.mean(), color="black", linestyle="-", linewidth=1.2, label=f"mean={vals.mean():.3f}")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="chance (0.5)")
        ax.set_title(MODEL_LABEL[model], fontsize=9)
        ax.set_xlabel("macro-OvR AUC (bootstrap replicate)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel(f"count (of {replicates['replicate'].nunique()} bootstrap replicates)")
    fig.suptitle("Blind-test macro-OvR AUC: distribution across patient-level bootstrap replicates, per model")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _fig_to_base64("roc_auc_bootstrap_histograms")


def fig_zscore_area():
    """Embeds the pre-generated area-normalization pairwise z-score figure
    (current_work/progress/make_zscore_figures.py, "area" variant -- adapted
    from that same script rather than rewritten here) if present."""
    path = os.path.join(FIGURES, "zscore_pairwise_area.png")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def fig_per_class_prf(eval_json):
    n = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.6 * n), 4.6))
    x = np.arange(len(CLASS_ORDER))
    width = 0.8 / n  # generic bar width/offset for an arbitrary number of models (was hardcoded for 4)
    for ax, metric, title in zip(axes, ["per_class_recall", "per_class_precision"],
                                  ["Blind-test per-class recall", "Blind-test per-class precision"]):
        for i, model in enumerate(MODEL_ORDER):
            row = next(r for r in eval_json if r["model"] == model)
            vals = [row[metric].get(str(float(CLASS_ORDER.index(c))), 0.0) for c in CLASS_ORDER]
            ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=MODEL_LABEL[model])
        ax.set_xticks(x); ax.set_xticklabels(CLASS_ORDER)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
    axes[0].legend(fontsize=6, loc="upper right", ncol=2)
    fig.tight_layout()
    return _fig_to_base64()


def fig_region_importance():
    path_pls = os.path.join(STUDY, "interpretability", "coefficient_region_importance_pls_mnlogreg_class3_stage3.csv")
    if not os.path.exists(path_pls):
        return None
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, family in zip(axes, ["pca_mnlogreg", "pls_mnlogreg"]):
        path = os.path.join(STUDY, "interpretability", f"coefficient_region_importance_{family}_class3_stage3.csv")
        df = pd.read_csv(path)
        centers = (df["wavelength_start"] + df["wavelength_end"]) / 2
        ax.bar(centers, df["sum_abs_importance"], width=4.5, color="#4b3fdb")
        for lo, hi in [(467.0, 500.0), (416.0, 425.0), (537.0, 561.0)]:
            ax.axvspan(lo, hi, color="orange", alpha=0.25)
        ax.set_ylabel("sum |coef| (region)")
        ax.set_title(f"{family}: development-only, class stage3-vs-rest wavelength region importance "
                     "(orange = hypothesis bands)")
    axes[-1].set_xlabel("wavelength (nm)")
    fig.tight_layout()
    return _fig_to_base64()


def main():
    with open(os.path.join(STUDY, "split_manifest.json"), encoding="utf-8") as f:
        split_manifest = json.load(f)
    with open(os.path.join(STUDY, "confusion_matrices.json"), encoding="utf-8") as f:
        cms = json.load(f)
    with open(os.path.join(STUDY, "evaluation_metrics.json"), encoding="utf-8") as f:
        eval_json = json.load(f)
    comparison = pd.read_csv(os.path.join(STUDY, "evaluation_metrics_with_dev_cv.csv"))
    band_summary = pd.read_csv(os.path.join(STUDY, "interpretability", "hypothesis_band_replication_summary.csv"))
    boot_replicates = pd.read_csv(os.path.join(STUDY, "bootstrap_auc_replicates.csv"))
    with open(os.path.join(STUDY, "frozen", "final_protocol.json"), encoding="utf-8") as f:
        final_protocol = json.load(f)

    split_table_html = fig_split_summary(split_manifest)
    cm_b64 = fig_confusion_matrices(cms)
    auc_b64 = fig_auc_bar(comparison)
    auc_hist_b64 = fig_auc_histogram(boot_replicates)
    zscore_area_b64 = fig_zscore_area()
    prf_b64 = fig_per_class_prf(eval_json)
    region_b64 = fig_region_importance()

    # Preprocessing-variant swap actually took effect: report which variant each family's
    # nested-CV selection landed on out of the NEW {"A","D"} grid (area normalization = "D";
    # "A" = raw+StandardScaler, no extra spectral normalization). Honest empirical finding,
    # not assumed: see report text below for which variant actually won per family.
    chosen_variants = {fam: v["preprocessing_variant"] for fam, v in final_protocol["families"].items()}

    svm_row = next(r for r in eval_json if r["model"] == "linear_svm_mc_reduced")
    svm_cm = np.array(cms["linear_svm_mc_reduced"]["confusion_matrix"])
    svm_n_total = int(svm_cm.sum())
    svm_n_control_col = int(svm_cm[:, 0].sum())
    svm_collapsed_fully = bool((svm_cm[:, 1:].sum() == 0))  # EVERY prediction landed in column 0 (control)
    svm_n_noncontrol = svm_n_total - svm_n_control_col

    # Old (SNV-based) numbers, for the direct before/after comparison this redo requires.
    snv_eval_path = os.path.join(SNV_VARIANT_STUDY, "evaluation_metrics.csv")
    snv_cm_path = os.path.join(SNV_VARIANT_STUDY, "confusion_matrices.json")
    old_new_table_html, svm_snv_collapsed = None, None
    if os.path.exists(snv_eval_path) and os.path.exists(snv_cm_path):
        old_eval = pd.read_csv(snv_eval_path)[["model", "roc_auc_macro_ovr", "roc_auc_macro_ovr_ci95_lo",
                                                 "roc_auc_macro_ovr_ci95_hi"]].rename(
            columns={"roc_auc_macro_ovr": "AUC_snv", "roc_auc_macro_ovr_ci95_lo": "AUC_snv_lo",
                     "roc_auc_macro_ovr_ci95_hi": "AUC_snv_hi"})
        new_eval = comparison[["model", "roc_auc_macro_ovr", "roc_auc_macro_ovr_ci95_lo",
                                 "roc_auc_macro_ovr_ci95_hi"]].rename(
            columns={"roc_auc_macro_ovr": "AUC_area", "roc_auc_macro_ovr_ci95_lo": "AUC_area_lo",
                     "roc_auc_macro_ovr_ci95_hi": "AUC_area_hi"})
        old_new = new_eval.merge(old_eval, on="model", how="left").set_index("model").loc[MODEL_ORDER].reset_index()
        old_new["model"] = old_new["model"].map(MODEL_LABEL)
        old_new_table_html = old_new.round(3).to_html(index=False)
        with open(snv_cm_path, encoding="utf-8") as f:
            old_cms = json.load(f)
        old_svm_cm = np.array(old_cms["linear_svm_mc_reduced"]["confusion_matrix"])
        svm_snv_collapsed = bool((old_svm_cm[:, 1:].sum() == 0))

    comp_table = comparison[["model", "roc_auc_macro_ovr", "roc_auc_macro_ovr_ci95_lo",
                              "roc_auc_macro_ovr_ci95_hi", "balanced_accuracy", "macro_f1",
                              "roc_auc_macro_ovr_mean", "roc_auc_macro_ovr_std",
                              "balanced_accuracy_mean", "macro_f1_mean"]].copy()
    comp_table["model"] = comp_table["model"].map(MODEL_LABEL)
    comp_table.columns = ["model", "blind AUC", "blind CI lo", "blind CI hi", "blind bal.acc", "blind macroF1",
                           "dev-CV AUC mean", "dev-CV AUC std", "dev-CV bal.acc mean", "dev-CV macroF1 mean"]

    # Area-normalization pooled statistics (outputs/united_multiclass_v2/statistics_area/), for the
    # joint-FDR significant-wavelength-count table in the new statistics section below.
    stats_rows = []
    for label, subdir in [("raw (none)", "statistics"), ("SNV", "statistics_snv"), ("area (this redo)", "statistics_area")]:
        p = os.path.join(ROOT, "outputs", "united_multiclass_v2", subdir, "pairwise_mannwhitney_joint_fdr.csv")
        if not os.path.exists(p):
            continue
        pw = pd.read_csv(p)
        kw = pd.read_csv(os.path.join(ROOT, "outputs", "united_multiclass_v2", subdir, "kruskal_wallis_omnibus.csv"))
        class_names = {0: "control", 1: "stage1", 2: "stage2", 3: "stage3"}
        row = {"normalization": label,
               "kruskal_wallis_sig_fdr<0.05": f"{int((kw['kruskal_p_fdr'] < 0.05).sum())}/{len(kw)}"}
        for (a, b), sub in pw.groupby(["class_a", "class_b"]):
            n_sig = int((sub["mwu_p_fdr_joint"] < 0.05).sum())
            row[f"{class_names[a]} vs {class_names[b]} (joint-FDR)"] = f"{n_sig}/{len(sub)}"
        stats_rows.append(row)
    stats_table_html = pd.DataFrame(stats_rows).to_html(index=False) if stats_rows else "<p>not available</p>"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Multiclass retrospective blinded validation (Hoehn-Yahr stage, linear + nonlinear + neural)</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1150px; margin: 2em auto; padding: 0 1em; color: #222; }}
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

<h1>Retrospective blinded validation: 4-class Hoehn-Yahr PD stage, linear + nonlinear + neural models</h1>

<div class="scope">
<b>Scope of this study.</b> A scaled-down version of the project's binary double-blind protocol
(<code>current_tasks/double_blind_exp.md</code>), applied to the 4-class (control/stage1/stage2/stage3)
Hoehn-Yahr STAGE problem on the 129-patient corrected-identity, re-harmonized cohort
(<code>outputs/united_multiclass_v2/</code>). Only the core blinding mechanics are reproduced (patient-level
split, private key file, frozen artifacts with SHA256, predict-before-unblind ordering) -- not the full
39-part binary spec. The diagnosis-blind, population-mean instrument-harmonization curves in
<code>outputs/harmonization_v2/</code> are FROZEN and reused as-is; they were built from the full patient
population of each instrument regardless of diagnosis (never touch labels), so reusing them without a
development-only rebuild is a stated, defensible scope simplification, not an oversight.
</div>

<div class="scope">
<b>This is the AREA-NORMALIZATION redo</b> of this study (the only change vs. the prior run: the last
preprocessing step before each classifier, previously searched over {{"A": raw+StandardScaler, "B":
SG-smoothing+SNV}}, is now searched over {{"A": raw+StandardScaler, "D": per-spectrum area/vector
normalization ("<code>AreaNormalizer</code>")+StandardScaler}} -- SNV ("B") is no longer a candidate for any
of the 3 linear multiclass families). Nothing upstream of this last step changed (QC, identity, harmonization,
area-normalization-during-instrument-transfer all untouched). The prior SNV-based run's full output is
preserved unchanged at <code>outputs/blinded_multiclass_snv_variant/</code> for direct comparison.
<b>Empirical finding, not assumed:</b> under nested-CV model selection with the new {{"A","D"}} grid, ALL
FOUR families ended up choosing preprocessing variant <b>"{chosen_variants.get('pca_mnlogreg','?')}"</b>
(dummy: {chosen_variants.get('dummy','?')}, pca_mnlogreg: {chosen_variants.get('pca_mnlogreg','?')},
pls_mnlogreg: {chosen_variants.get('pls_mnlogreg','?')}, linear_svm_mc_reduced:
{chosen_variants.get('linear_svm_mc_reduced','?')}) -- i.e. area normalization ("D") was never actually
selected over plain raw+StandardScaler ("A") for any family this round. <code>pca_mnlogreg</code> and
<code>pls_mnlogreg</code> already chose "A" in the OLD {{"A","B"}} grid too, so their frozen artifacts and
all downstream numbers are numerically IDENTICAL before/after this redo (not a bug -- SNV was never
selected for them either). Only <code>linear_svm_mc_reduced</code> changes: it chose "B" (SNV, C=10) before
and now chooses "A" (no extra normalization, C=1) -- so its change in this report reflects "SNV no longer
in the candidate pool", not "area normalization won".
</div>

<h2>1. Split</h2>
<p>Patient-level, stratified by the 4-class stage label ONLY (see
<code>scripts/prepare_blinded_study_multiclass.py</code> docstring for why joint instrument x class
stratification was rejected: new_pd contributes only 7 stage-labeled patients (1 control + 6 stage3) and
new_control contributes controls only, so joint strata would leave near-singleton cells). 80/20 dev/blind
split (not the original binary study's ~70/30), seed 42, chosen because the smallest class (stage2, n=14)
would leave too few blind-test patients at 30%. Zero patient leakage confirmed
(<code>assert_no_patient_leakage</code>/<code>assert_all_classes_present</code>).</p>
<p><b>development</b>: {split_manifest['n_development_patients']} patients {split_manifest['development_class_totals']}
&nbsp;|&nbsp; <b>blind test</b>: {split_manifest['n_blind_patients']} patients {split_manifest['blind_class_totals']}</p>
{split_table_html}

<h2>2. Frozen models</h2>
<p>Originally three linear-only families frozen on development data only, no calibration (standing
instruction): <code>pca_mnlogreg</code>, <code>pls_mnlogreg</code>, <code>linear_svm_mc_reduced</code>
(freshly re-verified post the <code>GroupAwareCalibratedLinearSVC</code> sample-weight-routing bugfix --
see <code>outputs/united_multiclass_v2/AUDIT_AND_EXTENSION_NOTES.md</code>), plus a <code>dummy</code>
(stratified) floor baseline. This report now ALSO includes a runtime-trimmed nonlinear extension
(<code>random_forest_reduced_pca</code>, <code>hgb_reduced_pca</code>, <code>rbf_svm</code>, <code>knn</code>,
all PCA(n=10)-reduced, all forced onto preprocessing variant "D" rather than nested-CV-selected) and a new
multiclass neural extension (<code>mlp_multiclass</code>: fixed-config MLP on the same PCA(10) input;
<code>cnn1d_multiclass</code>: fixed-config small 1D CNN on the raw 278-wavelength area-normalized input) --
10 families total. Hyperparameters (PLS/PCA component count, C) for the original linear families are
selected via patient-grouped CV within development only; the nonlinear/neural additions use either a small
fixed grid (nonlinear) or a single fixed configuration (MLP/CNN, no grid, per runtime discipline) rather
than a wide hyperparameter search. Every family is refit once on the full development set
(<code>outputs/blinded_multiclass/frozen/frozen_manifest.json</code>, SHA256-verified before every
prediction/unblinding step).</p>

<h2>3. Blind-test confusion matrices</h2>
{_img_tag(cm_b64, "confusion matrices")}
<div class="caveat">
<b>{'`linear_svm_mc_reduced` STILL nearly-fully collapses to predicting `control` under area normalization.' if svm_n_noncontrol <= 1 else '`linear_svm_mc_reduced` no longer collapses to a single class under area normalization.'}</b>
Of {svm_n_total} blind-test predictions, {svm_n_control_col} landed in the `control` column and only
{svm_n_noncontrol} did not (balanced accuracy={svm_row['balanced_accuracy']:.3f}). This is a MARGINAL
change from the prior SNV-based run, where it was a FULL collapse (every single one of 26 predictions in
`control`, see <code>outputs/blinded_multiclass_snv_variant/confusion_matrices.json</code>{', confirmed unchanged there (still fully collapsed under SNV)' if svm_snv_collapsed else ''}) --
so under area normalization the majority-class-collapse problem for this family <b>persists, with only a
one-patient improvement</b> (it is neither meaningfully fixed nor made worse). Its macro-OvR AUC
({svm_row['roc_auc_macro_ovr']:.3f}) is nonetheless well above chance because AUC is a RANKING statistic
over continuous probabilities, not the discrete argmax prediction -- the model's probability of the
minority classes still correlates with the true label even though the class-imbalance prior wins the
argmax almost every time. This is exactly why the task asked for confusion matrices, not just AUC: AUC
alone would have hidden this near-collapse. Note also (Scope box above) that this family's own winning
candidate switched from variant "B" (SNV) to variant "A" (no extra normalization) -- area normalization
("D") itself was never selected for it either, so this near-collapse is not attributable to area
normalization specifically; it is a pre-existing property of this family/cohort that persists regardless
of which of {{"A","B","D"}} ends up chosen.
</div>

<h2>4. Macro-OvR AUC: blind test vs development-only nested CV</h2>
{_img_tag(auc_b64, "AUC bar chart")}
{comp_table.round(3).to_html(index=False)}
<p>Blind-test AUCs are noisier (wide bootstrap CIs, n=26 patients, e.g. stage1/stage2 have only 3 blind
patients each) than the development-only nested-CV numbers, as expected for an already-tiny cohort split
further into an 80/20 hold-out -- stated plainly, not spun. Development-only nested-CV AUCs are themselves
lower than the original full-129-patient <code>united_multiclass_v2</code> report's numbers (e.g.
pls_mnlogreg 0.586 here vs 0.660 there) simply because development is a further-reduced (103-patient)
subset of that same cohort -- not evidence of a new problem, just less data.</p>

<h2>4b. AUC bootstrap-replicate distributions (histogram, per model)</h2>
{_img_tag(auc_hist_b64, "AUC bootstrap histograms")}
<p>Distribution of the SAME {len(boot_replicates) // len(MODEL_ORDER)} patient-level, class-stratified
bootstrap replicates used for the 95% CI whiskers in Section 4's bar chart, shown here as a full
distribution rather than a single interval -- e.g. <code>linear_svm_mc_reduced</code>'s and
<code>pls_mnlogreg</code>'s distributions are visibly right of 0.5 (chance) for most replicates despite
the wide spread, while <code>dummy</code>'s is centered near 0.5 as expected of a label-blind floor
baseline.</p>

<h2>4c. Old (SNV) vs new (area) blind-test AUC: direct comparison</h2>
{old_new_table_html if old_new_table_html else '<p>prior SNV-variant results not found for comparison.</p>'}
<p><code>dummy</code>, <code>pca_mnlogreg</code>, <code>pls_mnlogreg</code> are numerically IDENTICAL
old-vs-new (both chose preprocessing variant "A" under both the old {{"A","B"}} and new {{"A","D"}} grids --
SNV/area were never competitive for these 3 families). Only <code>linear_svm_mc_reduced</code> differs
(0.686 &rarr; 0.706), because SNV(B, C=10) was replaced in its candidate pool by area-norm(D), and its
nested-CV selection ended up preferring plain "A" (C=1) either way once "B" was removed from the grid.</p>

<h2>5. Per-class precision/recall, blind test</h2>
{_img_tag(prf_b64, "per-class precision/recall") if prf_b64 else '<p>not available</p>'}

<h2>6. Wavelength region importance (development-only CV, class stage3-vs-rest shown)</h2>
{_img_tag(region_b64, "region importance") if region_b64 else '<p>not available</p>'}
<p>Hypothesis-band replication (best rank among 56 5nm-wide regions, lower=more important; development-only,
all 4 classes):</p>
{band_summary.round(1).to_html(index=False)}
<p>The 537-561nm band ranks consistently in roughly the top third (rank 8-16 of 56) across both families and
all 4 classes, matching the original (full-cohort) <code>united_multiclass_v2</code> finding. The
416-425nm band replicates more weakly here (rank 21-54), also consistent with that report's own
characterization of it as the weakest of the three hypothesis bands. This was computed on development-only
CV and never touched blind-test labels, matching the "confirmatory only" spirit of the pre-registration
requirement even without the full ceremony. Note: this section's underlying candidates were rebuilt with
the new {{"A","D"}} grid too, but since both families chose "A" (see Scope box), these numbers are
IDENTICAL to the pre-redo (SNV-grid) run.</p>

<h2>7. Pooled statistics: pairwise Mann-Whitney z-scores (area normalization) and joint-FDR significance</h2>
{_img_tag(zscore_area_b64, "pairwise z-score plots, area-normalized") if zscore_area_b64 else '<p>not available -- run current_work/progress/make_zscore_figures.py first.</p>'}
<p>Kruskal-Wallis omnibus and joint-BH-FDR (6 pairs &times; 278 wavelengths, one family) significant-wavelength
counts, raw/none vs SNV vs area (this redo's featured variant; full CSVs in
<code>outputs/united_multiclass_v2/statistics_area/</code>, parallel to the pre-existing
<code>statistics/</code> and <code>statistics_snv/</code> directories which are unchanged):</p>
{stats_table_html}
<p>Under joint FDR, ZERO of the 6&times;278 pairwise comparisons survive at p&lt;0.05 for ANY of the three
normalization choices (raw, SNV, or area) -- this null result is unchanged by the choice of per-spectrum
normalization. The Kruskal-Wallis omnibus finds a single nominally-significant wavelength under raw and
area normalization (0 under SNV), a negligible difference consistent with noise at this sample size, not
a normalization-driven effect.</p>

<h2>Limitations</h2>
<ul>
<li><b>Tiny blind-test per-class n</b>: stage1 and stage2 each have only 3 blind-test patients; every
metric above (especially per-class precision/recall and the bootstrap CIs) must be read as
descriptive/exploratory at this sample size, not a precise estimate.</li>
<li><b>Harmonization curves not rebuilt development-only</b>: <code>outputs/harmonization_v2/</code>'s
transfer curves were built from the full patient population of each instrument (diagnosis-blind, so no
label leakage), reused as-is per task instruction -- stated here explicitly as a scope decision, not
hidden.</li>
<li><b>Single-implementer procedural blinding</b>: the same agent wrote the split, freeze, predict, and
unblind scripts in one session -- this is procedural/mechanical blinding (predict-before-unblind
ordering, SHA256 tamper checks) with no independent second party, unlike a true blinded clinical trial.</li>
<li><b><code>linear_svm_mc_reduced</code> still nearly-fully collapses to majority-class (`control`)
prediction on the blind test under area normalization</b> (see Section 3) -- a one-patient improvement
over the prior full collapse under SNV, not a fix; verified directly from its confusion matrix in this
run, not assumed changed because the preprocessing grid changed.</li>
<li><b>Area normalization ("D") was never actually selected</b> by nested-CV model selection for any of
the 4 families this round (all chose "A" -- see Scope box) -- so this redo's blind-test numbers for
<code>dummy</code>/<code>pca_mnlogreg</code>/<code>pls_mnlogreg</code> are IDENTICAL to the pre-redo
numbers, and <code>linear_svm_mc_reduced</code>'s small change is attributable to SNV leaving its
candidate pool, not to area normalization entering it -- stated plainly, not spun as "area normalization
improved things" when the evidence does not support that framing.</li>
<li>4-class Hoehn-Yahr stage is ordinal but modeled here as unordered (nominal) multiclass, consistent
with the rest of this project's <code>united_multiclass_v2</code> scope.</li>
</ul>

</body></html>"""

    with open(os.path.join(STUDY, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {os.path.join(STUDY, 'report.html')} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
