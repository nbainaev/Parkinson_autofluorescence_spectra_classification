"""Integrity-check, unblind, and evaluate the pooled harmonized test cohort."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from spectra_ml.evaluation import compute_patient_metrics
from spectra_ml.integrity import sha256_file, verify_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "outputs", "final_study_unified")

def boot(y,p,n=2000,seed=42):
    y,p=np.asarray(y),np.asarray(p); rng=np.random.default_rng(seed); vals=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))==2: vals.append(roc_auc_score(y[i],p[i]))
    return {"median":float(np.median(vals)),"ci95_low":float(np.percentile(vals,2.5)),
            "ci95_high":float(np.percentile(vals,97.5)),"valid_replicates":len(vals)}

def main():
    fm=os.path.join(STUDY,"frozen","frozen_manifest.json"); pm=os.path.join(STUDY,"blinded_predictions_manifest.json")
    verify_manifest(fm,base_dir=STUDY); verify_manifest(pm,base_dir=STUDY)
    pred_meta=json.load(open(pm,encoding="utf-8"))
    if pred_meta["frozen_manifest_sha256"] != sha256_file(fm): raise RuntimeError("Frozen manifest changed")
    pred=pd.read_csv(os.path.join(STUDY,"blinded_predictions.csv"),dtype={"study_id":str})
    key=pd.read_csv(os.path.join(STUDY,"private","blinding_key.csv"),dtype={"study_id":str})
    ev=pred.merge(key[["study_id","target","instrument_role"]],on="study_id",validate="many_to_one")
    ev.to_csv(os.path.join(STUDY,"unblinded_predictions.csv"),index=False)
    reports=[]
    for (branch,model),g in ev.groupby(["analysis_branch","model"]):
        for subgroup,sub in [("pooled_primary",g)]+[(f"instrument_{r}",q) for r,q in g.groupby("instrument_role")]:
            if sub.target.nunique()<2: metrics={"n_patients":sub.study_id.nunique(),"roc_auc":None}
            else:
                metrics=compute_patient_metrics(sub.target.to_numpy(),sub.predicted_probability_PD.to_numpy(),
                                                threshold=float(sub.frozen_threshold.iloc[0]))
                metrics["roc_auc_bootstrap"]=boot(sub.target,sub.predicted_probability_PD)
            reports.append({"analysis_branch":branch,"model":model,"cohort":subgroup,**metrics})
    with open(os.path.join(STUDY,"evaluation_metrics.json"),"w",encoding="utf-8") as f: json.dump(reports,f,indent=2)
    table=pd.json_normalize(reports); table.to_csv(os.path.join(STUDY,"evaluation_metrics.csv"),index=False)
    primary=table[table.cohort=="pooled_primary"]
    html="""<!doctype html><html><head><meta charset='utf-8'><title>Unified harmonized PD study</title>
<style>body{font-family:sans-serif;max-width:1200px;margin:2em auto}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.35em}</style></head><body>
<h1>Unified-cohort retrospective blinded hold-out validation</h1>
<p>Updated design: diagnosis-independent global empirical mean-spectrum instrument harmonization was built first using all available patient spectra after predefined QC. Patients from OLD, NEW-1 and NEW-2 were then pooled and split at patient level, stratified by instrument and diagnosis. Instrument is metadata, not a diagnostic predictor.</p>
<h2>Primary pooled test endpoint</h2>"""+primary.to_html(index=False)+"<h2>Secondary instrument subgroups</h2>"+table.to_html(index=False)+"""
<h2>Artifact scope</h2><ul><li>Global pre-split artifact: diagnosis-independent instrument transfer.</li><li>Training-only artifacts: supervised preprocessing, model selection, calibration, thresholds and feature regions.</li><li>Frozen test artifacts: models and predictions verified by SHA256 before label merge.</li></ul>
</body></html>"""
    open(os.path.join(STUDY,"final_blinded_study_report.html"),"w",encoding="utf-8").write(html)
    print(f"Evaluated {ev.study_id.nunique()} pooled test patients")

if __name__=="__main__": main()
