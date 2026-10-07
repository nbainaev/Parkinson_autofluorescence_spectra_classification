"""Stage 4: patient-level statistical comparison (build_project_and_exp.md
section 8). Deliberately separate from the ML pipeline.

Key correctness point: every test here operates on ONE REPRESENTATIVE SPECTRUM
PER PATIENT (mean or, preferably, median across that patient's QC-passing
spectra), never on individual spectra treated as independent observations --
a patient's 1-16 repeat spectra are correlated, so a spectrum-level t-test
would pretend to have far more independent samples than it does (pseudo-
replication) and understate p-values.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

_EPS = 1e-12


@dataclass
class PatientRepr:
    patient_id: np.ndarray     # (n_patients,)
    target: np.ndarray         # (n_patients,) 0/1
    spectra: np.ndarray        # (n_patients, n_wave), representative (mean/median) spectrum
    waves: np.ndarray          # (n_wave,)
    method: str
    n_spectra_used: np.ndarray  # (n_patients,) how many QC-passing spectra contributed


def compute_patient_representative_spectra(
    spectra: np.ndarray, patient_id: np.ndarray, target: np.ndarray, waves: np.ndarray,
    qc_status: np.ndarray | None = None, exclude_statuses=("HARD_FAIL",), method: str = "median",
) -> PatientRepr:
    if qc_status is not None:
        keep = ~np.isin(qc_status, exclude_statuses)
    else:
        keep = ~np.isnan(spectra).all(axis=1)

    df_idx = pd.DataFrame({"patient_id": patient_id[keep], "target": target[keep], "_row": np.where(keep)[0]})
    agg_fn = np.nanmedian if method == "median" else np.nanmean

    uniq_patients = df_idx["patient_id"].unique()
    n_patients = len(uniq_patients)
    out_spectra = np.full((n_patients, spectra.shape[1]), np.nan)
    out_target = np.full(n_patients, np.nan)
    n_used = np.zeros(n_patients, dtype=int)

    for i, pid in enumerate(uniq_patients):
        rows = df_idx.loc[df_idx["patient_id"] == pid, "_row"].to_numpy()
        targets_here = target[rows]
        uniq_t = pd.unique(targets_here[~np.isnan(targets_here)])
        if len(uniq_t) > 1:
            raise ValueError(f"patient {pid!r} has conflicting target values among its QC-passing spectra: {uniq_t}")
        out_target[i] = uniq_t[0] if len(uniq_t) else np.nan
        out_spectra[i] = agg_fn(spectra[rows], axis=0)
        n_used[i] = len(rows)

    return PatientRepr(patient_id=uniq_patients, target=out_target, spectra=out_spectra,
                        waves=waves, method=method, n_spectra_used=n_used)


def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """x, y: 1D arrays (e.g. Parkinson, Control) at one wavelength. Positive g -> x > y."""
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2 + _EPS))
    d = (np.mean(x) - np.mean(y)) / (pooled_sd + _EPS)
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d * j)


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Mann-Whitney U (x vs y) + rank-biserial correlation. Positive r -> x tends higher than y."""
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    r = 1 - 2 * u / (len(x) * len(y))
    return float(u), float(p), float(r)


def mannwhitney_z_score(x: np.ndarray, y: np.ndarray, u: float) -> float:
    """Tie-corrected normal-approximation z-score for the Mann-Whitney U
    statistic (x vs y), NO continuity correction (matches
    ``scipy.stats.mannwhitneyu(x, y, method="asymptotic", use_continuity=False)``,
    see scipy.stats._mannwhitneyu._get_mwu_z):

        z = (U - n1*n2/2) / sqrt(n1*n2*(n1+n2+1)/12
                                  - (n1*n2 / (12*(n1+n2)*(n1+n2-1))) * sum(t_k^3 - t_k))

    where t_k are tie-group sizes (over the POOLED x+y values). Ties only occur
    at floating precision here (real intensity data), but the correction is
    computed properly rather than skipped -- with real data, t_k == 1 for
    every group and the term vanishes, reducing to the textbook untied
    variance n1*n2*(n1+n2+1)/12.
    """
    n1, n2 = len(x), len(y)
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 2:
        return float("nan")
    combined = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_counts = tie_counts.astype(float)
    tie_term = float(np.sum(tie_counts ** 3 - tie_counts))
    mu = n1 * n2 / 2.0
    variance = n1 * n2 * (n + 1) / 12.0 - (n1 * n2 * tie_term) / (12.0 * n * (n - 1))
    if variance <= 0:
        return float("nan")
    return float((u - mu) / np.sqrt(variance))


def _bootstrap_ci(park: np.ndarray, control: np.ndarray, n_boot: int, ci: float,
                   estimator, seed: int) -> dict:
    """Patient-level bootstrap: independently resample WITHIN each group (stratified
    by class), never mixing/resampling individual spectra of one patient. Returns
    CI for each group's own estimate and for their difference (same resampling
    draws, so the difference CI accounts for the joint variability correctly)."""
    rng = np.random.default_rng(seed)
    n1, n2 = len(park), len(control)
    park_boot = np.empty(n_boot)
    ctrl_boot = np.empty(n_boot)
    for b in range(n_boot):
        park_boot[b] = estimator(park[rng.integers(0, n1, n1)])
        ctrl_boot[b] = estimator(control[rng.integers(0, n2, n2)])
    diff_boot = park_boot - ctrl_boot
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "park_ci_lo": float(np.percentile(park_boot, lo_q)), "park_ci_hi": float(np.percentile(park_boot, hi_q)),
        "control_ci_lo": float(np.percentile(ctrl_boot, lo_q)), "control_ci_hi": float(np.percentile(ctrl_boot, hi_q)),
        "diff_ci_lo": float(np.percentile(diff_boot, lo_q)), "diff_ci_hi": float(np.percentile(diff_boot, hi_q)),
    }


def compare_groups_per_wavelength(repr_: PatientRepr, n_boot: int = 2000, ci: float = 95,
                                   seed: int = 42) -> pd.DataFrame:
    park_mask = repr_.target == 1
    ctrl_mask = repr_.target == 0
    n_park, n_ctrl = int(park_mask.sum()), int(ctrl_mask.sum())
    if n_park < 2 or n_ctrl < 2:
        raise ValueError(f"need >=2 patients per class, got park={n_park} control={n_ctrl}")

    park = repr_.spectra[park_mask]
    ctrl = repr_.spectra[ctrl_mask]
    estimator = np.median if repr_.method == "median" else np.mean

    rows = []
    for j, w in enumerate(repr_.waves):
        p_col, c_col = park[:, j], ctrl[:, j]
        p_valid, c_valid = p_col[~np.isnan(p_col)], c_col[~np.isnan(c_col)]
        if len(p_valid) < 2 or len(c_valid) < 2:
            rows.append({"wavelength": w, "n_park": len(p_valid), "n_control": len(c_valid)})
            continue

        park_center, ctrl_center = estimator(p_valid), estimator(c_valid)
        diff = park_center - ctrl_center
        rel_diff = diff / (abs(ctrl_center) + _EPS)

        t, welch_p = stats.ttest_ind(p_valid, c_valid, equal_var=False)
        g = _hedges_g(p_valid, c_valid)
        u, mwu_p, rb = _rank_biserial(p_valid, c_valid)
        mwu_z = mannwhitney_z_score(p_valid, c_valid, u)
        boot = _bootstrap_ci(p_valid, c_valid, n_boot, ci, estimator, seed + j)

        rows.append({
            "wavelength": w,
            "n_park": len(p_valid), "n_control": len(c_valid),
            "control_mean": float(np.mean(c_valid)), "control_median": float(np.median(c_valid)),
            "park_mean": float(np.mean(p_valid)), "park_median": float(np.median(p_valid)),
            "difference": float(diff), "relative_difference": float(rel_diff),
            "welch_t": float(t), "welch_p": float(welch_p), "hedges_g": g,
            "mwu_u": u, "mwu_p": mwu_p, "mwu_z": mwu_z, "rank_biserial": rb,
            f"park_bootstrap_ci{int(ci)}_lo": boot["park_ci_lo"], f"park_bootstrap_ci{int(ci)}_hi": boot["park_ci_hi"],
            f"control_bootstrap_ci{int(ci)}_lo": boot["control_ci_lo"], f"control_bootstrap_ci{int(ci)}_hi": boot["control_ci_hi"],
            f"diff_bootstrap_ci{int(ci)}_lo": boot["diff_ci_lo"], f"diff_bootstrap_ci{int(ci)}_hi": boot["diff_ci_hi"],
        })

    df = pd.DataFrame(rows)
    for col, out_col in [("welch_p", "welch_p_fdr"), ("mwu_p", "mwu_p_fdr")]:
        valid = df[col].notna()
        fdr = np.full(len(df), np.nan)
        if valid.sum():
            fdr[valid.to_numpy()] = multipletests(df.loc[valid, col], method="fdr_bh")[1]
        df[out_col] = fdr
    return df


def kruskal_wallis_per_wavelength(repr_: PatientRepr, class_labels=(0, 1, 2, 3)) -> pd.DataFrame:
    """Multiclass generalization of the binary Welch/Mann-Whitney comparison
    (compare_groups_per_wavelength): one omnibus Kruskal-Wallis H-test per
    wavelength across ALL `class_labels` groups simultaneously, with FDR-BH
    correction across wavelengths (same convention as compare_groups_per_
    wavelength's welch_p/mwu_p correction). `repr_.target` must already be
    restricted to (a subset of) `class_labels` by the caller -- patients with
    any other target value are ignored here, not treated as a group."""
    class_labels = list(class_labels)
    masks = {c: (repr_.target == c) for c in class_labels}
    n_per_class = {c: int(m.sum()) for c, m in masks.items()}
    for c, n in n_per_class.items():
        if n < 2:
            raise ValueError(f"need >=2 patients per class, got class={c} n={n}")

    rows = []
    for j, w in enumerate(repr_.waves):
        groups = []
        n_valid = {}
        for c in class_labels:
            col = repr_.spectra[masks[c], j]
            col = col[~np.isnan(col)]
            groups.append(col)
            n_valid[c] = len(col)
        if any(len(g) < 2 for g in groups):
            row = {"wavelength": w, **{f"n_class{c}": n_valid[c] for c in class_labels}}
            rows.append(row)
            continue

        h, p = stats.kruskal(*groups)
        row = {
            "wavelength": w, "kruskal_h": float(h), "kruskal_p": float(p),
            **{f"n_class{c}": n_valid[c] for c in class_labels},
            **{f"median_class{c}": float(np.median(g)) for c, g in zip(class_labels, groups)},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    fdr = np.full(len(df), np.nan)
    valid = df["kruskal_p"].notna() if "kruskal_p" in df.columns else pd.Series(dtype=bool)
    if valid.sum():
        fdr[valid.to_numpy()] = multipletests(df.loc[valid, "kruskal_p"], method="fdr_bh")[1]
    df["kruskal_p_fdr"] = fdr
    return df


def _pairwise_mannwhitney_raw_frames(repr_: PatientRepr, class_labels) -> list[pd.DataFrame]:
    """Shared row-computation for pairwise_mannwhitney_multiclass and
    pairwise_mannwhitney_multiclass_joint_fdr: one (uncorrected) DataFrame per
    class pair, columns class_a/class_b/wavelength/n_a/n_b/median_a/median_b/
    difference/mwu_u/mwu_p/mwu_z/hedges_g -- no FDR column yet (each caller
    applies its own correction convention on top of this)."""
    class_labels = list(class_labels)
    masks = {c: (repr_.target == c) for c in class_labels}
    for c, m in masks.items():
        if int(m.sum()) < 2:
            raise ValueError(f"need >=2 patients per class, got class={c} n={int(m.sum())}")

    pairs = list(itertools.combinations(class_labels, 2))
    frames = []
    for a, b in pairs:
        rows = []
        col_a = repr_.spectra[masks[a]]
        col_b = repr_.spectra[masks[b]]
        for j, w in enumerate(repr_.waves):
            x = col_a[:, j]
            y = col_b[:, j]
            x, y = x[~np.isnan(x)], y[~np.isnan(y)]
            if len(x) < 2 or len(y) < 2:
                rows.append({"class_a": a, "class_b": b, "wavelength": w, "n_a": len(x), "n_b": len(y)})
                continue
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            z = mannwhitney_z_score(x, y, u)
            g = _hedges_g(x, y)
            rows.append({
                "class_a": a, "class_b": b, "wavelength": w, "n_a": len(x), "n_b": len(y),
                "median_a": float(np.median(x)), "median_b": float(np.median(y)),
                "difference": float(np.median(x) - np.median(y)),
                "mwu_u": float(u), "mwu_p": float(p), "mwu_z": z, "hedges_g": g,
            })
        frames.append(pd.DataFrame(rows))
    return frames


def pairwise_mannwhitney_multiclass(repr_: PatientRepr, class_labels=(0, 1, 2, 3)) -> pd.DataFrame:
    """Post-hoc pairwise Mann-Whitney (tie-corrected z-score, U, p-value, Hedges'
    g) per wavelength for every pair of `class_labels` (6 pairs for 4 classes).
    Reuses mannwhitney_z_score + scipy.stats.mannwhitneyu + _hedges_g directly
    (no reimplementation of the binary machinery).

    FDR-BH correction is applied WITHIN each class pair, across wavelengths --
    the same convention compare_groups_per_wavelength already uses for its
    welch_p/mwu_p columns (within-test-across-wavelength), rather than pooling
    all pairs x wavelengths into one joint correction. Rationale: each pair is
    logically a separate hypothesis family (e.g. "does stage1 differ from
    control"), and keeping corrections pair-local matches how the binary
    single-pair case already behaves (it only ever had one pair) -- pooling
    across pairs would make the correction depend on how many OTHER pairs
    happen to be tested, which is a less standard and less interpretable
    choice here.

    See `pairwise_mannwhitney_multiclass_joint_fdr` for the alternative (all
    pairs x wavelengths corrected jointly, in one family) convention.
    """
    frames = _pairwise_mannwhitney_raw_frames(repr_, class_labels)
    all_frames = []
    for pair_df in frames:
        fdr = np.full(len(pair_df), np.nan)
        valid = pair_df["mwu_p"].notna() if "mwu_p" in pair_df.columns else pd.Series(dtype=bool)
        if valid.sum():
            fdr[valid.to_numpy()] = multipletests(pair_df.loc[valid, "mwu_p"], method="fdr_bh")[1]
        pair_df = pair_df.copy()
        pair_df["mwu_p_fdr"] = fdr
        all_frames.append(pair_df)

    return pd.concat(all_frames, ignore_index=True)


def pairwise_mannwhitney_multiclass_joint_fdr(repr_: PatientRepr, class_labels=(0, 1, 2, 3)) -> pd.DataFrame:
    """Same per-pair/per-wavelength Mann-Whitney statistics as
    `pairwise_mannwhitney_multiclass`, but the multiple-comparison correction
    is applied ONCE, JOINTLY, across the ENTIRE family of tests (all class
    pairs x all wavelengths pooled into one array -- e.g. 6 pairs x 278
    wavelengths = 1668 p-values for the 4-class Hoehn-Yahr stage task), via a
    single `statsmodels.stats.multitest.multipletests(..., method="fdr_bh")`
    call, rather than the per-pair-separately convention used by
    `pairwise_mannwhitney_multiclass` (which corrects each pair's 278
    wavelengths as its own, independent hypothesis family).

    This is intentionally MORE conservative than 6 independent per-pair FDR
    corrections would be: BH's rejection threshold at rank k is
    k/m * alpha, so pooling all m=1668 tests into one array raises the
    effective m used for every wavelength, shrinking the adjusted-p-value
    budget compared to correcting each pair's own (much smaller) m=278
    separately. Use this function when the "family" of hypotheses that must
    jointly control the false discovery rate is defined as ALL pairwise
    comparisons together (the requested convention here), not each pair in
    isolation.

    Returns the same row layout as `pairwise_mannwhitney_multiclass` (one row
    per class-pair x wavelength), with the corrected p-value in a column named
    `mwu_p_fdr_joint` (deliberately NOT `mwu_p_fdr`, so a caller that
    accidentally mixes the two conventions gets a clear KeyError instead of a
    silently wrong column).
    """
    frames = _pairwise_mannwhitney_raw_frames(repr_, class_labels)
    combined = pd.concat(frames, ignore_index=True)

    fdr = np.full(len(combined), np.nan)
    valid = combined["mwu_p"].notna() if "mwu_p" in combined.columns else pd.Series(dtype=bool)
    if valid.sum():
        fdr[valid.to_numpy()] = multipletests(combined.loc[valid, "mwu_p"], method="fdr_bh")[1]
    combined["mwu_p_fdr_joint"] = fdr
    return combined


def mixed_effects_per_wavelength(spectra: np.ndarray, patient_id: np.ndarray, target: np.ndarray,
                                  waves: np.ndarray, qc_status: np.ndarray | None = None,
                                  exclude_statuses=("HARD_FAIL",), wavelength_indices=None) -> pd.DataFrame:
    """intensity ~ target + (1 | patient_id) per wavelength, fit on individual
    (QC-passing) spectra -- uses the full repeated-measures structure instead of
    collapsing to one value per patient. Optional / slow (one MixedLM fit per
    wavelength): pass `wavelength_indices` to restrict to a subset for a quick
    pass; None fits every wavelength."""
    import statsmodels.formula.api as smf

    keep = ~np.isin(qc_status, exclude_statuses) if qc_status is not None else ~np.isnan(spectra).all(axis=1)
    pid, tgt = patient_id[keep], target[keep]
    spectra = spectra[keep]
    valid_t = ~np.isnan(tgt)
    pid, tgt, spectra = pid[valid_t], tgt[valid_t], spectra[valid_t]

    indices = range(len(waves)) if wavelength_indices is None else wavelength_indices
    rows = []
    for j in indices:
        y = spectra[:, j]
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            continue
        df = pd.DataFrame({"intensity": y[valid], "target": tgt[valid], "patient_id": pid[valid]})
        try:
            model = smf.mixedlm("intensity ~ target", df, groups=df["patient_id"])
            res = model.fit(reml=True, method="lbfgs")
            coef, se, p = res.params["target"], res.bse["target"], res.pvalues["target"]
        except Exception as e:  # noqa: BLE001 -- MixedLM can fail to converge on some wavelengths
            rows.append({"wavelength": waves[j], "coef": np.nan, "se": np.nan, "p_value": np.nan,
                         "converged": False, "error": str(e)})
            continue
        rows.append({"wavelength": waves[j], "coef": float(coef), "se": float(se), "p_value": float(p),
                     "converged": bool(res.converged)})

    out = pd.DataFrame(rows)
    if len(out) and out["p_value"].notna().any():
        valid = out["p_value"].notna()
        fdr = np.full(len(out), np.nan)
        fdr[valid.to_numpy()] = multipletests(out.loc[valid, "p_value"], method="fdr_bh")[1]
        out["p_value_fdr"] = fdr
    return out
