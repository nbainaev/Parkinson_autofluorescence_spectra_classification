"""Stage 6-7 classical baselines (build_project_and_exp.md sections 12-13).
Small hyperparameter grids on purpose (section 12.6: "не делать слишком
огромный hyperparameter search" for a small medical dataset); Stage 6's goal
is a *correct* nested CV, not an exhaustive search.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from spectra_ml.models.base import ModelCandidate

# Preprocessing variants included in the default grid. Kept to a subset of
# spectra_ml.preprocessing's A-D for nested-CV runtime; extend once Stage 6/7's
# correctness is validated (section 40 priority: correct validation first).
DEFAULT_PREPROCESSING_VARIANTS = ("A", "B")

# Multiclass-only override (Hoehn-Yahr stage task, area-normalization redo):
# swaps the SNV branch ("B") for the area/vector-normalization branch ("D")
# in every multiclass candidate builder's preprocessing search grid, while
# leaving DEFAULT_PREPROCESSING_VARIANTS (still "A","B") untouched for the
# BINARY Stage 6/7 candidate builders above (build_logreg_candidates,
# build_linear_svm_candidates, build_rbf_svm_candidates, build_pls_logreg_
# candidates) and for the separate nonlinear multiclass extension pool
# (multiclass_stage_candidate_pool_v2_nonlinear / _extended, task-2 extension,
# out of scope for this redo) -- deliberately scoped to just the 5 multiclass
# builders feeding multiclass_stage_candidate_pool / _pool_v2, per explicit
# user request ("use area normalization instead of SNV ... just change the
# normalization").
# Forced to "D" only (no "A" plain-scaling alternative): nested CV kept picking
# "A" over "D" for 3/4 families in the prior grid-search redo (area normalization
# was offered but never selected), which does not satisfy the user's explicit
# request to actually USE area normalization instead of SNV. Forcing it removes
# the choice rather than leaving it to a nested-CV vote that empirically always
# went the other way.
MULTICLASS_PREPROCESSING_VARIANTS = ("D",)


def build_dummy_candidates() -> list[ModelCandidate]:
    return [ModelCandidate("dummy_prior", "dummy", "A", {"strategy": "prior"},
                            lambda: DummyClassifier(strategy="prior"))]


def build_logreg_candidates(preprocessing_variants=DEFAULT_PREPROCESSING_VARIANTS,
                             C_grid=(0.1, 1.0, 10.0), seed: int = 42) -> list[ModelCandidate]:
    # sklearn>=1.8 deprecated the `penalty` string in favor of `l1_ratio` (0.0 -> L2,
    # 1.0 -> L1); passing penalty='l1'/'l2' alongside the new default l1_ratio=0.0
    # triggers an "Inconsistent values" warning and silently fits L2 regardless of
    # what `penalty` said. Use l1_ratio directly to actually get the two penalties.
    out = []
    for variant in preprocessing_variants:
        for penalty_label, l1_ratio in (("l1", 1.0), ("l2", 0.0)):
            for C in C_grid:
                params = {"C": C, "l1_ratio": l1_ratio, "class_weight": "balanced"}
                out.append(ModelCandidate(
                    f"logreg_{penalty_label}_C{C}_{variant}", "logreg", variant,
                    {**params, "penalty_label": penalty_label},
                    lambda params=params: LogisticRegression(solver="liblinear", max_iter=2000,
                                                               random_state=seed, **params),
                ))
    return out


def build_linear_svm_candidates(preprocessing_variants=DEFAULT_PREPROCESSING_VARIANTS,
                                 C_grid=(0.1, 1.0, 10.0), seed: int = 42) -> list[ModelCandidate]:
    out = []
    for variant in preprocessing_variants:
        for C in C_grid:
            params = {"C": C, "class_weight": "balanced"}
            out.append(ModelCandidate(
                f"linear_svm_C{C}_{variant}", "linear_svm", variant, params,
                lambda params=params: LinearSVC(max_iter=5000, random_state=seed, **params),
                has_decision_function=True,
            ))
    return out


def build_rbf_svm_candidates(preprocessing_variants=DEFAULT_PREPROCESSING_VARIANTS,
                              C_grid=(1.0, 10.0), gamma_grid=("scale", 0.01), seed: int = 42) -> list[ModelCandidate]:
    out = []
    for variant in preprocessing_variants:
        for C in C_grid:
            for gamma in gamma_grid:
                params = {"C": C, "gamma": gamma, "class_weight": "balanced"}
                out.append(ModelCandidate(
                    f"rbf_svm_C{C}_g{gamma}_{variant}", "rbf_svm", variant, params,
                    # decision_function used for ranking/AUC; SVC's own probability=True calibration
                    # is a black box we deliberately avoid per section 12.4 -- calibrate separately (Stage 8).
                    lambda params=params: SVC(kernel="rbf", probability=False, random_state=seed, **params),
                    has_decision_function=True,
                ))
    return out


def build_gradient_boosting_candidates(preprocessing_variants=("A",), seed: int = 42) -> list[ModelCandidate]:
    # class_weight was missing here originally (every other family gets 'balanced'
    # explicitly) -- with park=140/control=43 (~3.3:1), an unweighted HGB is biased
    # toward the majority class; sklearn's HistGradientBoostingClassifier does
    # support class_weight (unlike some older tree APIs), so there's no excuse.
    out = []
    for variant in preprocessing_variants:
        for max_depth in (3, None):
            params = {"max_depth": max_depth, "class_weight": "balanced"}
            out.append(ModelCandidate(
                f"hgb_depth{max_depth}_{variant}", "hgb", variant, params,
                lambda params=params: HistGradientBoostingClassifier(random_state=seed, **params),
            ))
    return out


def build_random_forest_candidates(preprocessing_variants=("A",), seed: int = 42) -> list[ModelCandidate]:
    from sklearn.ensemble import RandomForestClassifier

    out = []
    for variant in preprocessing_variants:
        for n_estimators in (200, 500):
            for max_depth in (None, 8):
                params = {"n_estimators": n_estimators, "max_depth": max_depth, "class_weight": "balanced"}
                out.append(ModelCandidate(
                    f"rf_n{n_estimators}_depth{max_depth}_{variant}", "random_forest", variant, params,
                    lambda params=params: RandomForestClassifier(random_state=seed, **params),
                ))
    return out


def build_xgboost_candidates(preprocessing_variants=("A",), seed: int = 42) -> list[ModelCandidate]:
    """XGBoost has no 'balanced' class_weight string (unlike sklearn/LightGBM);
    scale_pos_weight must be a number computed from the actual class ratio. A
    fixed value baked in from the full dataset would be wrong inside CV (each
    fold's ratio differs slightly, and it's a leak of full-dataset info into a
    fold-local fit) -- so this subclass recomputes it from `y` at each .fit()
    call, exactly like class_weight='balanced' does internally elsewhere."""
    import xgboost as xgb

    class _BalancedXGBClassifier(xgb.XGBClassifier):
        def fit(self, X, y, sample_weight=None, **kwargs):
            import numpy as np
            n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
            self.scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)

    out = []
    for variant in preprocessing_variants:
        for n_estimators in (100, 300):
            for max_depth in (3, 5):
                params = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": 0.1}
                out.append(ModelCandidate(
                    f"xgb_n{n_estimators}_d{max_depth}_{variant}", "xgboost", variant, params,
                    lambda params=params: _BalancedXGBClassifier(random_state=seed, eval_metric="logloss",
                                                                    **params),
                ))
    return out


def build_lightgbm_candidates(preprocessing_variants=("A",), seed: int = 42) -> list[ModelCandidate]:
    import lightgbm as lgb

    out = []
    for variant in preprocessing_variants:
        for n_estimators in (100, 300):
            for max_depth in (3, -1):
                params = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": 0.1,
                          "class_weight": "balanced"}
                out.append(ModelCandidate(
                    f"lgbm_n{n_estimators}_d{max_depth}_{variant}", "lightgbm", variant, params,
                    lambda params=params: lgb.LGBMClassifier(random_state=seed, verbosity=-1, **params),
                ))
    return out


def build_catboost_candidates(preprocessing_variants=("A",), seed: int = 42) -> list[ModelCandidate]:
    from catboost import CatBoostClassifier

    out = []
    for variant in preprocessing_variants:
        for n_estimators in (100, 300):
            for depth in (3, 5):
                params = {"iterations": n_estimators, "depth": depth, "learning_rate": 0.1,
                          "auto_class_weights": "Balanced"}
                out.append(ModelCandidate(
                    f"catboost_n{n_estimators}_d{depth}_{variant}", "catboost", variant, params,
                    lambda params=params: CatBoostClassifier(random_state=seed, verbose=False, **params),
                ))
    return out


def build_extra_trees_candidates(preprocessing_variants=("A",), seed: int = 42) -> list[ModelCandidate]:
    out = []
    for variant in preprocessing_variants:
        for n_estimators in (200, 500):
            params = {"n_estimators": n_estimators, "class_weight": "balanced"}
            out.append(ModelCandidate(
                f"extratrees_n{n_estimators}_{variant}", "extratrees", variant, params,
                lambda params=params: ExtraTreesClassifier(random_state=seed, **params),
            ))
    return out


def build_pls_logreg_candidates(preprocessing_variants=DEFAULT_PREPROCESSING_VARIANTS,
                                 n_components_grid=(3, 8), C_grid=(1.0,), seed: int = 42) -> list[ModelCandidate]:
    """PLS -> LogisticRegression (section 13). PLS is fit with y inside the
    pipeline's own fit(), so it only ever sees the current training fold."""
    from sklearn.pipeline import Pipeline

    from spectra_ml.reduction import PLSFeatureExtractor

    out = []
    for variant in preprocessing_variants:
        for n_comp in n_components_grid:
            for C in C_grid:
                params = {"n_components": n_comp, "C": C, "class_weight": "balanced"}
                out.append(ModelCandidate(
                    f"pls{n_comp}_logreg_C{C}_{variant}", "pls_logreg", variant, params,
                    lambda params=params: Pipeline([
                        ("pls", PLSFeatureExtractor(n_components=params["n_components"])),
                        ("logreg", LogisticRegression(C=params["C"], class_weight=params["class_weight"],
                                                        solver="lbfgs", max_iter=2000, random_state=seed)),
                    ]),
                ))
    return out


def default_candidate_pool(seed: int = 42) -> list[ModelCandidate]:
    return (
        build_dummy_candidates()
        + build_logreg_candidates(seed=seed)
        + build_linear_svm_candidates(seed=seed)
        + build_pls_logreg_candidates(seed=seed)
    )


def extended_candidate_pool(seed: int = 42) -> list[ModelCandidate]:
    """Stage 7 additions: RBF SVM, HistGradientBoosting, ExtraTrees, RandomForest."""
    return (
        default_candidate_pool(seed=seed)
        + build_rbf_svm_candidates(seed=seed)
        + build_gradient_boosting_candidates(seed=seed)
        + build_extra_trees_candidates(seed=seed)
        + build_random_forest_candidates(seed=seed)
    )


def gbm_candidate_pool(seed: int = 42) -> list[ModelCandidate]:
    """Optional third-party GBM libraries (section 12.5: not required, tried here
    because sklearn's HistGradientBoostingClassifier underperformed). Requires
    xgboost/lightgbm/catboost to be installed -- imported lazily inside each
    build_* function so this module doesn't hard-depend on them."""
    return build_xgboost_candidates(seed=seed) + build_lightgbm_candidates(seed=seed) + build_catboost_candidates(seed=seed)


# ---------------------------------------------------------------------------
# Multiclass (4-class Hoehn-Yahr PD stage) candidates -- additive, small
# family set per task scope (runtime discipline: no neural nets, no tree
# ensembles, no extended hyperparameter grids here; see
# scripts/run_experiment_united_stage.py). None of these are used by any
# binary pipeline.
# ---------------------------------------------------------------------------

class GroupAwareCalibratedLinearSVC:
    """LinearSVC (native OvR multiclass) wrapped in CalibratedClassifierCV for
    predict_proba (LinearSVC has no native predict_proba). The calibration
    CV split is done via StratifiedGroupKFold on `patient_id` (passed at
    fit() time, same `requires_patient_groups=True` convention already used
    for torch models in spectra_ml.experiment) instead of
    CalibratedClassifierCV's own default (non-group-aware) CV -- a patient's
    repeated spectra are correlated, so letting CalibratedClassifierCV split
    them internally without respecting groups would leak the same patient
    across its own internal calibration train/val, which every other CV
    boundary in this project is careful to avoid.

    `reducer_kind` (None | "pca" | "pls"): if set, the base estimator becomes
    a Pipeline([("reduce", <PCA/PLSFeatureExtractor>), ("svc", LinearSVC(...))])
    instead of a bare LinearSVC -- the reducer is fit INSIDE the same Pipeline
    that CalibratedClassifierCV internally refits per calibration-CV split, so
    each split's own PCA/PLS is fit only on that split's own training rows
    (never on the rows held out for that split's calibration), avoiding the
    leakage that would occur if the reducer were instead fit once on the
    whole pre-calibration-split training set."""

    def __init__(self, C: float = 1.0, class_weight="balanced", cv_splits: int = 3, seed: int = 42,
                 reducer_kind: str | None = None, n_components: int = 8):
        self.C = C
        self.class_weight = class_weight
        self.cv_splits = cv_splits
        self.seed = seed
        self.reducer_kind = reducer_kind
        self.n_components = n_components

    def _make_base_estimator(self):
        svc = LinearSVC(C=self.C, class_weight=self.class_weight, max_iter=5000, random_state=self.seed)
        if self.reducer_kind is None:
            return svc
        from sklearn.pipeline import Pipeline

        from spectra_ml.reduction import PCAFeatureExtractor, PLSFeatureExtractor

        if self.reducer_kind == "pca":
            reducer = PCAFeatureExtractor(n_components=self.n_components)
        elif self.reducer_kind == "pls":
            reducer = PLSFeatureExtractor(n_components=self.n_components)
        else:
            raise ValueError(f"reducer_kind must be None/'pca'/'pls', got {self.reducer_kind!r}")
        return Pipeline([("reduce", reducer), ("svc", svc)])

    def fit(self, X, y, sample_weight=None, patient_id=None):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedGroupKFold

        if patient_id is None:
            raise ValueError("GroupAwareCalibratedLinearSVC.fit requires patient_id for group-aware calibration CV.")

        base = self._make_base_estimator()
        uniq_patients, first_idx = np.unique(patient_id, return_index=True)
        min_class = int(np.min([np.sum(y[first_idx] == c) for c in np.unique(y)]))
        n_splits = max(2, min(self.cv_splits, min_class))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        splits = list(sgkf.split(X, y, groups=patient_id))

        self.calibrated_ = CalibratedClassifierCV(base, method="sigmoid", cv=splits)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
            if self.reducer_kind is not None:
                # BUG (found in the united_multiclass_v2 audit, task 1 item 4): when
                # `base` is a Pipeline([("reduce", ...), ("svc", ...)]) (reducer_kind
                # set), CalibratedClassifierCV.fit's own sample_weight support check
                # (`"sample_weight" in signature(estimator.fit).parameters`) inspects
                # Pipeline.fit's literal signature, which is `fit(self, X, y=None,
                # **params)` -- "sample_weight" is never a literal parameter name for
                # a Pipeline, so CalibratedClassifierCV silently drops the bare
                # sample_weight= kwarg for the classifier's own fit (only using it for
                # the sigmoid calibration step) and just warns
                # ("... does not appear to accept sample_weight ... calibration is
                # likely to be incorrect"). Route it explicitly to the "svc" step by
                # name instead -- sklearn's fit_params/metadata-routing DOES forward a
                # `<step>__sample_weight` kwarg through Pipeline.fit to that step.
                fit_kwargs["svc__sample_weight"] = sample_weight
        try:
            if self.reducer_kind is not None and "svc__sample_weight" in fit_kwargs:
                # sklearn's own supports_sw check (see comment above) always warns here
                # regardless of the svc__sample_weight routing fix -- it inspects only
                # the top-level Pipeline.fit signature, which by construction never
                # literally names "sample_weight". The warning's claim ("sample weights
                # will only be used for the calibration itself") is exactly the bug this
                # fix addresses, so suppress this ONE specific, known-stale message
                # (not warnings in general) rather than let it misreport already-fixed
                # behavior.
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.filterwarnings(
                        "ignore", message=r".*does not appear to accept sample_weight.*", category=UserWarning,
                    )
                    self.calibrated_.fit(X, y, **fit_kwargs)
            else:
                self.calibrated_.fit(X, y, **fit_kwargs)
        except TypeError:
            self.calibrated_.fit(X, y)
        self.classes_ = self.calibrated_.classes_
        return self

    def predict_proba(self, X):
        return self.calibrated_.predict_proba(X)

    def predict(self, X):
        return self.calibrated_.predict(X)


def build_dummy_multiclass_candidates() -> list[ModelCandidate]:
    """Floor baseline for the multiclass task: DummyClassifier(strategy=
    'stratified') (draws predictions from the observed class distribution) --
    the binary 'dummy_prior' candidate is not reused here because the user's
    scope explicitly calls for a 'stratified' floor for this task."""
    return [ModelCandidate("dummy_stratified", "dummy", "A", {"strategy": "stratified"},
                            lambda: DummyClassifier(strategy="stratified", random_state=42))]


def build_multinomial_logreg_candidates(preprocessing_variants=MULTICLASS_PREPROCESSING_VARIANTS,
                                         C_grid=(0.1, 1.0, 10.0), seed: int = 42) -> list[ModelCandidate]:
    """Multinomial (softmax) logistic regression. sklearn>=1.7 removed the
    `multi_class` constructor argument entirely (verified via
    inspect.signature(LogisticRegression.__init__) against the installed
    sklearn==1.8.0): solver='lbfgs' now ALWAYS fits a genuine multinomial
    (softmax) model for a >2-class y, with no silent one-vs-rest fallback to
    worry about -- stated explicitly here rather than relying on it silently.
    liblinear (used by the binary logreg family above) does NOT support
    multiclass in this way, so this family uses lbfgs, not liblinear."""
    out = []
    for variant in preprocessing_variants:
        for C in C_grid:
            params = {"C": C, "class_weight": "balanced"}
            out.append(ModelCandidate(
                f"mn_logreg_C{C}_{variant}", "mn_logreg", variant, params,
                lambda params=params: LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed,
                                                           **params),
            ))
    return out


def build_linear_svm_multiclass_candidates(preprocessing_variants=MULTICLASS_PREPROCESSING_VARIANTS,
                                            C_grid=(0.1, 1.0, 10.0), seed: int = 42) -> list[ModelCandidate]:
    """LinearSVC, native OvR multiclass strategy, wrapped for probability
    estimates via GroupAwareCalibratedLinearSVC (see class docstring)."""
    out = []
    for variant in preprocessing_variants:
        for C in C_grid:
            params = {"C": C, "class_weight": "balanced"}
            out.append(ModelCandidate(
                f"linear_svm_mc_C{C}_{variant}", "linear_svm_mc", variant, params,
                lambda params=params: GroupAwareCalibratedLinearSVC(C=params["C"],
                                                                       class_weight=params["class_weight"],
                                                                       seed=seed),
                requires_patient_groups=True,
            ))
    return out


def build_pls_multinomial_logreg_candidates(preprocessing_variants=MULTICLASS_PREPROCESSING_VARIANTS,
                                             n_components_grid=(3, 8), C_grid=(1.0,),
                                             seed: int = 42) -> list[ModelCandidate]:
    """PLS -> multinomial LogisticRegression, multiclass sibling of
    build_pls_logreg_candidates. PLSFeatureExtractor is fit on the raw integer
    stage label (0/1/2/3) as a single continuous PLS regression target -- a
    defensible choice since Hoehn-Yahr stage is ORDINAL (severity-ordered), so
    a supervised projection that correlates with the numeric stage value is a
    reasonable dimensionality reduction, not an arbitrary encoding choice."""
    from sklearn.pipeline import Pipeline

    from spectra_ml.reduction import PLSFeatureExtractor

    out = []
    for variant in preprocessing_variants:
        for n_comp in n_components_grid:
            for C in C_grid:
                params = {"n_components": n_comp, "C": C, "class_weight": "balanced"}
                out.append(ModelCandidate(
                    f"pls{n_comp}_mnlogreg_C{C}_{variant}", "pls_mnlogreg", variant, params,
                    lambda params=params: Pipeline([
                        ("pls", PLSFeatureExtractor(n_components=params["n_components"])),
                        ("logreg", LogisticRegression(C=params["C"], class_weight=params["class_weight"],
                                                        solver="lbfgs", max_iter=2000, random_state=seed)),
                    ]),
                ))
    return out


def multiclass_stage_candidate_pool(seed: int = 42) -> list[ModelCandidate]:
    """Task scope: dummy (stratified floor) + mn_logreg + linear_svm_mc +
    pls_mnlogreg. Deliberately excludes neural nets and tree ensembles per
    runtime discipline (see scripts/run_experiment_united_stage.py)."""
    return (
        build_dummy_multiclass_candidates()
        + build_multinomial_logreg_candidates(seed=seed)
        + build_linear_svm_multiclass_candidates(seed=seed)
        + build_pls_multinomial_logreg_candidates(seed=seed)
    )


# ---------------------------------------------------------------------------
# v2 multiclass pool (united_multiclass_v2 task): route EVERY classical model
# head through a PLS or PCA dimensionality-reduction step first (except a
# dummy floor baseline, which has no features to reduce) -- explicit user
# request ("dimensionality reduction to avoid overfitting, except CNN"; no
# CNN is used anywhere in this project's classical pool). Concretely: replaces
# the raw-278-wavelength mn_logreg/linear_svm_mc families used by
# multiclass_stage_candidate_pool with PCA-reduced siblings (keeping the
# already-PLS-reduced pls_mnlogreg unchanged) -- also a runtime win, since a
# raw-278-dim lbfgs multinomial logreg was the slowest part of the prior round.
# ---------------------------------------------------------------------------

def build_pca_multinomial_logreg_candidates(preprocessing_variants=MULTICLASS_PREPROCESSING_VARIANTS,
                                             n_components_grid=(5, 10), C_grid=(1.0,),
                                             seed: int = 42) -> list[ModelCandidate]:
    """PCA -> multinomial LogisticRegression, PCA sibling of
    build_pls_multinomial_logreg_candidates. Unlike PLS, PCA is unsupervised
    (ignores y for the projection itself) -- included alongside PLS per the
    task's "PLS and/or PCA" instruction, and as a check that the PLS
    projection's mild label-awareness is not doing all the work."""
    from sklearn.pipeline import Pipeline

    from spectra_ml.reduction import PCAFeatureExtractor

    out = []
    for variant in preprocessing_variants:
        for n_comp in n_components_grid:
            for C in C_grid:
                params = {"n_components": n_comp, "C": C, "class_weight": "balanced"}
                out.append(ModelCandidate(
                    f"pca{n_comp}_mnlogreg_C{C}_{variant}", "pca_mnlogreg", variant, params,
                    lambda params=params: Pipeline([
                        ("pca", PCAFeatureExtractor(n_components=params["n_components"])),
                        ("logreg", LogisticRegression(C=params["C"], class_weight=params["class_weight"],
                                                        solver="lbfgs", max_iter=2000, random_state=seed)),
                    ]),
                ))
    return out


def build_reduced_linear_svm_multiclass_candidates(preprocessing_variants=MULTICLASS_PREPROCESSING_VARIANTS,
                                                     n_components_grid=(5, 10), C_grid=(1.0, 10.0),
                                                     reducer_kind: str = "pca",
                                                     seed: int = 42) -> list[ModelCandidate]:
    """PCA/PLS-reduced sibling of build_linear_svm_multiclass_candidates: the
    reduction step is fit INSIDE GroupAwareCalibratedLinearSVC's own
    CalibratedClassifierCV Pipeline (see that class's docstring) rather than
    on the raw 278-wavelength feature space, so PLS/PCA component count is
    itself effectively part of what nested CV selects via the usual inner-fold
    model-selection loop (spectra_ml.multiclass._select_best_candidate_multiclass)."""
    out = []
    for variant in preprocessing_variants:
        for n_comp in n_components_grid:
            for C in C_grid:
                params = {"n_components": n_comp, "C": C, "class_weight": "balanced", "reducer_kind": reducer_kind}
                out.append(ModelCandidate(
                    f"{reducer_kind}{n_comp}_linear_svm_mc_C{C}_{variant}", "linear_svm_mc_reduced", variant,
                    params,
                    lambda params=params: GroupAwareCalibratedLinearSVC(
                        C=params["C"], class_weight=params["class_weight"], seed=seed,
                        reducer_kind=params["reducer_kind"], n_components=params["n_components"],
                    ),
                    requires_patient_groups=True,
                ))
    return out


def multiclass_stage_candidate_pool_v2(seed: int = 42) -> list[ModelCandidate]:
    """v2 task scope: dummy (stratified floor) + PCA+multinomial-logreg +
    PLS+multinomial-logreg + PCA-reduced linear SVM (OvR, calibrated) --
    EVERY non-dummy family now goes through a PLS/PCA dimensionality-reduction
    step first (no raw-278-wavelength linear model, unlike
    multiclass_stage_candidate_pool). Still deliberately excludes neural nets
    and tree ensembles per runtime discipline (see
    scripts/run_experiment_united_stage_v2.py)."""
    return (
        build_dummy_multiclass_candidates()
        + build_pca_multinomial_logreg_candidates(seed=seed)
        + build_pls_multinomial_logreg_candidates(seed=seed)
        + build_reduced_linear_svm_multiclass_candidates(seed=seed, reducer_kind="pca")
    )


# ---------------------------------------------------------------------------
# united_multiclass_v2 Task 2 (extension, separate metrics_extended.csv run):
# nonlinear candidate families -- Random Forest, HistGradientBoosting,
# RBF-kernel SVM, k-NN. Added to the SAME nested-CV multiclass framework
# (spectra_ml.multiclass), same modeling cohort, same patient-grouped
# splitters/metrics as multiclass_stage_candidate_pool_v2. Kept in a SEPARATE
# pool function (multiclass_stage_candidate_pool_v2_nonlinear) rather than
# folded into multiclass_stage_candidate_pool_v2 itself, so the original v2
# run (outputs/united_multiclass_v2/metrics.csv) stays exactly reproducible;
# multiclass_stage_candidate_pool_v2_extended below combines both for the
# side-by-side comparison report.
#
# Deliberate, explicit exception to multiclass_stage_candidate_pool_v2's own
# "every non-dummy family goes through PLS/PCA" rule: tree ensembles (RF, HGB)
# have built-in regularization (subsampling / depth-leaf limits) and a
# different overfitting profile than linear models on 278 correlated
# wavelength features, so each is run BOTH on the raw 278-wavelength feature
# space (preprocessing variant "A" = StandardScaler only -- a no-op for tree
# split decisions, kept just so every candidate goes through the same
# build_preprocessing_pipeline scaffolding) AND on PCA-reduced features using
# the SAME component count already nested-CV-selected for the linear models
# this round (outputs/united_multiclass_v2/chosen_n_components.csv: PCA=10) --
# a FIXED count here, not its own grid, to keep runtime bounded per explicit
# task instruction. RBF-SVM and k-NN are run PCA-reduced ONLY (both are
# meaningfully affected by raw high-dimensionality/distance concentration) --
# no raw variant for either. PLS-reduced tree variants were considered but
# dropped for runtime discipline (doubling the tree-ensemble candidate count
# for a family that already gets a raw-feature variant is a lower priority
# than keeping the whole extended run fast) -- stated explicitly, not silently
# skipped; see scripts/run_experiment_united_stage_v2_extended.py.
# ---------------------------------------------------------------------------

_FIXED_PCA_N_COMPONENTS = 10  # matches outputs/united_multiclass_v2/chosen_n_components.csv (pca_mnlogreg)


def _make_reducer(reducer_kind: str, n_components: int):
    from spectra_ml.reduction import PCAFeatureExtractor, PLSFeatureExtractor

    if reducer_kind == "pca":
        return PCAFeatureExtractor(n_components=n_components)
    if reducer_kind == "pls":
        return PLSFeatureExtractor(n_components=n_components)
    raise ValueError(f"reducer_kind must be 'pca' or 'pls', got {reducer_kind!r}")


def build_random_forest_multiclass_raw_candidates(preprocessing_variants=("A",),
                                                    n_estimators_grid=(300,), max_depth_grid=(None, 8),
                                                    seed: int = 42) -> list[ModelCandidate]:
    """Random Forest on the raw 278-wavelength feature space (no PLS/PCA) --
    deliberate exception, see module note above."""
    from sklearn.ensemble import RandomForestClassifier

    out = []
    for variant in preprocessing_variants:
        for n_estimators in n_estimators_grid:
            for max_depth in max_depth_grid:
                params = {"n_estimators": n_estimators, "max_depth": max_depth,
                          "class_weight": "balanced_subsample"}
                out.append(ModelCandidate(
                    f"rf_raw_n{n_estimators}_depth{max_depth}_{variant}", "random_forest_raw", variant, params,
                    lambda params=params: RandomForestClassifier(random_state=seed, **params),
                ))
    return out


def build_random_forest_multiclass_reduced_candidates(preprocessing_variants=DEFAULT_PREPROCESSING_VARIANTS,
                                                        n_estimators_grid=(300,), max_depth_grid=(None, 8),
                                                        reducer_kind: str = "pca",
                                                        seed: int = 42) -> list[ModelCandidate]:
    """Random Forest on PCA(n=10)-reduced features (fixed component count, see
    module note above)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    n_comp = _FIXED_PCA_N_COMPONENTS
    out = []
    for variant in preprocessing_variants:
        for n_estimators in n_estimators_grid:
            for max_depth in max_depth_grid:
                params = {"n_estimators": n_estimators, "max_depth": max_depth,
                          "class_weight": "balanced_subsample", "n_components": n_comp, "reducer_kind": reducer_kind}

                def build(params=params):
                    return Pipeline([
                        ("reduce", _make_reducer(params["reducer_kind"], params["n_components"])),
                        ("clf", RandomForestClassifier(random_state=seed, n_estimators=params["n_estimators"],
                                                         max_depth=params["max_depth"],
                                                         class_weight=params["class_weight"])),
                    ])

                out.append(ModelCandidate(
                    f"rf_{reducer_kind}{n_comp}_n{n_estimators}_depth{max_depth}_{variant}",
                    f"random_forest_reduced_{reducer_kind}", variant, params, build,
                ))
    return out


def build_hgb_multiclass_raw_candidates(preprocessing_variants=("A",), max_depth_grid=(3,), max_iter: int = 50,
                                         seed: int = 42) -> list[ModelCandidate]:
    """HistGradientBoosting on the raw 278-wavelength feature space (no
    PLS/PCA) -- deliberate exception, see module note above. class_weight is
    passed explicitly (sklearn's HistGradientBoostingClassifier supports it
    natively, unlike XGBoost -- see build_gradient_boosting_candidates's own
    note above for the binary sibling of this reasoning).

    max_depth is fixed to a single small value and max_iter kept modest
    (measured empirically, not guessed): for a 4-class multinomial fit,
    HistGradientBoostingClassifier builds one tree per class per boosting
    iteration, so max_iter=150 means up to 600 trees -- combined with this
    cohort's small n (~100 rows/outer-training-fold), a max_depth_grid=(3, 6)
    x max_iter=150 grid measured 10-45s PER FIT (not per candidate) on this
    dataset shape, which made the united_multiclass_v2 task-2 extension run
    balloon past 30 minutes without finishing; max_iter=50 + a single
    max_depth=3 cuts single-fit time to ~2-4s while still exercising the
    family, keeping total nested-CV wall-clock bounded (runtime discipline)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    out = []
    for variant in preprocessing_variants:
        for max_depth in max_depth_grid:
            params = {"max_depth": max_depth, "max_iter": max_iter, "class_weight": "balanced"}
            out.append(ModelCandidate(
                f"hgb_raw_depth{max_depth}_{variant}", "hgb_raw", variant, params,
                lambda params=params: HistGradientBoostingClassifier(random_state=seed, **params),
            ))
    return out


def build_hgb_multiclass_reduced_candidates(preprocessing_variants=DEFAULT_PREPROCESSING_VARIANTS,
                                             max_depth_grid=(3,), max_iter: int = 50,
                                             reducer_kind: str = "pca", seed: int = 42) -> list[ModelCandidate]:
    """HistGradientBoosting on PCA(n=10)-reduced features (fixed component
    count, see module note above)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    n_comp = _FIXED_PCA_N_COMPONENTS
    out = []
    for variant in preprocessing_variants:
        for max_depth in max_depth_grid:
            params = {"max_depth": max_depth, "max_iter": max_iter, "class_weight": "balanced",
                      "n_components": n_comp, "reducer_kind": reducer_kind}

            def build(params=params):
                return Pipeline([
                    ("reduce", _make_reducer(params["reducer_kind"], params["n_components"])),
                    ("clf", HistGradientBoostingClassifier(random_state=seed, max_depth=params["max_depth"],
                                                             max_iter=params["max_iter"],
                                                             class_weight=params["class_weight"])),
                ])

            out.append(ModelCandidate(
                f"hgb_{reducer_kind}{n_comp}_depth{max_depth}_{variant}", f"hgb_reduced_{reducer_kind}", variant,
                params, build,
            ))
    return out


def build_rbf_svm_multiclass_candidates(preprocessing_variants=DEFAULT_PREPROCESSING_VARIANTS,
                                         C_grid=(1.0, 10.0), gamma_grid=("scale", 0.01),
                                         reducer_kind: str = "pca", seed: int = 42) -> list[ModelCandidate]:
    """RBF-kernel SVM, multiclass via sklearn's native OvR decision_function
    shape (decision_function_shape='ovr'). PCA(n=10)-reduced features ONLY
    (raw 278-dim distance concentration would make an RBF kernel's notion of
    "distance" close to meaningless -- same reasoning that already restricts
    k-NN below to reduced features only).

    No predict_proba: SVC(probability=True) is its own internal 5-fold Platt
    calibration black box, excluded by the same reasoning as
    build_rbf_svm_candidates's binary sibling ("calibrate separately, not
    here") -- and this round has calibration disabled entirely, so adding a
    CalibratedClassifierCV wrapper just for this one family would be
    inconsistent scope creep. Instead this candidate is
    has_decision_function=True: spectra_ml.multiclass._fit_predict_proba_
    multiclass sigmoid-squashes the raw per-class OvR decision_function values
    for ranking/AUC/Brier/log-loss ONLY (not a calibrated probability, columns
    need not sum to 1) -- exactly mirroring the pre-existing binary linear_svm
    convention (spectra_ml.experiment._fit_predict_scores), not a new pattern."""
    from sklearn.pipeline import Pipeline

    n_comp = _FIXED_PCA_N_COMPONENTS
    out = []
    for variant in preprocessing_variants:
        for C in C_grid:
            for gamma in gamma_grid:
                params = {"C": C, "gamma": gamma, "class_weight": "balanced", "n_components": n_comp,
                          "reducer_kind": reducer_kind}

                def build(params=params):
                    svc = SVC(kernel="rbf", C=params["C"], gamma=params["gamma"],
                              class_weight=params["class_weight"], decision_function_shape="ovr",
                              probability=False, random_state=seed)
                    return Pipeline([("reduce", _make_reducer(params["reducer_kind"], params["n_components"])),
                                      ("svc", svc)])

                out.append(ModelCandidate(
                    f"rbf_svm_{reducer_kind}{n_comp}_C{C}_g{gamma}_{variant}", "rbf_svm", variant, params, build,
                    has_decision_function=True,
                ))
    return out


def build_knn_multiclass_candidates(preprocessing_variants=DEFAULT_PREPROCESSING_VARIANTS,
                                     n_neighbors_grid=(5, 11, 21), reducer_kind: str = "pca",
                                     seed: int = 42) -> list[ModelCandidate]:
    """k-NN, PCA(n=10)-reduced features ONLY (distance concentration in raw
    278-dim space). k-NN has no class_weight mechanism and its own .fit() does
    not accept a training-time sample_weight either (non-parametric nearest-
    neighbor rule) -- spectra_ml.experiment._fit_with_sample_weight's existing
    try/except TypeError fallback to an unweighted .fit() already handles
    this gracefully (stated explicitly, not a silent gap introduced here).
    `weights='distance'` (nearer neighbors count more) is used instead of the
    uniform default as a partial, cheap mitigation for the class imbalance
    k-NN cannot otherwise account for."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline

    n_comp = _FIXED_PCA_N_COMPONENTS
    out = []
    for variant in preprocessing_variants:
        for k in n_neighbors_grid:
            params = {"n_neighbors": k, "n_components": n_comp, "reducer_kind": reducer_kind}

            def build(params=params):
                return Pipeline([("reduce", _make_reducer(params["reducer_kind"], params["n_components"])),
                                  ("knn", KNeighborsClassifier(n_neighbors=params["n_neighbors"],
                                                                  weights="distance"))])

            out.append(ModelCandidate(f"knn_{reducer_kind}{n_comp}_k{k}_{variant}", "knn", variant, params, build))
    return out


def multiclass_stage_candidate_pool_v2_nonlinear(seed: int = 42) -> list[ModelCandidate]:
    """united_multiclass_v2 task 2 extension pool: Random Forest + HistGradient
    Boosting (each BOTH raw-278-wavelength AND PCA(n=10)-reduced) + RBF-SVM
    (PCA(n=10)-reduced only) + k-NN (PCA(n=10)-reduced only). See module note
    above for the deliberate raw-feature exception for tree ensembles and the
    PLS-reduced-tree-variant runtime trade-off."""
    return (
        build_random_forest_multiclass_raw_candidates(seed=seed)
        + build_random_forest_multiclass_reduced_candidates(seed=seed, reducer_kind="pca")
        + build_hgb_multiclass_raw_candidates(seed=seed)
        + build_hgb_multiclass_reduced_candidates(seed=seed, reducer_kind="pca")
        + build_rbf_svm_multiclass_candidates(seed=seed, reducer_kind="pca")
        + build_knn_multiclass_candidates(seed=seed, reducer_kind="pca")
    )


def multiclass_stage_candidate_pool_v2_nonlinear_reduced(seed: int = 42) -> list[ModelCandidate]:
    """Blinded-study runtime-trimmed nonlinear pool (additive sibling of
    multiclass_stage_candidate_pool_v2_nonlinear, does not replace it):
    PCA-reduced-only RF/HistGB/RBF-SVM/kNN, no raw-278-dim RF/HistGB siblings
    (cut entirely -- limited scientific value here for roughly double the
    runtime), and every family forced onto preprocessing_variants=("D",) (the
    same forced area-normalization variant as the linear multiclass families
    in this blinded study, not this module's own DEFAULT_PREPROCESSING_
    VARIANTS=("A","B")). Further trims vs. the defaults, for runtime:
    RF max_depth_grid=(8,) only (drop unbounded-depth trees -- slow and prone
    to memorizing 103 patients); RBF-SVM gamma_grid=("scale",) only (drop the
    second gamma value, halving that family's grid)."""
    return (
        build_random_forest_multiclass_reduced_candidates(
            preprocessing_variants=("D",), max_depth_grid=(8,), reducer_kind="pca", seed=seed)
        + build_hgb_multiclass_reduced_candidates(preprocessing_variants=("D",), reducer_kind="pca", seed=seed)
        + build_rbf_svm_multiclass_candidates(
            preprocessing_variants=("D",), C_grid=(1.0, 10.0), gamma_grid=("scale",), reducer_kind="pca", seed=seed)
        + build_knn_multiclass_candidates(preprocessing_variants=("D",), reducer_kind="pca", seed=seed)
    )


def multiclass_stage_candidate_pool_v2_extended(seed: int = 42) -> list[ModelCandidate]:
    """Full extended pool for the side-by-side comparison report: the
    ORIGINAL v2 pool (dummy + pca_mnlogreg + pls_mnlogreg +
    linear_svm_mc_reduced), UNCHANGED, plus the new nonlinear families above.
    Used by scripts/run_experiment_united_stage_v2_extended.py; the original
    scripts/run_experiment_united_stage_v2.py + multiclass_stage_candidate_
    pool_v2 are untouched and still reproduce outputs/united_multiclass_v2/
    metrics.csv exactly."""
    return multiclass_stage_candidate_pool_v2(seed=seed) + multiclass_stage_candidate_pool_v2_nonlinear(seed=seed)
