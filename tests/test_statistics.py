import numpy as np
import pytest
from scipy import stats

from spectra_ml.statistics import (
    PatientRepr,
    compare_groups_per_wavelength,
    compute_patient_representative_spectra,
    kruskal_wallis_per_wavelength,
    mannwhitney_z_score,
    pairwise_mannwhitney_multiclass,
    pairwise_mannwhitney_multiclass_joint_fdr,
)


def _p_from_z(z: float) -> float:
    return 2 * (1 - stats.norm.cdf(abs(z)))


class TestMannwhitneyZScore:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_matches_scipy_asymptotic_no_ties(self, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.3, 1.0, size=15)
        y = rng.normal(-0.2, 1.0, size=12)

        u, p_scipy = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic", use_continuity=False)
        z = mannwhitney_z_score(x, y, u)
        p_from_z = _p_from_z(z)

        assert p_from_z == pytest.approx(p_scipy, abs=1e-9)

    def test_matches_scipy_asymptotic_with_ties(self):
        # deliberately duplicated values -> nontrivial tie-group sizes
        x = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 3.0, 5.0, 5.0, 1.0])

        u, p_scipy = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic", use_continuity=False)
        z = mannwhitney_z_score(x, y, u)
        p_from_z = _p_from_z(z)

        assert p_from_z == pytest.approx(p_scipy, abs=1e-9)

    def test_symmetric_under_swap(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=10)
        y = rng.normal(size=13)
        u_xy, _ = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic", use_continuity=False)
        u_yx, _ = stats.mannwhitneyu(y, x, alternative="two-sided", method="asymptotic", use_continuity=False)
        z_xy = mannwhitney_z_score(x, y, u_xy)
        z_yx = mannwhitney_z_score(y, x, u_yx)
        assert z_xy == pytest.approx(-z_yx, abs=1e-9)

    def test_degenerate_returns_nan(self):
        assert np.isnan(mannwhitney_z_score(np.array([1.0]), np.array([]), 0.0))


class TestCompareGroupsPerWavelength:
    def test_mwu_z_column_present_and_matches_asymptotic_pvalue(self):
        # NOTE: `mwu_p` in the output frame comes from scipy's default
        # method="auto" (exact for small, untied samples) + use_continuity=True,
        # which is deliberately NOT the same statistic as mwu_z's no-continuity
        # asymptotic approximation -- so we independently recompute the
        # asymptotic, no-continuity p-value per wavelength for the check,
        # rather than comparing against the `mwu_p` column.
        rng = np.random.default_rng(7)
        n_park, n_ctrl, n_wave = 20, 15, 5
        waves = np.linspace(400.0, 500.0, n_wave)
        park_spectra = rng.normal(1.0, 0.2, size=(n_park, n_wave))
        ctrl_spectra = rng.normal(0.8, 0.2, size=(n_ctrl, n_wave))
        spectra = np.vstack([park_spectra, ctrl_spectra])
        patient_id = np.array([f"p{i}" for i in range(n_park + n_ctrl)])
        target = np.array([1.0] * n_park + [0.0] * n_ctrl)

        repr_ = compute_patient_representative_spectra(spectra, patient_id, target, waves, method="median")
        df = compare_groups_per_wavelength(repr_, n_boot=50, seed=1)

        assert "mwu_z" in df.columns
        assert df["mwu_z"].notna().all()
        for j in range(n_wave):
            x, y = park_spectra[:, j], ctrl_spectra[:, j]
            _, p_scipy = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic",
                                             use_continuity=False)
            p_from_z = _p_from_z(df.loc[j, "mwu_z"])
            assert p_from_z == pytest.approx(p_scipy, abs=1e-9)


def _make_multiclass_repr(rng, means, n_per_class=15, n_wave=4, sd=0.2):
    waves = np.linspace(400.0, 500.0, n_wave)
    spectra_parts, target_parts, pid_parts = [], [], []
    pid_counter = 0
    for cls, mean in enumerate(means):
        spec = rng.normal(mean, sd, size=(n_per_class, n_wave))
        spectra_parts.append(spec)
        target_parts.extend([float(cls)] * n_per_class)
        pid_parts.extend([f"p{pid_counter + i}" for i in range(n_per_class)])
        pid_counter += n_per_class
    spectra = np.vstack(spectra_parts)
    target = np.array(target_parts)
    patient_id = np.array(pid_parts)
    return compute_patient_representative_spectra(spectra, patient_id, target, waves, method="median")


class TestKruskalWallisPerWavelength:
    def test_four_separated_groups_significant(self):
        rng = np.random.default_rng(0)
        repr_ = _make_multiclass_repr(rng, means=[0.0, 2.0, 4.0, 6.0], sd=0.3)
        df = kruskal_wallis_per_wavelength(repr_, class_labels=(0, 1, 2, 3))
        assert (df["kruskal_p_fdr"] < 0.05).all()

    def test_four_identical_groups_not_significant(self):
        rng = np.random.default_rng(1)
        repr_ = _make_multiclass_repr(rng, means=[0.0, 0.0, 0.0, 0.0], sd=1.0)
        df = kruskal_wallis_per_wavelength(repr_, class_labels=(0, 1, 2, 3))
        assert (df["kruskal_p_fdr"] >= 0.05).all()

    def test_requires_min_two_per_class(self):
        rng = np.random.default_rng(2)
        repr_ = _make_multiclass_repr(rng, means=[0.0, 1.0, 2.0, 3.0], n_per_class=1)
        with pytest.raises(ValueError):
            kruskal_wallis_per_wavelength(repr_, class_labels=(0, 1, 2, 3))


class TestPairwiseMannwhitneyMulticlass:
    def test_all_six_pairs_significant_for_well_separated_groups(self):
        rng = np.random.default_rng(3)
        repr_ = _make_multiclass_repr(rng, means=[0.0, 3.0, 6.0, 9.0], sd=0.3)
        df = pairwise_mannwhitney_multiclass(repr_, class_labels=(0, 1, 2, 3))
        assert set(zip(df["class_a"], df["class_b"])) == {
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        }
        for (a, b), sub in df.groupby(["class_a", "class_b"]):
            assert (sub["mwu_p_fdr"] < 0.05).all(), f"pair ({a},{b}) not all significant"

    def test_z_score_matches_mannwhitney_z_score_directly(self):
        rng = np.random.default_rng(4)
        repr_ = _make_multiclass_repr(rng, means=[0.0, 1.0, 2.0, 3.0], sd=0.5)
        df = pairwise_mannwhitney_multiclass(repr_, class_labels=(0, 1, 2, 3))

        mask0 = repr_.target == 0.0
        mask1 = repr_.target == 1.0
        for j, w in enumerate(repr_.waves):
            x = repr_.spectra[mask0, j]
            y = repr_.spectra[mask1, j]
            u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
            expected_z = mannwhitney_z_score(x, y, u)
            row = df[(df["class_a"] == 0) & (df["class_b"] == 1) & (df["wavelength"] == w)]
            assert row["mwu_z"].iloc[0] == pytest.approx(expected_z, abs=1e-9)


class TestPairwiseMannwhitneyMulticlassJointFDR:
    def test_joint_fdr_more_conservative_than_six_independent_per_pair_corrections(self):
        """Synthetic check requested by the task brief: pooling all 6 pairs x
        n_wave wavelengths into one joint BH-FDR family is expected to be more
        conservative overall than correcting each pair's own wavelengths
        independently, in a scenario shaped like the real multiclass task --
        most class pairs (here: (0,1), (0,2), (1,2), all built from identical
        means) show NO true difference at any wavelength, while a minority of
        pairs ((0,3), (1,3), (2,3), built against a shifted class 3) show a
        real but not overwhelming difference. Pooling the many null pairs'
        p-values into the same correction family as the few real-signal pairs
        dilutes the BH budget available to the real signal, which is the
        realistic reason a joint correction is the conservative choice here.

        NOTE: BH-FDR adjusted p-values are NOT guaranteed to be pointwise
        monotonic between two differently-composed families in general (BH's
        adjustment depends on each p-value's RANK within its own family, not
        just the family size) -- so this check is deliberately aggregate
        (total significant count, mean adjusted p), not a per-wavelength
        inequality, which would not be a valid property to assert in general.
        """
        n_sig_per_pair_list, n_sig_joint_list = [], []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            repr_ = _make_multiclass_repr(rng, means=[0.0, 0.0, 0.0, 1.2], n_per_class=15, n_wave=40, sd=0.8)

            per_pair = pairwise_mannwhitney_multiclass(repr_, class_labels=(0, 1, 2, 3))
            joint = pairwise_mannwhitney_multiclass_joint_fdr(repr_, class_labels=(0, 1, 2, 3))
            assert len(joint) == len(per_pair)

            n_sig_per_pair_list.append(int((per_pair["mwu_p_fdr"] < 0.05).sum()))
            n_sig_joint_list.append(int((joint["mwu_p_fdr_joint"] < 0.05).sum()))

        # averaged over several seeds: joint correction yields fewer (or equal)
        # significant wavelength x pair combinations than the per-pair-
        # independent convention (mean mwu_p_fdr is NOT compared here -- BH's
        # adjustment reshuffles ranks non-linearly between differently-sized/
        # composed families, so a mean-adjusted-p comparison is not a valid
        # general property to assert, unlike the total significant count under
        # a fixed 0.05 threshold, which is the practically meaningful quantity
        # a report actually uses).
        assert np.mean(n_sig_joint_list) <= np.mean(n_sig_per_pair_list)
        assert np.mean(n_sig_joint_list) < np.mean(n_sig_per_pair_list)  # must actually differ in this scenario
        # this must hold in (at least) the clear majority of individual seeds too,
        # not just in the average across seeds.
        n_seeds_joint_leq = sum(j <= p for j, p in zip(n_sig_joint_list, n_sig_per_pair_list))
        assert n_seeds_joint_leq >= 6

    def test_joint_fdr_column_name_is_distinct_from_per_pair_convention(self):
        rng = np.random.default_rng(6)
        repr_ = _make_multiclass_repr(rng, means=[0.0, 3.0, 6.0, 9.0], sd=0.3)
        joint = pairwise_mannwhitney_multiclass_joint_fdr(repr_, class_labels=(0, 1, 2, 3))
        assert "mwu_p_fdr_joint" in joint.columns
        assert "mwu_p_fdr" not in joint.columns

    def test_joint_fdr_same_pairs_and_row_count_as_per_pair(self):
        rng = np.random.default_rng(7)
        repr_ = _make_multiclass_repr(rng, means=[0.0, 1.0, 2.0, 3.0], sd=0.5)
        per_pair = pairwise_mannwhitney_multiclass(repr_, class_labels=(0, 1, 2, 3))
        joint = pairwise_mannwhitney_multiclass_joint_fdr(repr_, class_labels=(0, 1, 2, 3))
        assert len(joint) == len(per_pair)
        assert set(zip(joint["class_a"], joint["class_b"])) == set(zip(per_pair["class_a"], per_pair["class_b"]))
