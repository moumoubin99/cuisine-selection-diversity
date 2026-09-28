"""Estimators added in response to the Phase-4 critical review.

The heterogeneity test is the one that most needed checking: the reviewer
rejected the score-permutation null for A4 because it tests the wrong
hypothesis, so the replacement has to be shown to have correct size under
EQUAL-BUT-NONZERO country effects, which is exactly where the rejected null
would have fired spuriously.
"""
import numpy as np
import pytest

from src.analysis import (MIN_POOLS_FOR_HETEROGENEITY, country_marginal,
                          equivalence_test, heterogeneity_test,
                          heterogeneity_verdict, log_contrast,
                          pool_contrasts_by_country, rarefy_to_common_count,
                          simultaneous_country_intervals)


def test_log_contrast_is_a_proportional_change():
    out = log_contrast(1.10, [1.0] * 10)
    assert out["proportional_change"] == pytest.approx(0.10, abs=1e-9)
    out = log_contrast(0.9, [1.0] * 10)
    assert out["proportional_change"] == pytest.approx(-0.10, abs=1e-9)


def test_log_contrast_is_scale_free_where_the_raw_difference_is_not():
    """The reason for the log scale: a 7x vocabulary spread must not dominate."""
    small = log_contrast(0.9, [1.0] * 20)      # pool diversity ~1
    large = log_contrast(90.0, [100.0] * 20)   # pool diversity ~100
    assert small["delta"] == pytest.approx(large["delta"], abs=1e-9)
    assert (1.0 - 0.9) != (100.0 - 90.0)       # the raw difference is not


def test_log_contrast_marks_zero_diversity_non_estimable():
    assert log_contrast(0.0, [1.0])["non_estimable"]


def test_heterogeneity_size_is_measured_not_assumed():
    """All countries share the same nonzero effect: A4 must not fire at 5%.

    The assertion is two-sided on purpose.  The previous version of this test
    only checked that the rejection rate was BELOW a bound, and so passed
    against two implementations the code review later showed were badly
    wrong -- including one whose real rejection rate was 0.1%, which "passes"
    every upper bound while having no power at all.  A test of calibration has
    to fail in both directions.
    """
    rng = np.random.default_rng(0)
    fires = 0
    trials = 400
    for _ in range(trials):
        by = {f"C{i}": rng.normal(-0.3, 0.1 + 0.05 * i,
                                  size=MIN_POOLS_FOR_HETEROGENEITY)
              for i in range(8)}
        if heterogeneity_test(by, n_perm=1, rng=rng)["p_value"] < 0.05:
            fires += 1
    assert 0.015 < fires / trials < 0.09, fires / trials


def test_heterogeneity_flags_small_pool_counts_as_underpowered():
    rng = np.random.default_rng(11)
    by = {f"C{i}": rng.normal(-0.3, 0.12, size=4) for i in range(8)}
    out = heterogeneity_test(by, n_perm=50, rng=rng)
    assert out["underpowered"] and out["min_pools_per_country"] == 4
    big = {f"C{i}": rng.normal(-0.3, 0.12, size=12) for i in range(8)}
    assert not heterogeneity_test(big, n_perm=50, rng=rng)["underpowered"]


def test_heterogeneity_verdict_needs_a_disjoint_pair_not_just_a_p_value():
    """Welch is liberal at small n; the decision rule is what protects A4."""
    rng = np.random.default_rng(12)
    by = {f"C{i}": rng.normal(-0.2 - 0.02 * i, 0.02, size=8) for i in range(8)}
    het = heterogeneity_test(by, n_perm=1, rng=rng)
    wide = {c: v for c, v in by.items()}
    iv = simultaneous_country_intervals(wide, n_boot=600, rng=rng)
    v = heterogeneity_verdict(het, iv)
    assert v["verdict"] in ("heterogeneous", "not_shown")
    # A rejection with no disjoint pair must never be reported as heterogeneity.
    fake_iv = {"non_estimable": False,
               "intervals": {c: (-5.0, 5.0) for c in by},
               "point": {c: 0.0 for c in by}}
    assert heterogeneity_verdict({"p_value": 1e-9, "non_estimable": False},
                                 fake_iv)["verdict"] == "not_shown"


def test_heterogeneity_test_detects_genuinely_unequal_effects():
    rng = np.random.default_rng(1)
    by = {f"C{i}": rng.normal(-0.05 - 0.10 * i, 0.08, size=8) for i in range(8)}
    out = heterogeneity_test(by, n_perm=200, rng=rng)
    assert out["p_value"] < 0.01, out
    assert out["p_value_bootstrap_conservative"] >= out["p_value"]
    assert out["spread"] > 0.5


def test_heterogeneity_is_non_estimable_with_one_pool_per_country():
    by = {"A": np.array([-0.2]), "B": np.array([-0.4])}
    assert heterogeneity_test(by)["non_estimable"]


def test_simultaneous_intervals_are_wider_than_pointwise():
    rng = np.random.default_rng(2)
    by = {f"C{i}": rng.normal(-0.2, 0.1, size=8) for i in range(8)}
    out = simultaneous_country_intervals(by, n_boot=800, rng=rng)
    assert out["critical_value"] > 1.96
    for c, (lo, hi) in out["intervals"].items():
        assert lo < out["point"][c] < hi


def test_equivalence_test_returns_all_three_verdicts():
    rng = np.random.default_rng(3)
    quiet = rng.normal(0.0, 0.01, size=64)
    assert equivalence_test(quiet, n_boot=800, rng=rng)["verdict"] == "equivalent"
    harmful = rng.normal(-0.5, 0.05, size=64)
    assert equivalence_test(harmful, n_boot=800, rng=rng)["verdict"] == "harmful"
    noisy = rng.normal(-0.05, 0.6, size=8)
    assert equivalence_test(noisy, n_boot=800, rng=rng)["verdict"] == "inconclusive"


def test_equivalence_margin_is_ten_percent_by_default():
    out = equivalence_test(np.zeros(10) + 1e-9, n_boot=200)
    assert out["margin_proportional"] == pytest.approx(0.10, abs=1e-9)


def test_rarefy_marks_insufficient_authentic_counts_non_estimable():
    out = rarefy_to_common_count(["a", "b", "c"], m=8)
    assert out["non_estimable"] and out["n_authentic"] == 3


def test_rarefy_is_insensitive_to_authentic_count_at_fixed_diversity():
    """The bug this guards: filtering changes n, and entropy moves with n.

    Subsampling m without replacement from an i.i.d. sample of size n is
    itself an i.i.d. sample of size m, so the rarefied value is unbiased for
    the same population quantity at every n.  It is compared in expectation
    over independent pools, because a single small pool is a noisy draw --
    which is itself the reason the reviewer insists the unit of replication is
    the pool and that short pools be flagged rather than silently compared.
    """
    rng = np.random.default_rng(4)
    vocab = [f"a{i}" for i in range(12)]

    def mean_at(n, reps=40):
        vals = []
        for _ in range(reps):
            pool = list(rng.choice(vocab, size=n))
            vals.append(rarefy_to_common_count(pool, m=8, n_sub=100, rng=rng)["value"])
        return float(np.mean(vals))

    few, many = mean_at(20), mean_at(200)
    assert abs(few - many) < 0.02, (few, many)


def test_pool_contrasts_group_by_country_and_drop_non_estimable():
    rows = [{"country": "A", "delta": -0.1, "non_estimable": False},
            {"country": "A", "delta": -0.2, "non_estimable": False},
            {"country": "A", "delta": float("nan"), "non_estimable": True},
            {"country": "B", "delta": -0.3, "non_estimable": False}]
    by = pool_contrasts_by_country(rows)
    assert by["A"].size == 2 and by["B"].size == 1


def test_simultaneous_intervals_refuse_to_report_degenerate_bands():
    """The 8.1e10 critical value the code review reproduced.

    With few pools a bootstrap resample is constant with real probability; its
    standard error is zero; dividing by a 1e-12 floor produced "95% intervals"
    spanning billions of log units.  Too few pools has to come back as
    non-estimable, not as an enormous band.
    """
    by = {"A": np.array([-0.2, -0.2, -0.2, -0.2]),
          "B": np.array([-0.3, -0.1, -0.25, -0.15])}
    out = simultaneous_country_intervals(by, n_boot=500)
    assert out["non_estimable"] and "variance" in out["reason"]

    rng = np.random.default_rng(21)
    tiny = {f"C{i}": rng.normal(-0.2, 0.1, size=3) for i in range(8)}
    out = simultaneous_country_intervals(tiny, n_boot=2000, rng=rng)
    if not out["non_estimable"]:
        assert out["critical_value"] < 1e3, out["critical_value"]


def test_simultaneous_intervals_cover_all_countries_at_the_nominal_rate():
    rng = np.random.default_rng(22)
    covered = 0
    trials = 150
    for _ in range(trials):
        truth = {f"C{i}": -0.2 - 0.05 * i for i in range(8)}
        by = {c: rng.normal(mu, 0.12, size=10) for c, mu in truth.items()}
        out = simultaneous_country_intervals(by, n_boot=600, rng=rng)
        if out["non_estimable"]:
            continue
        if all(lo <= truth[c] <= hi for c, (lo, hi) in out["intervals"].items()):
            covered += 1
    assert covered / trials > 0.85, covered / trials


def test_equivalence_margins_are_asymmetric_around_zero():
    """A 10% loss is log(0.90), not -log(1.10).

    The code review's finding: the symmetric implementation made the harmful
    boundary a 9.09% loss, so a tight set of pools at a 9.5% loss came back
    `harmful` under a rule that had pre-registered 10%.
    """
    rng = np.random.default_rng(31)
    out = equivalence_test(rng.normal(-0.02, 0.01, 16), n_boot=200)
    lo_m, hi_m = out["margin_log"]
    assert lo_m == pytest.approx(np.log(0.90), abs=1e-12)
    assert hi_m == pytest.approx(np.log(1.10), abs=1e-12)

    at_95 = np.log(0.905) + np.random.default_rng(5).normal(0, 1e-4, size=32)
    assert equivalence_test(at_95, n_boot=400)["verdict"] != "harmful"
    at_11 = np.log(0.89) + np.random.default_rng(6).normal(0, 1e-4, size=32)
    assert equivalence_test(at_11, n_boot=400)["verdict"] == "harmful"


def test_equivalence_does_not_declare_equivalence_at_the_margin():
    """False-equivalence rate at the boundary, measured.

    The percentile bootstrap this replaced ran at 12.8% false equivalence with
    4 pools and 8.4% with 8, against a nominal 5%.
    """
    rng = np.random.default_rng(7)
    for n_pools in (4, 8):
        wrong = 0
        trials = 400
        for _ in range(trials):
            deltas = rng.normal(np.log(0.90), 0.08, size=n_pools)
            if equivalence_test(deltas, n_boot=1, rng=rng)["verdict"] == "equivalent":
                wrong += 1
        assert wrong / trials <= 0.06, (n_pools, wrong / trials)


def test_equivalence_is_non_estimable_below_the_minimum_pool_count():
    out = equivalence_test(np.array([-0.01, 0.0, 0.01]), n_boot=10)
    assert out["verdict"] == "non_estimable"


def test_country_marginal_weights_countries_equally():
    """Concatenating pools weights countries by how many happened to survive.

    Pools go missing exactly where authentic artifacts are scarce, so that
    weighting is informative rather than incidental.
    """
    by = {"A": np.full(20, -0.40), "B": np.full(2, 0.00)}
    out = country_marginal(by, n_boot=200)
    assert out["mean_delta"] == pytest.approx(-0.20, abs=1e-12)
    pooled = np.concatenate(list(by.values())).mean()
    assert pooled == pytest.approx(-0.3636, abs=1e-3)


def test_country_marginal_names_countries_that_dropped_out_entirely():
    by = {"A": np.full(6, -0.2), "B": np.full(6, -0.1)}
    out = country_marginal(by, n_boot=100, expected_countries=["A", "B", "C"])
    assert out["missing_countries"] == ["C"] and not out["complete_design"]


def test_country_marginal_interval_holds_the_country_set_fixed():
    """Resampling countries too would widen the band for a design fact."""
    rng = np.random.default_rng(8)
    by = {f"C{i}": rng.normal(-0.2 - 0.15 * i, 0.05, size=8) for i in range(6)}
    out = country_marginal(by, n_boot=2000, rng=rng)
    lo, hi = out["ci"]
    spread = np.std([v.mean() for v in by.values()])
    assert (hi - lo) < spread, (hi - lo, spread)


def test_log_contrast_averages_the_log_of_the_null_not_the_log_of_its_mean():
    """Mutation guard for the code review's finding 12.

    Both proportional-change tests above use a CONSTANT null, where
    mean(log D_r) and log(mean D_r) coincide, so they passed unchanged when
    the estimator was mutated to the second form.  Here the two null draws are
    1.0 and 4.0: their geometric mean is exactly 2.0 and their arithmetic mean
    is 2.5, so a selected diversity of 2.0 is a 0% change under the specified
    estimator and a -20% change under the mutant.
    """
    out = log_contrast(2.0, [1.0, 4.0])
    assert out["delta"] == pytest.approx(0.0, abs=1e-12)
    assert out["proportional_change"] == pytest.approx(0.0, abs=1e-12)
    assert out["log_null_mean"] == pytest.approx(np.log(2.0), abs=1e-12)
    assert abs(float(np.log(2.0 / 2.5))) > 0.2      # the mutant is not within tolerance


def test_log_contrast_is_scale_free_with_a_nonconstant_null():
    """The scale-free property, pinned to an exact value rather than a match.

    Asserting only that two designs agree cannot separate estimators, because
    both the specified one and the mean-then-log mutant are scale
    equivariant.  The null {0.5, 1, 2} has geometric mean exactly 1 and
    arithmetic mean 7/6, so the specified delta is log(0.9) and the mutant's
    is log(0.9) - log(7/6).
    """
    null = [0.5, 1.0, 2.0] * 7
    small = log_contrast(0.9, null)
    large = log_contrast(90.0, [100.0 * d for d in null])
    assert small["delta"] == pytest.approx(float(np.log(0.9)), abs=1e-12)
    assert large["delta"] == pytest.approx(float(np.log(0.9)), abs=1e-12)
    assert small["proportional_change"] == pytest.approx(-0.10, abs=1e-12)


def test_rarefaction_matches_exact_enumeration_on_small_pools():
    """Mutation guard for the code review's finding 12.

    The insensitivity test above compares two rarefied means to each other, so
    an implementation returning the constant 0.5 passes it.  These three
    four-image pools have different label distributions and therefore three
    different exact rarefied values, obtained by enumerating all C(4,2)=6
    two-image subsets of each:

        aabb -> (0.5 + 0.5 + 4 x 1.0) / 6 = 5/6
        aaab -> (3 x 0.5 + 3 x 1.0) / 6   = 3/4
        aaaa -> 6 x 0.5 / 6               = 1/2

    The Monte-Carlo estimate with 20000 subsamples has a standard error near
    0.003, so the 0.01 tolerance is about three standard errors and no
    constant can satisfy all three.
    """
    cases = {("a", "a", "b", "b"): 5.0 / 6.0,
             ("a", "a", "a", "b"): 3.0 / 4.0,
             ("a", "a", "a", "a"): 1.0 / 2.0}
    for labels, exact in cases.items():
        out = rarefy_to_common_count(list(labels), m=2, n_sub=20000,
                                     rng=np.random.default_rng(11))
        assert not out["non_estimable"]
        assert out["value"] == pytest.approx(exact, abs=0.01), (labels, out)
