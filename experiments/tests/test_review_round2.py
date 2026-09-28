"""One pinning test per finding of the SECOND cross-model code review.

The first review's fixes were verified by that review and six of its twelve
findings came back FIXED.  The nine findings pinned here are the ones it found
on re-reading, four of them P1, and its verdict on the state of the code was
that "the executed runner can falsely certify equivalence at twice nominal
size, certify an incomplete target, and accept annotations belonging to
different images."  Each test below is the reviewer's own reproduction, made
executable, so that a regression reintroduces a failing test rather than a
plausible number.

The annotation-join findings (P1-4, P2-8) live in `test_annotations.py`
alongside the rest of the join, and the special-function boundary cases (P2-9)
in `test_review_estimators.py`.
"""
import numpy as np
import pytest

from src.analysis import (MAX_NULL_INSUFFICIENT, _fixed_country_moments,
                          _student_t_ppf, country_marginal, dose_response,
                          equivalence_test, equivalence_test_country_marginal,
                          heterogeneity_verdict, null_bias_bounds)
from src.backends import MockGenerator, MockGeoTagger, MockScorer
from src.data import Prompt
from src.pipeline import _endpoint, run_audit
from src.selectors import random_selector

MARGIN = float(np.log(1.1))


# ------------------------------------------------------------------ P1-1

def _reviewers_miscalibration_world(rng, n_pools=4):
    """Seven countries at zero, one at 8*log(1.1); one country far noisier.

    The equal-weight marginal is exactly log(1.1), i.e. sitting ON the upper
    equivalence margin, so a calibrated test may call it `equivalent` at most
    alpha of the time.
    """
    mu = np.array([0.0] * 7 + [8 * MARGIN])
    sd = np.array([0.02] * 7 + [0.8])
    return {f"C{i}": rng.normal(mu[i], sd[i], n_pools) for i in range(8)}


def test_the_marginal_equivalence_test_uses_within_country_pool_variance():
    """P1: the runner fed country means to a test for exchangeable replicates.

    That makes the spread BETWEEN countries the noise, although the eight
    countries are fixed by design and not a sample of anything.  Measured over
    200,000 datasets of the world above: 10.13% false equivalence at a nominal
    5%.  The fixed-country calculation gives 4.42% on the same data.
    """
    rng = np.random.default_rng(0)
    by = _reviewers_miscalibration_world(rng)
    theta, se, df, _ = _fixed_country_moments(by)
    K = 8.0
    n = np.array([len(v) for v in by.values()], dtype=float)
    v = np.array([np.var(x, ddof=1) for x in by.values()])
    expect_se = float(np.sqrt(np.sum((v / n) / (K * K))))
    assert se == pytest.approx(expect_se, rel=1e-12)
    # ... and that is NOT the standard error of the country means.
    naive_se = float(np.std([x.mean() for x in by.values()], ddof=1) / np.sqrt(K))
    assert naive_se > 2 * se


def test_the_marginal_equivalence_test_is_calibrated_where_the_old_one_was_not():
    """A short run of the reviewer's simulation: 800 datasets, not 200,000.

    Enough to separate ~4.5% from ~10% with the seed pinned; the full-length
    numbers are in reviews/code_review_round2.md.
    """
    rng = np.random.default_rng(20260923)
    new = old = 0
    for _ in range(800):
        by = _reviewers_miscalibration_world(rng)
        if equivalence_test_country_marginal(by)["verdict"] == "equivalent":
            new += 1
        means = np.array([v.mean() for v in by.values()])
        if equivalence_test(means, n_boot=1, rng=rng)["verdict"] == "equivalent":
            old += 1
    assert new / 800.0 <= 0.07, new / 800.0          # nominal 5% on the margin
    assert old / 800.0 > 0.08, old / 800.0           # what the runner did


def test_one_pool_per_country_is_not_a_calibrated_equivalence_design():
    """P1: the minimum-pool rule had become a minimum-COUNTRY rule.

    Eight countries with one pool each satisfied `MIN_POOLS_FOR_EQUIVALENCE=4`
    -- eight numbers went in -- and returned `equivalent` on a zero-width
    interval, because a country with one pool contributes no variance.
    """
    by = {f"C{i}": np.array([0.0]) for i in range(8)}
    out = equivalence_test_country_marginal(by)
    assert out["verdict"] == "non_estimable"
    assert "< 2 pools" in out["reason"]
    # Two pools per country is the minimum that carries a variance estimate,
    # and there the test becomes estimable again.
    assert equivalence_test_country_marginal(
        {f"C{i}": np.array([0.0, 1e-3]) for i in range(8)}
    )["verdict"] != "non_estimable"


# ------------------------------------------------------------------ P1-2

def _world(countries, templates, seed=0):
    vocab = {c: [f"{c.lower()}{i}" for i in range(40)] for c in countries}
    g = MockGenerator(vocab, seed=seed)
    w = g.artifact_embedding(None, f"{countries[0].lower()}0")
    w = w / np.linalg.norm(w)
    s = MockScorer(w, taste_weight=1.0, fidelity_weight=0.0, noise=0.05, seed=seed + 1)
    t = MockGeoTagger({}, seed=seed + 2)
    prompts = [Prompt(f"cuisine|{c}|{tp}", f"a dish from {c}", c, "cuisine", tp)
               for c in countries for tp in templates]
    return g, s, t, prompts, vocab


def test_a_country_with_no_estimable_pool_does_not_get_a_full_target_verdict():
    """P1: four estimable countries and four absent ones still read `equivalent`.

    The pre-registered margin is a statement about the equal-weight marginal
    over ALL eight countries.  A marginal over the four that survived answers a
    different question, and reporting it under the same verdict vocabulary is
    how an incomplete design acquires a complete-design claim.
    """
    by = {f"C{i}": np.random.default_rng(i).normal(0.0, 0.01, 4) for i in range(4)}
    expected = [f"C{i}" for i in range(8)]
    out = equivalence_test_country_marginal(by, expected_countries=expected)
    assert out["verdict"] == "non_estimable"
    assert out["missing_countries"] == ["C4", "C5", "C6", "C7"]
    # Without the expectation the same data is perfectly estimable, so the
    # verdict is being gated by the DESIGN, not by the numbers.
    assert equivalence_test_country_marginal(by)["verdict"] != "non_estimable"


def test_losing_a_whole_template_makes_the_design_incomplete():
    """P1: `complete_design` looked only at countries.

    Dropping every B-template pool left it True, although the marginal had
    silently become a single-template marginal.
    """
    by = {f"C{i}": np.zeros(4) + 0.01 * i for i in range(8)}
    cells = [(f"C{i}", tp) for i in range(8) for tp in ("A", "B")]
    seen_a_only = [(f"C{i}", "A") for i in range(8)]
    out = country_marginal(by, n_boot=1, expected_countries=list(by),
                           expected_cells=cells, observed_cells=seen_a_only)
    assert not out["complete_design"]
    assert len(out["missing_cells"]) == 8
    assert country_marginal(by, n_boot=1, expected_countries=list(by),
                            expected_cells=cells,
                            observed_cells=cells)["complete_design"]


def test_the_runner_gates_the_verdict_on_a_complete_design():
    """End to end: a template that produced no estimable pool blocks the verdict."""
    countries = [f"C{i}" for i in range(8)]
    g, s, t, prompts, vocab = _world(countries, ("A", "B"))
    out = run_audit(g, s, t, prompts, vocab, n=32, k=8, m_auth=4, n_pools=2,
                    seed=3, n_null=20)
    assert out["conditional"]["design"]["complete_design"]
    # Now hide every B pool from the aggregation, exactly as a template whose
    # pools were all non-estimable would.
    kept = [r for r in out["rows"] if r["template"] == "A"]
    out2 = run_audit(g, s, t, [p for p in prompts if p.template == "A"],
                     vocab, n=32, k=8, m_auth=4, n_pools=2, seed=3, n_null=20)
    assert kept and out2["conditional"]["design"]["complete_design"]


# ------------------------------------------------------------------ P1-3

def test_the_reviewers_survivorship_pool_is_refused_not_reported():
    """P1: 16.67% of the null subsets excluded, -42.4% reported, +1.6% true.

    Pool `aaabbbbbb`, k=4, m=3, the last two images unresolved, top-k =
    [0,1,2,7].  Every null subset containing both unresolved images has only
    two usable labels and was silently dropped: 21 of C(9,4)=126.  The
    survivors are the kept-rich subsets, so survivorship, not selection,
    produced the -42.4%.  The complete-label exact contrast is +1.63%.
    """
    labels = ["a", "a", "a", "b", "b", "b", "b", "b", "b"]
    scores = np.array([9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0, 8.0, 0.5])
    keep = [True] * 7 + [False, False]
    out = _endpoint(labels, scores, k=4, m_auth=3, n_null=4000,
                    rng=np.random.default_rng(0), keep=keep, keep_name="resolved")
    assert out["non_estimable"], out
    assert out["null_insufficient_frac"] == pytest.approx(21 / 126.0, abs=0.02)
    assert "biased reference" in out["reason"]
    assert 21 / 126.0 > MAX_NULL_INSUFFICIENT      # and used to be under it


def test_a_clean_pool_still_estimates_and_carries_a_zero_width_bracket():
    """The guard must not refuse pools that have nothing wrong with them."""
    labels = [f"a{i % 5}" for i in range(32)]
    out = _endpoint(labels, np.arange(32.0), k=8, m_auth=4, n_null=200,
                    rng=np.random.default_rng(1), keep=[True] * 32,
                    keep_name="resolved")
    assert not out["non_estimable"]
    assert out["null_insufficient_frac"] == 0.0
    lo, hi = out["delta_bounds"]
    assert hi - lo == pytest.approx(0.0, abs=1e-12)
    assert lo == pytest.approx(out["delta"], abs=1e-12)


def test_the_bias_bracket_is_the_assumption_free_one():
    """`f * log(m)` wide; its ends are the contrasts the discarded draws give
    at the extremes of the normalised range, log diversity -log m and 0."""
    f, m, l_sel, l_surv = 21 / 126.0, 3, np.log(1.0), -0.551
    lo, hi = null_bias_bounds(l_sel, l_surv, f, m)
    assert hi - lo == pytest.approx(f * np.log(3.0), rel=1e-12)
    for l_excl in (-np.log(m), 0.0):
        c = l_sel - ((1 - f) * l_surv + f * l_excl)
        assert lo - 1e-12 <= c <= hi + 1e-12
    assert lo == pytest.approx(l_sel - (1 - f) * l_surv, rel=1e-12)


# ------------------------------------------------------------------ P2-5

def test_the_marginal_interval_is_studentised_not_a_raw_percentile():
    """P2: the percentile interval covered 88.92% at 4 pools, 92.38% at 8."""
    rng = np.random.default_rng(5)
    mu = np.linspace(-0.3, 0.05, 8)
    sd = np.array([0.05, 0.08, 0.12, 0.20, 0.05, 0.30, 0.10, 0.15])
    true = float(mu.mean())
    hits = 0
    trials = 1200
    for _ in range(trials):
        by = {f"C{i}": rng.normal(mu[i], sd[i], 4) for i in range(8)}
        lo, hi = country_marginal(by, n_boot=1, rng=rng)["ci"]
        hits += lo <= true <= hi
    assert 0.93 <= hits / trials <= 0.97, hits / trials


def test_the_marginal_interval_matches_the_fixed_country_formula():
    rng = np.random.default_rng(6)
    by = {f"C{i}": rng.normal(0.0, 0.1, 5) for i in range(8)}
    out = country_marginal(by, n_boot=1, rng=rng)
    theta, se, df, _ = _fixed_country_moments(by)
    t = _student_t_ppf(0.975, df)
    assert out["ci"][0] == pytest.approx(theta - t * se, rel=1e-12)
    assert out["ci_method"] == "student_t_fixed_countries"
    assert out["ci_percentile"] != out["ci"]


# ------------------------------------------------------------------ P2-6

def test_the_runner_executes_the_heterogeneity_decision_rule():
    """P2: `heterogeneity_verdict` was unit-tested and never called.

    Replacing every heterogeneity p-value with 1.0 still passed all nine smoke
    controls, so the round-1 disposition's claim that "the decision rule
    compensates" for Welch's small-sample liberality did not describe the code
    that runs.
    """
    countries = [f"C{i}" for i in range(4)]
    g, s, t, prompts, vocab = _world(countries, ("A",))
    out = run_audit(g, s, t, prompts, vocab, n=32, k=8, m_auth=4, n_pools=4,
                    seed=4, n_null=20)
    dec = out["conditional"]["heterogeneity_decision"]
    assert dec["verdict"] in ("heterogeneous", "not_shown", "non_estimable")
    if dec["verdict"] != "non_estimable":
        # the rule, re-derived from the two inputs the runner stored
        expect = heterogeneity_verdict(out["conditional"]["heterogeneity"],
                                       out["conditional"]["country_intervals"])
        assert dec["verdict"] == expect["verdict"]
        assert dec["welch_rejects"] == (
            out["conditional"]["heterogeneity"]["p_value"] < 0.05)


# ------------------------------------------------------------------ P2-7

def test_the_dose_curve_ranks_the_whole_pool_before_filtering():
    """P2: the caller handed `dose_response` an already-filtered pool.

    That restores the filter-first intervention at every point of the curve.
    The reviewer's reproduction reported N=4, a selected diversity of 1 and
    zero excess loss on a pool whose raw endpoint is correctly non-estimable
    with no kept selected image at all.
    """
    labels = ["x0", "x1", "x2", "x3", "n0", "n1", "n2", "n3"]
    scores = np.array([9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0])
    keep = [False] * 4 + [True] * 4
    rows = dose_response(labels, scores, ks=[4], null_fn=lambda kk, r:
                         random_selector(8, kk, r), n_null=40,
                         rng=np.random.default_rng(0), eval_size=2, keep=keep)
    assert len(rows) == 1
    row = rows[0]
    assert row["N"] == 8                      # the whole pool, not the survivors
    assert row["n_kept_selected"] == 0
    assert row["selected_deficit"] == 1.0
    assert row["non_estimable"] and "excess_loss" not in row


def test_the_smoke_controls_notice_when_every_welch_p_value_is_set_to_one():
    """P2-6 again, at the level the reviewer actually attacked it.

    Calling `heterogeneity_verdict` was not enough.  At four pools per country
    the simultaneous band is (correctly) non-estimable, so the decision came
    back `non_estimable` in all five smoke worlds and a control that only
    checked the rule had been *executed* was vacuous: the reviewer could
    replace every heterogeneity p-value with 1.0 and watch every control pass.

    The smoke suite now carries a sixth world in which heterogeneity is
    planted -- half the countries at the cliff, half inside the margin, run at
    the eight pools per country the band needs -- so the rule has a branch to
    take, and the consistency control has something to be inconsistent with.
    """
    import run as R

    planted = [c for c in R.WORLDS.values() if c.get("noise_by_country")]
    assert planted, "no world plants heterogeneity; the A4 control is vacuous"
    assert all(c.get("pools", 0) >= 8 for c in planted), (
        "the simultaneous band is not estimable below eight pools per country, "
        "so the decision would be `non_estimable` and the control vacuous again")

    intervals = {"intervals": {"A": (-2.0, -1.5), "B": (-0.1, 0.1)}}
    het = {"p_value": 1e-9}
    block = {"heterogeneity": het, "country_intervals": intervals,
             "heterogeneity_decision": heterogeneity_verdict(het, intervals)}
    assert block["heterogeneity_decision"]["verdict"] == "heterogeneous"
    assert R._het_decision_consistent(block)

    mutated = dict(block, heterogeneity={"p_value": 1.0})
    assert not R._het_decision_consistent(mutated), (
        "the stored decision says the Welch test rejected; a p-value of 1.0 "
        "must make that inconsistent, not agreeable")
