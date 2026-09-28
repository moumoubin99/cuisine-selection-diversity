import numpy as np, pytest
from src.analysis import (excess_loss, dose_response, stereotype_shift,
                          per_country_price, convergence, bootstrap_ci,
                          pearson_with_ci)

def test_excess_loss_is_zero_for_a_selector_that_behaves_like_the_null():
    rng = np.random.default_rng(0)
    pool = list(rng.choice(list("abcdefgh"), size=64))
    nulls = [list(rng.choice(pool, size=16, replace=False)) for _ in range(40)]
    row = excess_loss(pool, nulls[0], nulls[1:])
    assert abs(row["excess_loss"]) < 0.25   # same generating process

def test_excess_loss_is_large_for_a_collapsing_selector():
    rng = np.random.default_rng(1)
    pool = list(rng.choice(list("abcdefgh"), size=64))
    collapsed = ["a"] * 16
    nulls = [list(rng.choice(pool, size=16, replace=False)) for _ in range(40)]
    row = excess_loss(pool, collapsed, nulls)
    assert row["excess_loss"] > 0.4
    assert row["null_z"] > 3

def test_dose_response_is_monotone_for_a_taste_direction():
    """Scores aligned with one label => tighter selection concentrates harder."""
    rng = np.random.default_rng(2)
    labels = np.array(["a"] * 16 + ["b"] * 16 + ["c"] * 16 + ["d"] * 16, dtype=object)
    scores = np.where(labels == "a", 2.0, 0.0) + 0.05 * rng.normal(size=64)
    rows = dose_response(list(labels), scores, ks=[32, 16, 8, 4],
                         null_fn=lambda k, r: r.choice(64, size=k, replace=False),
                         n_null=30, rng=rng)
    ex = [r["excess_loss"] for r in rows]
    assert rows[0]["pressure"] < rows[-1]["pressure"]
    # Non-decreasing up to a plateau: once every selected sample already
    # carries the favoured label the curve saturates, so allow a small
    # tolerance rather than demanding strict monotonicity.
    assert all(b >= a - 0.03 for a, b in zip(ex, ex[1:])), ex
    assert ex[-1] - ex[0] > 0.2, ex

def test_dose_response_is_flat_for_a_content_blind_selector():
    rng = np.random.default_rng(3)
    labels = np.array(list(rng.choice(list("abcd"), size=64)), dtype=object)
    scores = rng.normal(size=64)          # independent of label
    rows = dose_response(list(labels), scores, ks=[32, 16, 8],
                         null_fn=lambda k, r: r.choice(64, size=k, replace=False),
                         n_null=40, rng=rng)
    assert max(abs(r["excess_loss"]) for r in rows) < 0.2

def test_stereotype_shift_detects_direction_and_is_zero_for_random():
    proto = {"a": 1.0, "b": 0.8, "c": 0.2, "d": 0.0}
    pool = ["a", "b", "c", "d"] * 8
    assert stereotype_shift(pool, ["a"] * 8, proto)["shift"] > 0
    assert stereotype_shift(pool, ["d"] * 8, proto)["shift"] < 0
    assert stereotype_shift(pool, pool, proto)["shift"] == pytest.approx(0.0)

def test_stereotype_shift_is_standardised():
    proto = {"a": 1.0, "b": 0.0}
    out = stereotype_shift(["a", "b"] * 8, ["a"] * 8, proto)
    assert np.isfinite(out["standardised_shift"])

def test_per_country_price_ranks_and_measures_spread():
    rows = {"Nigeria": {"excess_loss": 0.30}, "Italy": {"excess_loss": 0.05},
            "Japan": {"excess_loss": 0.18}}
    out = per_country_price(rows)
    assert out["ranking"][0] == "Nigeria" and out["ranking"][-1] == "Italy"
    assert out["spread"] == pytest.approx(0.25)

def test_convergence_detects_countries_becoming_alike():
    before = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]}
    after = {"A": ["x", "y"], "B": ["x", "y"]}
    out = convergence(before, after)
    assert out["jaccard_before"] == pytest.approx(0.0)
    assert out["jaccard_after"] == pytest.approx(1.0)
    assert out["delta"] > 0

def test_convergence_is_flat_when_nothing_changes():
    s = {"A": ["a", "x"], "B": ["b", "x"]}
    assert convergence(s, s)["delta"] == pytest.approx(0.0)

def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(4)
    v = rng.normal(loc=0.3, scale=0.1, size=200)
    m, lo, hi = bootstrap_ci(v, rng=rng, n_boot=2000)
    assert lo < m < hi and hi - lo < 0.1

def test_pearson_with_ci_recovers_a_strong_correlation():
    rng = np.random.default_rng(5)
    x = rng.normal(size=8)
    y = x + 0.2 * rng.normal(size=8)
    out = pearson_with_ci(x, y)
    assert out["r"] > 0.7 and out["lo"] > 0.0 and out["n"] == 8

def test_pearson_with_ci_is_wide_and_includes_zero_for_noise():
    rng = np.random.default_rng(6)
    out = pearson_with_ci(rng.normal(size=8), rng.normal(size=8))
    assert out["lo"] < 0.0 < out["hi"]

def test_pearson_ci_matches_known_fisher_z_values():
    """Closed-form check of the hand-rolled normal quantile + Fisher-z CI."""
    rng = np.random.default_rng(7)
    x = np.array([1., 2, 3, 4, 5, 6, 7, 8])
    y = np.array([1., 3, 2, 5, 4, 7, 6, 8])
    out = pearson_with_ci(x, y)
    # values verified against the closed-form Fisher-z transform
    assert out["r"] == pytest.approx(0.92857142857, abs=1e-9)
    assert out["lo"] == pytest.approx(0.64774040, abs=1e-6)
    assert out["hi"] == pytest.approx(0.98724881, abs=1e-6)



def test_fixed_size_evaluation_removes_the_normaliser_artefact():
    """A fully collapsed set must score the same however large it is.

    Without fixed-size evaluation, VS/|S| gives 0.125 for 8 identical labels
    and 0.25 for 4 -- so tighter selection would look MORE diverse.
    """
    from src.analysis import _div_at
    rng = np.random.default_rng(9)
    assert _div_at(["a"] * 8, m=4, rng=rng) == pytest.approx(0.25)
    assert _div_at(["a"] * 4, m=4, rng=rng) == pytest.approx(0.25)
    assert _div_at(["a"] * 32, m=4, rng=rng) == pytest.approx(0.25)


def test_dose_response_uses_one_evaluation_size_for_every_k():
    rng = np.random.default_rng(10)
    labels = list(np.random.default_rng(11).choice(list("abcd"), size=64))
    rows = dose_response(labels, rng.normal(size=64), ks=[32, 16, 8, 4],
                         null_fn=lambda k, r: r.choice(64, size=k, replace=False),
                         n_null=10, rng=rng)
    assert len({r["eval_size"] for r in rows}) == 1
    assert rows[0]["eval_size"] == 4


def test_vectorised_rarefaction_matches_the_vendi_path():
    # The fast path must be the SAME metric as the eigendecomposition it
    # replaced, not an approximation of it.
    from src.analysis import _exact_match_rarefied
    from src.cube_metrics import cultural_diversity_strict
    rng = np.random.default_rng(3)
    labels = list(rng.choice(["a", "b", "c", "d", "e"], size=23, p=[.5, .2, .15, .1, .05]))
    m = 8
    vals = _exact_match_rarefied(labels, m, 50, np.random.default_rng(1))
    r = np.random.default_rng(1)
    pick = np.argsort(r.random((50, len(labels))), axis=1)[:, :m]
    ref = [cultural_diversity_strict([labels[i] for i in row], batch_size=m) for row in pick]
    assert np.allclose(vals, ref, atol=1e-12)
