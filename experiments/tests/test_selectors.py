import numpy as np, pytest
from src.selectors import (select_topk, random_selector, filesize_selector,
                           random_direction_selector, mmr_select, dpp_map_select)
from src.cube_metrics import cultural_diversity_strict

def test_topk_is_sorted_and_correct():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    assert list(select_topk(s, 2)) == [1, 3]
    assert list(select_topk(s, 4)) == [1, 3, 2, 0]

def test_topk_handles_k_larger_than_n():
    assert len(select_topk(np.array([1.0, 2.0]), 5)) == 2

def test_nulls_have_matched_selectivity():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(64, 16)); sizes = rng.normal(size=64)
    for sel in (random_selector(64, 8, rng), filesize_selector(sizes, 8),
                random_direction_selector(E, 8, rng)):
        assert len(set(map(int, sel))) == 8

def test_random_direction_null_is_a_linear_selector():
    """The null must have the same functional form as a linear CLIP scorer:
    selecting on <v, E(x)> for a fixed v.  Two draws differ; one draw is
    deterministic given the seed."""
    E = np.random.default_rng(1).normal(size=(32, 8))
    a = random_direction_selector(E, 4, np.random.default_rng(2))
    b = random_direction_selector(E, 4, np.random.default_rng(2))
    c = random_direction_selector(E, 4, np.random.default_rng(3))
    assert list(a) == list(b)
    assert list(a) != list(c)

def test_mmr_lambda_one_reduces_to_topk():
    rng = np.random.default_rng(4)
    s = rng.normal(size=32); E = rng.normal(size=(32, 8))
    assert list(mmr_select(s, E, 5, lam=1.0)) == list(select_topk(s, 5))

def test_mmr_increases_diversity_over_topk_on_clustered_data():
    """Construct 4 tight clusters where the top-scoring items all sit in one
    cluster.  Top-k collapses onto that cluster; MMR must not."""
    rng = np.random.default_rng(5)
    labels, E, s = [], [], []
    for c in range(4):
        centre = np.zeros(8); centre[c] = 5.0
        for _ in range(8):
            E.append(centre + 0.01 * rng.normal(size=8))
            labels.append(f"cluster{c}")
            s.append(3.0 if c == 0 else rng.normal())
    E, s, labels = np.array(E), np.array(s), np.array(labels)
    top = labels[select_topk(s, 4)]
    mmr = labels[mmr_select(s, E, 4, lam=0.3)]
    assert len(set(top)) == 1
    assert len(set(mmr)) >= 3
    assert cultural_diversity_strict(list(mmr)) > cultural_diversity_strict(list(top))

def test_dpp_also_escapes_the_dominant_cluster():
    rng = np.random.default_rng(6)
    labels, E, s = [], [], []
    for c in range(4):
        centre = np.zeros(8); centre[c] = 5.0
        for _ in range(8):
            E.append(centre + 0.01 * rng.normal(size=8))
            labels.append(f"cluster{c}")
            s.append(3.0 if c == 0 else rng.normal())
    E, s, labels = np.array(E), np.array(s), np.array(labels)
    dpp = labels[dpp_map_select(s, E, 4, alpha=1.0)]
    assert len(set(dpp)) >= 3

def test_dpp_returns_k_distinct_indices():
    rng = np.random.default_rng(8)
    s = rng.normal(size=40); E = rng.normal(size=(40, 12))
    sel = dpp_map_select(s, E, 10)
    assert len(sel) == 10 and len(set(map(int, sel))) == 10


def test_permutation_null_preserves_score_distribution():
    from src.selectors import permutation_null
    rng = np.random.default_rng(20)
    s = rng.normal(size=64)
    sel = permutation_null(s, 8, rng)
    assert len(sel) == 8 and len(set(map(int, sel))) == 8


def test_permutation_null_is_uninformative_about_labels():
    """Over many draws the permutation null must not prefer any label."""
    from src.selectors import permutation_null
    rng = np.random.default_rng(21)
    labels = np.array(["a"] * 32 + ["b"] * 32, dtype=object)
    scores = np.where(labels == "a", 2.0, 0.0)      # perfectly informative
    frac_a = np.mean([np.mean(labels[permutation_null(scores, 8, rng)] == "a")
                      for _ in range(400)])
    assert 0.40 < frac_a < 0.60


def test_topk_on_informative_scores_is_not_uninformative():
    labels = np.array(["a"] * 32 + ["b"] * 32, dtype=object)
    scores = np.where(labels == "a", 2.0, 0.0)
    assert np.all(labels[select_topk(scores, 8)] == "a")


def test_permutation_null_does_not_prefer_any_pool_position():
    """Mutation guard for the code review's finding 12.

    `test_permutation_null_preserves_score_distribution` above checks only
    that eight distinct indices come back, which a selector returning the
    first k positions also satisfies.  Over many draws each of the 32
    positions must be selected about k/n = 25% of the time; a first-k selector
    leaves 24 of them at exactly zero.
    """
    from src.selectors import permutation_null
    rng = np.random.default_rng(7)
    n, k, reps = 32, 8, 2000
    scores = rng.normal(size=n)
    counts = np.zeros(n)
    for _ in range(reps):
        counts[permutation_null(scores, k, rng)] += 1
    freq = counts / reps
    assert freq.sum() == pytest.approx(k)
    assert freq.min() > 0.5 * k / n, freq.min()
    assert freq.max() < 1.8 * k / n, freq.max()


def test_topk_tie_policy_is_the_documented_one():
    """The code review's tie reproduction, made executable (finding 9).

    `argpartition` returned [2, 4, 10, 8, 5] for this vector -- a different
    MEMBERSHIP, not merely a different order, because index 5 displaces
    index 1 at the cutoff.  The stable sort gives the documented answer.
    """
    s = np.array([0., 1., 2., 1., 2., 1., 0., 0., 2., 1., 2., 0.])
    assert list(select_topk(s, 5)) == [2, 4, 8, 10, 1]
    assert list(select_topk(s, 8)) == [2, 4, 8, 10, 1, 3, 5, 9]
