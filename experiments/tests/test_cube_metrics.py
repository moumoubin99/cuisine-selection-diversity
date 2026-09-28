import numpy as np, pytest
from src.cube_metrics import (vendi_score, vendi_score_from_kernel, GLOBAL_KERNEL,
                              cultural_diversity, cultural_diversity_strict,
                              simpson_diversity, coverage)

def test_vendi_identity_is_n():
    assert vendi_score_from_kernel(np.eye(8)) == pytest.approx(8.0)

def test_vendi_all_identical_is_one():
    assert vendi_score_from_kernel(np.ones((8, 8))) == pytest.approx(1.0)

def test_vendi_two_equal_blocks_is_two():
    K = np.zeros((8, 8)); K[:4, :4] = 1; K[4:, 4:] = 1
    assert vendi_score_from_kernel(K) == pytest.approx(2.0)

def test_exact_match_vendi_equals_label_perplexity():
    """Key property: under the exact-match kernel the Vendi score is exactly
    exp(H(p)) of the empirical label distribution -- the *effective number*
    of artifacts, not the raw count of distinct ones.  {pizza x2, pasta,
    gelato} has p = (1/2, 1/4, 1/4), so VS = 2^1.5 = 2.828, not 3."""
    labels = ["pizza", "pizza", "pasta", "gelato"]
    samples = [(l, None, None) for l in labels]
    assert vendi_score(samples, GLOBAL_KERNEL) == pytest.approx(2 ** 1.5)

def test_exact_match_vendi_equals_perplexity_generally():
    rng = np.random.default_rng(7)
    for _ in range(20):
        labels = list(rng.choice(list("abcdef"), size=16,
                                 p=rng.dirichlet(np.ones(6))))
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        expected = float(np.exp(-(p * np.log(p)).sum()))
        got = vendi_score([(l, None, None) for l in labels], GLOBAL_KERNEL)
        assert got == pytest.approx(expected, rel=1e-9)

def test_within_culture_mode_is_artifact_level():
    diverse = [f"artifact_{i}" for i in range(8)]
    collapsed = ["pizza"] * 8
    assert cultural_diversity_strict(diverse) == pytest.approx(1.0)
    assert cultural_diversity_strict(collapsed) == pytest.approx(1.0 / 8)

def test_global_mode_is_continent_level_only():
    """Two different Italian artifacts are INDISTINGUISHABLE in global mode."""
    recs = [{"continent": "Europe", "country": "Italy", "artifact": a}
            for a in ["pizza", "pasta", "gelato", "risotto",
                      "lasagne", "tiramisu", "focaccia", "polenta"]]
    assert cultural_diversity_strict(recs, _global=True) == pytest.approx(1.0 / 8)

def test_released_variant_truncates_below_32():
    """Documented hazard: <32 labels are silently truncated to 24."""
    labels = [f"a{i}" for i in range(31)]
    # 24 labels -> 3 full chunks of 8; released version keeps only those
    assert cultural_diversity(labels) == pytest.approx(1.0)

def test_released_variant_penalises_partial_chunk():
    """Documented hazard: a trailing chunk of 4 distinct labels scores 4/8, not 1."""
    labels = [f"a{i}" for i in range(36)]
    released = cultural_diversity(labels)
    strict = cultural_diversity_strict(labels)
    assert strict == pytest.approx(1.0)
    assert released < strict

def test_selection_reduces_diversity_monotonically():
    rng = np.random.default_rng(0)
    pool = list(rng.choice([f"a{i}" for i in range(12)], size=64))
    full = cultural_diversity_strict(pool)
    collapsed = cultural_diversity_strict(["a0"] * 64)
    assert collapsed < full

def test_simpson_and_coverage():
    assert simpson_diversity(["a"] * 10) == pytest.approx(0.0)
    assert simpson_diversity(["a", "b"]) == pytest.approx(0.5)
    assert coverage(["a", "b"], ["a", "b", "c", "d"]) == pytest.approx(0.5)


def test_strict_variant_does_not_zero_out_sets_smaller_than_batch():
    """Regression: k=4 with batch_size=8 must not silently return 0.0 -- that
    would blank the most-selective conditions in the dose-response curve."""
    assert cultural_diversity_strict(["a", "b", "c", "d"]) == pytest.approx(1.0)
    assert cultural_diversity_strict(["a", "a", "a", "a"]) == pytest.approx(0.25)
    assert cultural_diversity_strict([]) == pytest.approx(0.0)


def test_strict_variant_is_bounded_by_one_over_n_and_one():
    rng = np.random.default_rng(3)
    for n in (4, 8, 16, 40):
        labels = list(rng.choice(list("abcde"), size=n))
        v = cultural_diversity_strict(labels, batch_size=min(8, n))
        assert 0.0 < v <= 1.0 + 1e-9
