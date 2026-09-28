import numpy as np, pytest
from src.backends import MockGenerator, MockScorer, MockGeoTagger

VOCAB = {"Nigeria": [f"n{i}" for i in range(20)],
         "Italy": [f"i{i}" for i in range(20)]}

def test_generator_follows_a_zipf_distribution():
    g = MockGenerator(VOCAB, seed=0)
    imgs = g.generate("p", 2000, seed=1, country="Nigeria")
    from collections import Counter
    c = Counter(im.artifact for im in imgs)
    assert c["n0"] > c["n19"] * 3          # head dominates the tail

def test_generator_plants_the_fidelity_confound_only_when_asked():
    pure = MockGenerator(VOCAB, seed=0, rare_artifact_fidelity=1.0)
    conf = MockGenerator(VOCAB, seed=0, rare_artifact_fidelity=0.2)
    assert {im.fidelity for im in pure.generate("p", 50, 1, country="Italy")} == {1.0}
    assert len({im.fidelity for im in conf.generate("p", 50, 1, country="Italy")}) > 1

def test_scorer_with_zero_taste_is_uninformative_about_artifacts():
    g = MockGenerator(VOCAB, seed=0)
    imgs = g.generate("p", 400, 1, country="Italy")
    s = MockScorer(np.zeros(64), taste_weight=0.0, fidelity_weight=0.0, noise=1.0)
    sc = s.score(imgs)
    head = np.array([im.artifact == "i0" for im in imgs], dtype=float)
    assert abs(np.corrcoef(sc, head)[0, 1]) < 0.15

def test_scorer_with_taste_prefers_the_planted_direction():
    g = MockGenerator(VOCAB, seed=0)
    imgs = g.generate("p", 400, 1, country="Italy")
    w = g.artifact_embedding("Italy", "i0")
    w = w / np.linalg.norm(w)
    sc = MockScorer(w, taste_weight=1.0, noise=0.01).score(imgs)
    top = np.argsort(-sc)[:20]
    assert sum(imgs[i].artifact == "i0" for i in top) >= 15

def test_geotagger_is_exact_without_error():
    g = MockGenerator(VOCAB, seed=0)
    imgs = g.generate("p", 50, 1, country="Nigeria")
    tags = MockGeoTagger().tag(imgs)
    assert [t[2] for t in tags] == [im.artifact for im in imgs]

def test_geotagger_plants_differential_error():
    g = MockGenerator(VOCAB, seed=0)
    t = MockGeoTagger({"Nigeria": 0.5, "Italy": 0.0}, seed=2)
    ng = t.tag(g.generate("p", 400, 1, country="Nigeria"))
    it = t.tag(g.generate("p", 400, 1, country="Italy"))
    err_ng = np.mean([x[2] == "__tagger_error__" for x in ng])
    err_it = np.mean([x[2] == "__tagger_error__" for x in it])
    assert 0.4 < err_ng < 0.6 and err_it == 0.0


def test_adjacent_pools_share_no_image_seeds():
    """The code review's finding 2, made executable.

    The previous scheme assigned image seeds `pool_seed + i`, so two adjacent
    N=64 pools for the same prompt shared 63 of their 64 seeds and four pools
    drew only 67 distinct images instead of 256.  Pools are the unit of
    replication for every interval in the analysis, so that dependence is not
    a nuisance -- it invalidates the standard errors.
    """
    from src.backends import image_seeds_for_pool
    n = 64
    seeds = [set(map(int, image_seeds_for_pool(s, n))) for s in (7, 8, 9, 10)]
    for a in range(len(seeds)):
        assert len(seeds[a]) == n
        for b in range(a + 1, len(seeds)):
            assert not (seeds[a] & seeds[b]), (a, b, seeds[a] & seeds[b])


def test_the_run_audit_seed_grid_is_disjoint_across_prompts_and_pools():
    """`run_audit` lays pool seeds out as seed + 1000*prompt + pool."""
    from src.backends import image_seeds_for_pool
    pool_seeds = [7 + 1000 * i + j for i in range(8) for j in range(4)]
    assert len(set(pool_seeds)) == len(pool_seeds)
    all_seeds = [int(x) for s in pool_seeds for x in image_seeds_for_pool(s, 64)]
    assert len(set(all_seeds)) == len(all_seeds) == 32 * 64


def test_image_seeds_are_reproducible():
    from src.backends import image_seeds_for_pool
    assert list(image_seeds_for_pool(3, 16)) == list(image_seeds_for_pool(3, 16))
