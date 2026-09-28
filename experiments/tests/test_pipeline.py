"""Pipeline tests, built around planted worlds with known answers.

Three worlds matter:
  effect  -- a real taste direction; the endpoint must fire
  null    -- a pure-noise scorer; the endpoint must stay quiet
  merging -- no taste at all, but the TAGGER merges labels in the high-scoring
             set.  This is the Phase-4 reviewer's R1: the machine endpoint
             fires and the audit endpoint does not.  If the pipeline cannot
             show that separation, it cannot tell a cultural finding from an
             annotation artefact.
"""
import numpy as np
import pytest

from src.backends import MockGenerator, MockGeoTagger, MockScorer
from src.data import Prompt
from src.pipeline import (ESTABLISHED, OFF_COUNTRY, PLAUSIBLE, TAGGER_ERROR,
                          UNRESOLVED, _endpoint, classify_tags, run_audit,
                          run_pool)

VOCAB = {"Nigeria": [f"n{i}" for i in range(40)]}
PROMPT = Prompt("cuisine|Nigeria|A", "An image of a traditional dish from Nigeria",
                "Nigeria", "cuisine", "A")


def _world(taste, merge_top_frac=0.0, err=None, seed=0):
    g = MockGenerator(VOCAB, seed=seed)
    w = g.artifact_embedding("Nigeria", "n0")
    w = w / np.linalg.norm(w)
    s = MockScorer(w, taste_weight=taste, fidelity_weight=0.0, noise=0.05, seed=seed + 1)
    t = MockGeoTagger(err or {}, seed=seed + 2, merge_top_frac=merge_top_frac,
                      canonical_by_country={"Nigeria": "n0"})
    return g, s, t


def test_classify_tags_is_four_way_and_loses_nothing():
    tags = [("C", "Nigeria", "n0"), ("C", "Italy", "pizza"),
            ("C", "Nigeria", "invented"), ("U", "U", TAGGER_ERROR)]
    st, lab = classify_tags(tags, "Nigeria", ["n0", "n1"])
    assert st == [ESTABLISHED, OFF_COUNTRY, PLAUSIBLE, UNRESOLVED]
    assert lab[3] is None and lab[2] == "invented"
    assert len(st) == len(tags)


def test_plausible_unlisted_is_not_folded_into_error():
    """R3: non-membership in CSpace is not evidence of inauthenticity."""
    st, _ = classify_tags([("C", "Nigeria", "a_real_dish_wikidata_lacks")],
                          "Nigeria", ["n0"])
    assert st == [PLAUSIBLE]


def test_pool_reports_all_four_status_counts_and_the_deficit():
    g, s, t = _world(taste=1.0, err={"Nigeria": 0.2})
    r = run_pool(g, s, t, PROMPT, n=64, k=16, m_auth=8,
                 cspace_vocab=VOCAB["Nigeria"][:20], pool_seed=3, n_null=40)
    c = r["status_counts"]
    assert sum(c.values()) == 64
    assert c[UNRESOLVED] > 0 and c[PLAUSIBLE] > 0
    assert r["authenticity_deficit"] == pytest.approx(1 - c[ESTABLISHED] / 64)


def test_short_pools_are_non_estimable_not_silently_dropped():
    """R3: matching k does not match the number of authentic observations."""
    g, s, t = _world(taste=1.0, err={"Nigeria": 0.95})
    r = run_pool(g, s, t, PROMPT, n=32, k=16, m_auth=8,
                 cspace_vocab=VOCAB["Nigeria"], pool_seed=4, n_null=20)
    assert r["conditional"]["non_estimable"]
    assert "reason" in r["conditional"]
    assert r["authenticity_deficit"] > 0.8      # the deficit is still reported


def test_contrast_is_negative_when_selection_costs_diversity():
    g, s, t = _world(taste=1.0)
    r = run_pool(g, s, t, PROMPT, n=128, k=16, m_auth=8,
                 cspace_vocab=VOCAB["Nigeria"], pool_seed=5, n_null=100)
    assert r["conditional"]["delta"] < -0.05, r["conditional"]
    assert r["conditional"]["proportional_change"] < 0


def test_label_merging_fires_the_machine_endpoint_but_not_the_audit_endpoint():
    """R1, made executable: no taste at all, only score-dependent labelling."""
    g, s, t = _world(taste=0.0, merge_top_frac=0.25)
    machine, audited = [], []
    for j in range(8):
        r = run_pool(g, s, t, PROMPT, n=128, k=16, m_auth=8,
                     cspace_vocab=VOCAB["Nigeria"], pool_seed=100 + j,
                     n_null=100, audit=True, tagger_wants_scores=True)
        machine.append(r["raw"]["delta"])
        audited.append(r["audit"]["delta"])
    machine, audited = np.mean(machine), np.mean(audited)
    assert machine < -0.15, machine            # a convincing "finding"
    assert abs(audited) < 0.06, audited        # which is not there in truth
    assert audited - machine > 0.10


def test_audit_endpoint_agrees_with_machine_when_the_tagger_is_honest():
    g, s, t = _world(taste=1.0, merge_top_frac=0.0)
    gaps = []
    for j in range(6):
        r = run_pool(g, s, t, PROMPT, n=128, k=16, m_auth=8,
                     cspace_vocab=VOCAB["Nigeria"], pool_seed=200 + j,
                     n_null=100, audit=True, tagger_wants_scores=True)
        gaps.append(r["machine_minus_audit"])
    assert abs(np.mean(gaps)) < 0.06, np.mean(gaps)


def test_run_audit_replicates_pools_and_reports_an_equivalence_verdict():
    """Eight countries, because the verdict is taken on the country marginal.

    An earlier version of this test used three countries and still expected a
    verdict.  It cannot have one: the equivalence test is applied to the eight
    per-country mean contrasts, which is the unit the pre-registered 10% margin
    was written about, and three of them are below the minimum that supports a
    calibrated bound.  The test now runs the design that the claim needs.
    """
    names = ("Nigeria", "Italy", "Japan", "Mexico", "India", "Egypt",
             "Brazil", "Norway")
    vocab = {c: [f"{c[:3].lower()}{i}" for i in range(40)] for c in names}
    g = MockGenerator(vocab, seed=0)
    w = g.artifact_embedding(None, "nig0"); w = w / np.linalg.norm(w)
    s = MockScorer(w, taste_weight=1.0, fidelity_weight=0.0, noise=0.05, seed=1)
    t = MockGeoTagger({}, seed=2)
    prompts = [Prompt(f"cuisine|{c}|A", f"An image of a traditional dish from {c}",
                      c, "cuisine", "A") for c in names]
    out = run_audit(g, s, t, prompts, vocab, n=64, k=16, m_auth=8, n_pools=4,
                    seed=7, n_null=40)
    assert out["n_pools_total"] == 32
    assert out["conditional"]["n_estimable"] == 32
    assert out["conditional"]["equivalence"]["verdict"] in (
        "harmful", "equivalent", "inconclusive")
    assert set(out["conditional"]["heterogeneity"]["country_n_pools"].values()) == {4}
    marg = out["conditional"]["marginal"]
    assert marg["complete_design"] and marg["missing_countries"] == []
    # The marginal is the mean of the eight country means, not of 32 pools.
    assert marg["mean_delta"] == pytest.approx(
        float(np.mean(list(marg["country_means"].values()))), abs=1e-12)


def test_selection_ranks_the_whole_pool_before_any_filtering():
    """The code reviewer's reproduction, made executable.

    Four high-scoring off-country images followed by four low-scoring
    established ones, k=4.  Filtering first and then taking the top-4 of the
    survivors reports a clean contrast on images the deployed system would
    never have shipped; ranking all N first shows that the top-4 contains no
    established image at all.
    """
    labels = ["x0", "x1", "x2", "x3", "n0", "n1", "n2", "n3"]
    scores = np.array([9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0])
    established = [False] * 4 + [True] * 4
    rng = np.random.default_rng(0)
    out = _endpoint(labels, scores, k=4, m_auth=2, n_null=50, rng=rng,
                    keep=established, keep_name="established")
    assert out["non_estimable"], out
    assert out["n_kept_selected"] == 0
    assert out["selected_deficit"] == 1.0
    assert out["kept_shift"] < 0            # selection destroys authentic images


def test_null_draws_that_fall_short_are_counted_not_quietly_dropped():
    """Conditioning the null on having enough authentic images biases it."""
    labels = [f"n{i}" for i in range(4)] + [f"x{i}" for i in range(28)]
    scores = np.concatenate([np.full(4, 5.0), np.full(28, 1.0)])
    keep = [True] * 4 + [False] * 28
    out = _endpoint(labels, scores, k=8, m_auth=4, n_null=60,
                    rng=np.random.default_rng(1), keep=keep,
                    keep_name="established")
    assert out["null_insufficient_frac"] > 0.2
    assert out["non_estimable"] and "biased reference" in out["reason"]


def test_run_audit_with_one_pool_per_country_cannot_test_heterogeneity():
    """Eight countries x one pool each licenses no heterogeneity claim."""
    vocab = {c: [f"{c[:2].lower()}{i}" for i in range(40)] for c in ("Nigeria", "Italy")}
    g = MockGenerator(vocab, seed=0)
    w = g.artifact_embedding(None, "ni0"); w = w / np.linalg.norm(w)
    s = MockScorer(w, taste_weight=1.0, seed=1)
    t = MockGeoTagger({}, seed=2)
    prompts = [Prompt(f"cuisine|{c}|A", f"dish from {c}", c, "cuisine", "A") for c in vocab]
    out = run_audit(g, s, t, prompts, vocab, n=64, k=16, m_auth=8, n_pools=1,
                    seed=8, n_null=40)
    assert out["conditional"]["heterogeneity"]["non_estimable"]
