"""Known-label pools: the human-free replacement for the blinded human audit.

## What the human audit was for

The Phase-4 reviewer's most likely false finding (R1) is score-dependent label
merging: a VLM tagger names polished, canonical-looking images with a familiar
dish, several different dishes collapse onto one name *in the top-k*, and an
exact-match Vendi score reports homogenisation that selection never caused.
The pre-registered defence was two blinded human annotators on 2,048 images.
The user asked for no human-subject annotation, so that audit is gone.

## What replaces it

A pool whose true labels are known *by construction*.  Every image in a
known-label pool is generated from a prompt that NAMES its dish,

    "An image of {dish}, a traditional dish from {country}"

so the label is the prompt.  Within a pool the dishes are drawn i.i.d. from a
Zipf distribution over a random subset of that country's CSpace cuisine
vocabulary (random order, so Zipf rank is unrelated to prototypicality), which
gives a pool with repeats and a head -- the structure an under-specified
prompt produces -- but with every label known.

The pool is then treated exactly like an ordinary one: it is scored against the
UNDER-specified prompt of its country (what a deployed selector would see),
ranked over all N, and tagged by the same blind tagger.  Two endpoints result:

  * `raw`   -- the tagger's labels.  What the main experiment measures.
  * `audit` -- the prompted labels, with a closed-question VLM check ("does
               this image show {dish}?") as the keep mask, so that an image the
               generator failed to render as its dish does not count as that
               dish.  Source: `known_by_construction`.

`machine_minus_audit` on these pools is the R1 test, now executable on real
images: if the tagger merges labels in the high-scoring set, the tagger's
contrast is more negative than the prompted one.  The pre-registered 5%
equivalence margin on that gap is unchanged.

## What it does not establish

* The prompted label is what the generator was *asked* for.  The verifier
  mask is a second VLM judgement, not a human one; the verifier knows the dish
  name and could say "yes" too readily.  Both unfiltered and verifier-filtered
  audits are reported.
* Known-label pools have a designed label distribution.  They calibrate the
  TAGGER as a function of score; they are not a second estimate of how much a
  real under-specified pool homogenises, and are never pooled with one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["KNOWN_LABEL_TEMPLATE", "known_label_dishes", "known_label_prompt",
           "tagger_accuracy_by_score_quantile"]

KNOWN_LABEL_TEMPLATE = "An image of {dish}, a traditional dish from {country}"


def known_label_prompt(dish, country):
    return KNOWN_LABEL_TEMPLATE.format(dish=dish, country=country)


def known_label_dishes(vocab, n, pool_seed, n_dishes=32, zipf_s=1.1):
    """The n prompted dishes of one known-label pool.  Deterministic in pool_seed."""
    vocab = sorted(set(vocab))
    if len(vocab) < n_dishes:
        raise ValueError(f"vocabulary of {len(vocab)} < n_dishes={n_dishes}")
    rng = np.random.default_rng([int(pool_seed), 0x4B4C])      # "KL"
    subset = rng.choice(len(vocab), size=n_dishes, replace=False)
    w = 1.0 / np.power(np.arange(1, n_dishes + 1), zipf_s)
    w = w / w.sum()
    idx = rng.choice(n_dishes, size=n, p=w)
    return [vocab[subset[i]] for i in idx]


def tagger_accuracy_by_score_quantile(correct, scores, n_bins=4, keep=None,
                                      n_boot=2000, rng=None, groups=None):
    """Tagger accuracy per score quantile, and the top-minus-bottom difference.

    `correct[i]` is whether the tagger's label equals the prompted dish.
    Quantiles are taken WITHIN each pool (pass `scores` already converted to
    within-pool ranks in [0, 1)), so a pool that is uniformly prettier than
    another does not masquerade as a score effect.

    `groups[i]` names the pool image i came from.  With groups the interval
    resamples POOLS (images of one pool share a prompt, a seed stream and a
    tagger context, so they are not independent); without, it resamples images
    and is optimistic.  Round-3 code review, P1.
    """
    rng = rng or np.random.default_rng(0)
    c = np.asarray(correct, dtype=float)
    q = np.asarray(scores, dtype=float)
    g = np.asarray(groups if groups is not None else np.arange(c.size), dtype=object)
    if keep is not None:
        k = np.asarray(keep, dtype=bool)
        c, q, g = c[k], q[k], g[k]
    bins = np.minimum((q * n_bins).astype(int), n_bins - 1)
    acc = [float(c[bins == b].mean()) if np.any(bins == b) else float("nan")
           for b in range(n_bins)]
    ns = [int(np.sum(bins == b)) for b in range(n_bins)]
    top, bot = bins == n_bins - 1, bins == 0
    diff = float(c[top].mean() - c[bot].mean()) if top.any() and bot.any() else float("nan")
    ci = [float("nan")] * 2
    if top.any() and bot.any():
        ids = sorted(set(g.tolist()), key=str)
        # per cluster: (correct in top, n in top, correct in bottom, n in bottom)
        tab = np.array([[c[(g == u) & top].sum(), ((g == u) & top).sum(),
                         c[(g == u) & bot].sum(), ((g == u) & bot).sum()]
                        for u in ids], dtype=float)
        boots = []
        for _ in range(n_boot):
            t = tab[rng.integers(0, len(ids), len(ids))].sum(axis=0)
            if t[1] > 0 and t[3] > 0:
                boots.append(t[0] / t[1] - t[2] / t[3])
        if boots:
            ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    return {"accuracy_by_bin": acc, "n_by_bin": ns, "top_minus_bottom": diff,
            "ci": ci, "n": int(c.size),
            "n_clusters": int(len(set(g.tolist()))),
            "interval": "pool_cluster_bootstrap" if groups is not None else "image_bootstrap"}
