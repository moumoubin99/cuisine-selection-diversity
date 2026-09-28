"""Selection operators: the object under audit, plus the nulls that make the
audit non-trivial.

`Select_k(S, f)` -- draw N i.i.d. samples, score them with f, keep the top k --
is what deployed text-to-image systems do when they show "the best 4 of 16".
Any selector reduces diversity by construction, so the audit's claims are all
stated as *excess* loss relative to a content-blind null selector of MATCHED
SELECTIVITY (same N, same k, same sample pool, no access to image content that
correlates with culture).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "select_topk", "random_selector", "permutation_null", "filesize_selector",
    "random_direction_selector", "mmr_select", "dpp_map_select",
]


def select_topk(scores, k):
    """Indices of the top-k scores, highest first.  Ties broken by index.

    A full stable sort, not `argpartition`.  `argpartition` is unstable, and
    sorting the partition afterwards cannot recover index order at the cutoff:
    for `[0,1,2,1,2,1,0,0,2,1,2,0]` with k=5 it returns `[2,4,10,8,5]` where
    the documented index-order tie-break gives `[2,4,8,10,1]` -- a different
    *membership*, not just a different order.  Pools here are at most a few
    hundred images, so the sort is free and the contract is exact.

    This is the single tie policy for the whole codebase: `dose_response`,
    `mmr_select`, `dpp_map_select` and `permutation_null` all route through
    here or through `np.argsort(..., kind="stable")` on the same convention.
    When ties are common enough to matter, break them with an independent
    random priority *before* calling this, and use the same priorities for the
    null -- do not rely on index order to be arbitrary, because it is not.
    """
    scores = np.asarray(scores, dtype=np.float64)
    k = min(int(k), scores.shape[0])
    return np.argsort(-scores, kind="stable")[:k]


# ---------------------------------------------------------------- null selectors
#
# An earlier version of this design called JPEG file size and a random
# embedding direction "content-blind" nulls.  They are not: both select on
# image content, and a random direction in CLIP space is correlated with
# semantics, so excess loss measured against them is not attributable to
# learned taste.  The PRIMARY null is uniform k-subset rarefaction from the
# same pool, with an exchangeable score-permutation null for calibration.
# The two content-dependent selectors are retained only as secondary
# *reference selectors*, and are reported as such.


def random_selector(n, k, rng):
    """PRIMARY NULL. Uniformly random k of n, drawn from the same pool.

    Equal-size rarefaction.  Repeating this many times gives the exact null
    distribution of the diversity statistic under "a selector that ignores the
    image", at the same N and the same k, so any excess loss is attributable
    to the scorer rather than to subsetting.
    """
    return rng.choice(n, size=min(k, n), replace=False)


def permutation_null(scores, k, rng):
    """PRIMARY CALIBRATION NULL. Top-k after permuting the scores.

    Exchangeable under the hypothesis that scores carry no information about
    cultural content: it preserves the score *distribution* and the top-k
    mechanics exactly, and destroys only the score-to-image pairing.
    """
    scores = np.asarray(scores, dtype=np.float64)
    perm = rng.permutation(scores.shape[0])
    # Keep the same top-k positions of the real score vector, but read off
    # randomly reassigned images.  Permuting the scores and re-ranking would
    # instead leak the index-based tie-break whenever scores are tied.
    return perm[select_topk(scores, k)]


def filesize_selector(sizes, k):
    """SECONDARY REFERENCE (not a null). Select on encoded file size.

    A proxy for visual complexity that no scorer was trained on.  It is NOT
    content-blind -- busy images compress worse, and busyness correlates with
    what is depicted -- so it answers "does a non-learned complexity preference
    also homogenise?", not "is the loss attributable to learned taste?".
    """
    return select_topk(np.asarray(sizes, dtype=np.float64), k)


def random_direction_selector(embeddings, k, rng):
    """SECONDARY REFERENCE (not a null). Top-k along a random unit direction.

    Same functional form as a linear CLIP-family scorer, but the direction is
    unlearned.  It is NOT content-blind: random directions in CLIP space carry
    semantic signal, so a single draw can select strongly on content.  Averaged
    over many draws it bounds "how much homogenisation an arbitrary linear
    probe produces", which is a useful reference and a poor null.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    v = rng.normal(size=E.shape[1])
    v /= np.linalg.norm(v)
    return select_topk(E @ v, k)


# -------------------------------------------------------- diversity-aware selection

def mmr_select(scores, embeddings, k, lam=0.5):
    """Maximal Marginal Relevance: greedily trade score against redundancy.

    score_mmr(i) = lam * s_i - (1 - lam) * max_{j in chosen} sim(i, j)

    Scores are z-normalised first so `lam` is interpretable across scorers.
    Similarity is cosine on the supplied embeddings (use DINOv2 here, never
    the scorer's own CLIP space, or the re-ranker inherits the bias it is
    meant to correct).
    """
    s = np.asarray(scores, dtype=np.float64)
    s = (s - s.mean()) / (s.std() + 1e-12)
    E = np.asarray(embeddings, dtype=np.float64)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    S = E @ E.T
    n = len(s)
    k = min(int(k), n)
    chosen = [int(np.argmax(s))]
    while len(chosen) < k:
        redundancy = S[:, chosen].max(axis=1)
        obj = lam * s - (1.0 - lam) * redundancy
        obj[chosen] = -np.inf
        chosen.append(int(np.argmax(obj)))
    return np.asarray(chosen)


def dpp_map_select(scores, embeddings, k, alpha=1.0):
    """Greedy MAP inference for a quality-weighted DPP (Chen et al., 2018).

    L = diag(q) K diag(q) with q = exp(alpha * z(score) / 2) and K cosine
    similarity.  Greedy maximisation of log det L_Y via incremental Cholesky.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = (s - s.mean()) / (s.std() + 1e-12)
    q = np.exp(alpha * s / 2.0)
    E = np.asarray(embeddings, dtype=np.float64)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    K = E @ E.T
    L = (q[:, None] * K) * q[None, :]
    n = L.shape[0]
    k = min(int(k), n)

    cis = np.zeros((k, n))
    di2 = np.copy(np.diag(L))
    chosen = []
    j = int(np.argmax(di2))
    for it in range(k):
        chosen.append(j)
        if it == k - 1:
            break
        ei = (L[j, :] - cis[:it, :].T @ cis[:it, j]) / np.sqrt(max(di2[j], 1e-12))
        cis[it, :] = ei
        di2 = di2 - ei ** 2
        di2[chosen] = -np.inf
        j = int(np.argmax(di2))
    return np.asarray(chosen)
