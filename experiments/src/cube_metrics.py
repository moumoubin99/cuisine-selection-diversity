"""Cultural-diversity metrics, ported faithfully from CUBE's released
`cultural_diversity.ipynb` (google-deepmind/cube) and extended with the
controls this study requires.

The released notebook computes a *normalized Vendi score over discrete
VLM-assigned labels*, batched in groups of 8, with a hierarchical
similarity kernel.  Two modes:

  _global=True   samples are (continent, country, artifact) and the released
                 kernel `1*(a[0]==b[0]) + 0*(...) + 0*(...)` compares
                 CONTINENT only.
  _global=False  samples are (item, None, None) -- `item` is the raw label
                 string -- so the same kernel is an EXACT MATCH on the
                 artifact label.

This study uses country-conditioned CUBE-1K prompts, i.e. the
within-culture mode (`_global=False`), where the kernel is artifact-level.
The continent-only kernel is retained for the global-prompt comparison and
is *not* used for the headline numbers, because it cannot resolve the
artifact-level collapse we are measuring.

No dependency on the `vendi_score` package: the score is reimplemented
here (5 lines) so the metric is auditable and unit-testable on CPU.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "vendi_score_from_kernel",
    "vendi_score",
    "GLOBAL_KERNEL",
    "ARTIFACT_KERNEL",
    "cultural_diversity",
    "simpson_diversity",
    "coverage",
]


def vendi_score_from_kernel(K: np.ndarray) -> float:
    """Vendi score of a positive semi-definite similarity matrix with unit diagonal.

    VS(K) = exp(H(lambda)) where lambda are the eigenvalues of K/n and H is
    the Shannon entropy in nats.  Equals the effective number of distinct
    items: n for n mutually dissimilar items, 1 for n identical items.
    """
    n = K.shape[0]
    if n == 0:
        return 0.0
    w = np.linalg.eigvalsh(np.asarray(K, dtype=np.float64) / n)
    w = np.clip(w, 0.0, None)
    w = w[w > 1e-12]
    if w.size == 0:
        return 0.0
    return float(np.exp(-np.sum(w * np.log(w))))


def vendi_score(samples, similarity_function) -> float:
    """Vendi score of a list of samples under an arbitrary similarity function."""
    n = len(samples)
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        K[i, i] = similarity_function(samples[i], samples[i])
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = similarity_function(samples[i], samples[j])
    return vendi_score_from_kernel(K)


# The released CUBE kernel, verbatim in effect.
GLOBAL_KERNEL = lambda a, b: 1 * int(a[0] == b[0]) + 0 * int(a[1] == b[1]) + 0 * int(a[2] == b[2])
ARTIFACT_KERNEL = GLOBAL_KERNEL  # same callable; the *tuple packing* is what differs


def cultural_diversity(labels, similarity_function=GLOBAL_KERNEL, _global=False, batch_size=8):
    """Faithful port of CUBE's `calculate_cultural_diversity`.

    Args:
      labels: list of annotations.  If `_global`, each must be a mapping with
        'continent', 'country', 'artifact'; otherwise each is a raw label.
      similarity_function: kernel over the packed 3-tuples.
      _global: True for global prompts, False for within-culture prompts.
      batch_size: the notebook's batching, 8.

    Returns:
      Mean over batches of (Vendi score / batch_size).  Note this normalises
      by `batch_size`, not by the actual chunk length, exactly as released --
      so a trailing partial chunk is penalised.  We keep that behaviour for
      comparability and additionally expose `drop_last` below.
    """
    if len(labels) < 32:
        labels = labels[:24]

    chunks = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]
    all_vendi = []
    for chunk in chunks:
        if _global:
            samples = [(it["continent"], it["country"], it["artifact"]) for it in chunk]
        else:
            samples = [(it, None, None) for it in chunk]
        if len(samples) == 0:
            continue
        all_vendi.append(vendi_score(samples, similarity_function) / batch_size)
    if not all_vendi:
        return 0.0
    return float(np.mean(all_vendi))


def cultural_diversity_strict(labels, batch_size=8, drop_last=True, _global=False,
                              similarity_function=GLOBAL_KERNEL):
    """Same metric without the released version's two comparability hazards.

    The released code (a) truncates to 24 labels whenever fewer than 32 are
    supplied and (b) divides every chunk's Vendi score by `batch_size` even
    when the chunk is shorter.  Both are fine when every condition is
    evaluated at an identical, large label count, and both silently bias
    comparisons when they are not.  Since this study compares selected sets
    of deliberately *different* sizes (top-k for several k), the headline
    numbers use this variant, and the released variant is reported alongside
    for continuity with the CUBE paper.

    Two properties of this function are inherited from the released code and
    are NOT fixed here, because the point of the port is to reproduce it:

      * it is **order-dependent**.  Consecutive chunking means the same
        multiset can score differently depending on arrangement:
        ``['a']*8 + ['b']*8`` scores 0.125 while ``['a','b']*8`` scores 0.25.
      * it is **not comparable across set sizes** (see the note below).

    Both are why no headline number calls this directly.  Every reported
    quantity goes through `analysis.rarefy_to_common_count`, which evaluates a
    single fixed-size batch drawn at random -- order-invariant in expectation,
    and at one common size.
    """
    labels = list(labels)
    if not labels:
        return 0.0
    if batch_size > len(labels):
        # A selected set smaller than the batch size (e.g. top-k with k < 8)
        # has no full chunk.  Dropping the partial chunk would silently return
        # 0.0 for exactly the most-selective conditions this study cares about,
        # so we evaluate the set as a single batch instead.
        #
        # RETRACTION: an earlier version of this comment claimed the result is
        # "comparable across n".  It is NOT.  The Vendi score is divided by the
        # set size, so eight identical labels score 0.125 and four identical
        # labels score 0.25 -- the metric RISES as the set shrinks.  Nothing in
        # this function makes different `n` comparable.  Comparability is
        # supplied only by evaluating every condition at one common size, which
        # is what `analysis.rarefy_to_common_count` does and what every headline
        # number in this study goes through.
        batch_size = len(labels)
    chunks = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]
    if drop_last and len(chunks) > 1 and len(chunks[-1]) < batch_size:
        chunks = chunks[:-1]
    out = []
    for chunk in chunks:
        if not chunk:
            continue
        if _global:
            samples = [(it["continent"], it["country"], it["artifact"]) for it in chunk]
        else:
            samples = [(it, None, None) for it in chunk]
        out.append(vendi_score(samples, similarity_function) / len(samples))
    return float(np.mean(out)) if out else 0.0


def simpson_diversity(labels) -> float:
    """1 - sum p_i^2 over label frequencies.  Batch-free sanity companion."""
    labels = list(labels)
    if not labels:
        return 0.0
    _, counts = np.unique(np.asarray(labels, dtype=object), return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def coverage(labels, reference_vocabulary) -> float:
    """Fraction of the reference artifact vocabulary that appears at least once."""
    ref = set(reference_vocabulary)
    if not ref:
        return 0.0
    return len(set(labels) & ref) / len(ref)
