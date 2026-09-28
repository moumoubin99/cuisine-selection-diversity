"""Concept-neutral scoring: remove a cultural-concept subspace from the signal
an aesthetic/preference scorer uses, without retraining it.

## What is actually being claimed, stated precisely

An early version of this proposal said "project the cultural subspace out of
the scorer's *weight direction* `w`, which is different from projecting the
image embedding."  That distinction is false and must not appear in the paper.
For an orthogonal projector `P` (symmetric, idempotent):

        <w - Pw, E>  =  <(I - P)w, E>  =  <w, (I - P)E>

so projecting the weights and projecting the embedding are the *same function*.
`test_weight_and_embedding_projection_are_identical` pins this.

The second correction is that "CLIP-family scorers are linear in the image
embedding" is only true for some of them:

  * PickScore, HPSv2       -- score is a (scaled) cosine similarity between the
                              image and text embeddings, hence linear in the
                              image embedding at a fixed prompt.  A weight
                              direction `w` genuinely exists: w = s * t / |E|.
  * LAION-Aesthetics v2    -- an MLP head over CLIP ViT-L/14 embeddings.  There
                              is NO single weight direction.  (v1 was linear;
                              verify which checkpoint is loaded.)
  * ImageReward            -- a BLIP-based non-linear head.  No weight direction.

So the method is stated, correctly and more generally, as a projection of the
scorer's *input representation*:

        f'(x) = f( (I - P_C) E(x) )

which coincides with the weight-space form whenever f is linear, is well
defined when f is not, and is what the code below implements.  The paper's
contribution is therefore NOT the linear algebra -- projection debiasing is
established prior work (PRISM, SEM, FairImagen, LightFair, Latent Directions).
It is (i) the subspace, estimated text-only from CUBE-CSpace grounding
artifacts with no images, no labels and no training; (ii) the pipeline stage,
post-hoc selection over an i.i.d. sample set rather than generation; and
(iii) the quantitative mechanism claim in `predict_country_loss` below, which
is what a reviewer can falsify.

## The subspace

`C` is spanned by text embeddings of CUBE-CSpace artifacts.  Two variants:

  `global`      one subspace for all countries, from the top principal
                directions of all artifact embeddings after centring.  Removes
                "cultural specificity" as a whole.
  `differential` the directions along which *frequent* artifacts differ from
                *rare* ones within each country.  Removes the scorer's
                preference for the prototypical without removing the concept
                axis itself.  This is the variant the mechanism predicts.

`differential` is the one to report: `global` is a stronger intervention that
also deletes legitimate signal, and serves as the ablation that shows the
difference matters.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "orthonormal_basis", "projector", "neutralize_embeddings",
    "global_concept_subspace", "differential_concept_subspace",
    "predict_country_loss",
]


def orthonormal_basis(vectors, rank=None, tol=1e-8):
    """Orthonormal basis for the span of `vectors` (rows), via thin SVD."""
    V = np.asarray(vectors, dtype=np.float64)
    if V.ndim == 1:
        V = V[None, :]
    U, s, Vt = np.linalg.svd(V, full_matrices=False)
    keep = s > tol * (s[0] if s.size else 1.0)
    B = Vt[keep]
    if rank is not None:
        B = B[:int(rank)]
    return B


def projector(basis):
    """Orthogonal projector onto the row space of an orthonormal `basis`."""
    B = np.asarray(basis, dtype=np.float64)
    if B.size == 0:
        return np.zeros((0, 0))
    return B.T @ B


def neutralize_embeddings(E, basis, strength=1.0):
    """Return E with its component in span(basis) removed.

    `strength` in [0, 1] interpolates between the original embedding (0) and
    full removal (1), so the quality/diversity frontier can be traced as a
    continuous curve rather than a single point.
    """
    E = np.asarray(E, dtype=np.float64)
    B = np.asarray(basis, dtype=np.float64)
    if B.size == 0:
        return E.copy()
    return E - strength * (E @ B.T) @ B


def global_concept_subspace(artifact_text_embeddings, rank=8):
    """Top-`rank` principal directions of the centred artifact text embeddings."""
    A = np.asarray(artifact_text_embeddings, dtype=np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[:rank]


def differential_concept_subspace(artifact_text_embeddings, frequencies, rank=4,
                                  quantile=0.5):
    """Directions separating frequent artifacts from rare ones, pooled across countries.

    For each country the difference-in-means between the above-median-frequency
    artifacts and the below-median ones is one prototypicality direction; the
    subspace is the top-`rank` principal directions of the stacked differences.
    This is a text-only estimate: no images, no human labels, no training.

    Args:
      artifact_text_embeddings: mapping country -> (n_artifacts, d) array.
      frequencies: mapping country -> (n_artifacts,) array of CSpace frequencies.
    """
    diffs = []
    for country, A in artifact_text_embeddings.items():
        A = np.asarray(A, dtype=np.float64)
        f = np.asarray(frequencies[country], dtype=np.float64)
        if A.shape[0] < 2:
            continue
        thresh = np.quantile(f, quantile)
        hi, lo = A[f > thresh], A[f <= thresh]
        if hi.shape[0] == 0 or lo.shape[0] == 0:
            continue
        d = hi.mean(axis=0) - lo.mean(axis=0)
        n = np.linalg.norm(d)
        if n > 1e-12:
            diffs.append(d / n)
    if not diffs:
        return np.zeros((0, next(iter(artifact_text_embeddings.values())).shape[1]))
    D = np.stack(diffs)
    # NOT centred.  The per-country difference vectors are expected to share a
    # common direction -- that shared prototypicality axis IS the signal -- so
    # centring D would subtract exactly the thing we are estimating.  (An
    # earlier version centred here and recovered noise; pinned by
    # test_differential_subspace_recovers_the_prototypicality_axis.)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:min(rank, Vt.shape[0])]


def predict_country_loss(w, country_artifact_embeddings, rank=16, var_target=None):
    """The falsifiable mechanism prediction.

    If a linear scorer's diversity damage to a country comes from `w` having a
    large component inside the directions that separate that country's
    artifacts from one another, then the *predicted* per-country diversity loss
    is the norm of w projected onto that country's artifact subspace:

        pred(c) = || P_{A_c} w ||  /  ||w||

    The paper's stated survival criterion is Pearson r > 0.7 between this and
    the *observed* per-country diversity loss under Select_k.  Note the
    prediction is a *taste* prediction: render-fidelity preference (arXiv
    2608.23593) does not predict it, which is what makes the correlation the
    arbiter between the two explanations.

    Args:
      w: (d,) scoring direction, defined only for linear scorers.
      country_artifact_embeddings: mapping country -> (n_artifacts, d) array of
        that country's artifact TEXT embeddings, centred within country.
      rank: fixed truncation rank, the SAME for every country.  Required --
        see the note in the body.  `var_target` (e.g. 0.9) selects the rank by
        explained variance instead, but then the rank differs across countries
        and the vocabulary-size confound returns, so it is only for ablation.

    Returns:
      mapping country -> predicted loss in [0, 1].
    """
    w = np.asarray(w, dtype=np.float64)
    wn = np.linalg.norm(w)
    out = {}
    for country, A in country_artifact_embeddings.items():
        A = np.asarray(A, dtype=np.float64)
        A = A - A.mean(axis=0, keepdims=True)
        # Truncate to a FIXED rank, identical for every country.  Without this
        # the prediction saturates and becomes meaningless: CLIP ViT-L/14 text
        # embeddings are 768-dimensional, and the cuisine vocabularies of
        # India (1413), France (880) and Italy (876) each exceed that, so their
        # raw artifact span IS the whole space and ||P_A w|| / ||w|| == 1 by
        # construction while Nigeria (196) and Turkey (384) score below 1 --
        # turning the "mechanism" into a disguised readout of vocabulary size.
        # Pinned by test_predict_country_loss_does_not_saturate_on_large_vocabularies.
        _, s, Vt = np.linalg.svd(A, full_matrices=False)
        # drop numerically-null directions before truncating, so an over-large
        # `rank` cannot pull in arbitrary null-space directions
        nz = int(np.sum(s > 1e-8 * (s[0] if s.size else 1.0)))
        if var_target is not None and nz:
            cum = np.cumsum(s[:nz] ** 2) / np.sum(s[:nz] ** 2)
            r = int(np.searchsorted(cum, var_target) + 1)
        else:
            r = int(rank)
        B = Vt[:min(r, nz)]
        out[country] = float(np.linalg.norm(B @ w) / (wn + 1e-12)) if B.size else 0.0
    return out
