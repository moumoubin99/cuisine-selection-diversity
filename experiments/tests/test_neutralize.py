import numpy as np, pytest
from src.neutralize import (orthonormal_basis, projector, neutralize_embeddings,
                            global_concept_subspace, differential_concept_subspace,
                            predict_country_loss)

def test_weight_and_embedding_projection_are_identical():
    """The correction that killed the original novelty framing: for an
    orthogonal projector, projecting the weights equals projecting the input."""
    rng = np.random.default_rng(0)
    d = 32
    B = orthonormal_basis(rng.normal(size=(5, d)))
    P = projector(B)
    w = rng.normal(size=d)
    E = rng.normal(size=(64, d))
    via_weights = E @ (w - P @ w)
    via_embeddings = neutralize_embeddings(E, B) @ w
    assert np.allclose(via_weights, via_embeddings, atol=1e-10)

def test_projector_is_symmetric_idempotent_and_right_rank():
    rng = np.random.default_rng(1)
    B = orthonormal_basis(rng.normal(size=(6, 20)))
    P = projector(B)
    assert np.allclose(P, P.T, atol=1e-10)
    assert np.allclose(P @ P, P, atol=1e-10)
    assert np.linalg.matrix_rank(P, tol=1e-8) == 6

def test_neutralize_removes_exactly_the_subspace():
    rng = np.random.default_rng(2)
    B = orthonormal_basis(rng.normal(size=(3, 16)))
    E = rng.normal(size=(50, 16))
    En = neutralize_embeddings(E, B)
    assert np.allclose(En @ B.T, 0.0, atol=1e-10)
    # the orthogonal complement is untouched
    Bc = orthonormal_basis(np.eye(16) - projector(B))
    assert np.allclose(En @ Bc.T, E @ Bc.T, atol=1e-10)

def test_strength_interpolates():
    rng = np.random.default_rng(3)
    B = orthonormal_basis(rng.normal(size=(2, 12)))
    E = rng.normal(size=(20, 12))
    assert np.allclose(neutralize_embeddings(E, B, 0.0), E)
    half = neutralize_embeddings(E, B, 0.5)
    assert np.allclose(half, 0.5 * (E + neutralize_embeddings(E, B, 1.0)), atol=1e-10)

def test_empty_basis_is_a_no_op():
    E = np.random.default_rng(4).normal(size=(5, 8))
    assert np.allclose(neutralize_embeddings(E, np.zeros((0, 8))), E)

def test_global_subspace_recovers_a_planted_plane():
    rng = np.random.default_rng(5)
    d = 24
    plane = orthonormal_basis(rng.normal(size=(2, d)))
    coeffs = rng.normal(size=(200, 2)) * 10.0
    A = coeffs @ plane + 0.01 * rng.normal(size=(200, d))
    B = global_concept_subspace(A, rank=2)
    # recovered plane spans (nearly) the same space
    assert np.allclose(np.abs(np.linalg.svd(B @ plane.T, compute_uv=False)), 1.0, atol=1e-2)

def test_differential_subspace_recovers_the_prototypicality_axis():
    """Plant one direction along which frequent artifacts sit higher than rare
    ones, in every country.  The estimator must find it from text alone."""
    rng = np.random.default_rng(6)
    d = 32
    proto = rng.normal(size=d); proto /= np.linalg.norm(proto)
    emb, freq = {}, {}
    for c in range(8):
        n = 20
        f = rng.random(n)
        base = rng.normal(size=(n, d))
        emb[f"c{c}"] = base + 6.0 * (f > 0.5)[:, None] * proto
        freq[f"c{c}"] = f
    B = differential_concept_subspace(emb, freq, rank=1)
    assert abs(float(B[0] @ proto)) > 0.9

def test_predict_country_loss_is_high_when_w_lies_in_the_artifact_span():
    rng = np.random.default_rng(7)
    d = 20
    A_in = rng.normal(size=(6, d))
    B_in = orthonormal_basis(A_in - A_in.mean(0, keepdims=True))
    w_inside = B_in.sum(axis=0)
    # a direction orthogonal to that span
    w_outside = rng.normal(size=d)
    w_outside -= projector(B_in) @ w_outside
    pred = predict_country_loss(w_inside, {"x": A_in})
    assert pred["x"] > 0.99
    pred2 = predict_country_loss(w_outside, {"x": A_in})
    assert pred2["x"] < 1e-8

def test_predict_country_loss_is_in_unit_interval():
    rng = np.random.default_rng(8)
    d = 16
    emb = {f"c{i}": rng.normal(size=(5, d)) for i in range(6)}
    w = rng.normal(size=d)
    for v in predict_country_loss(w, emb).values():
        assert 0.0 <= v <= 1.0 + 1e-9


def test_predict_country_loss_does_not_saturate_on_large_vocabularies():
    """Regression: with vocab >= embedding dim the untruncated span is the whole
    space and every country scores 1.0, so the 'mechanism' would just be a
    readout of vocabulary size.  Fixed rank must break that tie."""
    rng = np.random.default_rng(11)
    d = 64
    emb = {"small": rng.normal(size=(20, d)), "large": rng.normal(size=(400, d))}
    w = rng.normal(size=d)
    saturated = predict_country_loss(w, emb, rank=d)          # no truncation
    assert saturated["large"] == pytest.approx(1.0, abs=1e-6)
    fixed = predict_country_loss(w, emb, rank=8)
    assert fixed["large"] < 0.9 and fixed["small"] < 0.9


def test_predict_country_loss_rank_is_uniform_across_countries():
    rng = np.random.default_rng(12)
    d = 64
    emb = {f"c{i}": rng.normal(size=(50 + 90 * i, d)) for i in range(5)}
    w = rng.normal(size=d)
    vals = np.array(list(predict_country_loss(w, emb, rank=10).values()))
    sizes = np.array([len(v) for v in emb.values()], dtype=float)
    # with a fixed rank the prediction must not simply track vocabulary size
    assert abs(np.corrcoef(vals, sizes)[0, 1]) < 0.9
