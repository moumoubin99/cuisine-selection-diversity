"""A3: the quality/diversity frontier, and the selector this study proposes.

## The proposed selector: stratified top-k (STK)

Top-k takes the k highest-scoring images wherever they are.  If the scorer
prefers one kind of image, those k all come from one region of the pool.
STK first partitions the pool into `c` visual strata (k-means on DINOv2
features, which the scorer never sees), gives each stratum a number of slots
proportional to its size (largest remainder), and takes the highest-scoring
images *within* each stratum.

  * c = 1 is top-k.  As c grows, the selected set's composition approaches the
    pool's own composition -- the thing random selection preserves -- while
    each slot is still filled by the best image available in its stratum.
  * It needs no cultural labels, so it is not tuned to the metric it is judged
    on: diversity is measured with the VLM's dish labels, and the strata come
    from a self-supervised image model.
  * Unlike MMR / DPP it does not push toward the pool's outliers.  It
    preserves proportions rather than maximising spread, which is the property
    "do not make the culture look more homogeneous than the generator already
    does" actually asks for.

MMR and a quality-weighted DPP, both on the same DINOv2 features, are the
baselines.  Every hyper-parameter is chosen on the tuning pools and then
frozen (`choose_on_tuning`).

## How a point on the frontier is measured

  diversity   log contrast of the selected set against random k-subsets of the
              same pool, on the tagger's labels, rarefied to m -- the raw
              endpoint of `pipeline._endpoint`, computed once per pool and
              shared by every selector.  Plus a label-free check: log Vendi of
              the DINOv2 cosine kernel, selected minus random.
  quality     the HELD-OUT scorer (ImageReward when selecting by PickScore and
              the reverse), as a within-pool standardised gain over random:
              (mean J(selected) - mean J(pool)) / sd J(pool).  The selecting
              scorer's own gain is reported too, but it cannot be the judge: a
              selector is always best under the score it maximises.

No human preference study stands behind "quality" here.  Claims are about the
two automated preference models, not about people.
"""

from __future__ import annotations

import numpy as np

from .analysis import MAX_NULL_INSUFFICIENT, rarefy_to_common_count
from .cube_metrics import vendi_score_from_kernel
from .selectors import dpp_map_select, mmr_select, random_selector, select_topk

__all__ = ["kmeans_cosine", "stratified_topk", "PoolNull", "embedding_log_vendi",
           "evaluate_selection", "run_selectors", "SELECTOR_GRID",
           "choose_on_tuning"]

SELECTOR_GRID = {
    "topk": [None],
    "stk": [2, 3, 4, 6, 8, 12, 16],
    "mmr": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    "dpp": [4.0, 2.0, 1.0, 0.5, 0.25],
}


def _unit(E):
    E = np.asarray(E, dtype=np.float64)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def kmeans_cosine(E, c, seed=0, iters=50):
    """Spherical k-means with k-means++ seeding.  Returns labels in [0, c)."""
    X = _unit(E)
    n = X.shape[0]
    c = int(min(c, n))
    rng = np.random.default_rng(seed)
    centers = [X[rng.integers(n)]]
    for _ in range(1, c):
        d = 1.0 - np.max(X @ np.stack(centers).T, axis=1)
        d = np.clip(d, 0, None)
        p = d / d.sum() if d.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=p)])
    C = np.stack(centers)
    lab = np.argmax(X @ C.T, axis=1)
    for _ in range(iters):
        newC = np.stack([X[lab == j].mean(axis=0) if np.any(lab == j) else C[j]
                         for j in range(c)])
        newC = _unit(newC)
        new = np.argmax(X @ newC.T, axis=1)
        C = newC
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def stratified_topk(scores, E, k, c, seed=0):
    """STK: proportional slots per DINOv2 stratum, best-scoring within each."""
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    k = int(min(k, n))
    if c is None or c <= 1:
        return select_topk(s, k)
    lab = kmeans_cosine(E, c, seed=seed)
    ids = np.unique(lab)
    sizes = np.array([np.sum(lab == j) for j in ids], dtype=float)
    quota = k * sizes / sizes.sum()
    base = np.floor(quota).astype(int)
    rem = k - base.sum()
    # Largest remainder; ties to the larger stratum, then the lower id.
    order = sorted(range(len(ids)), key=lambda i: (-(quota[i] - base[i]), -sizes[i], i))
    for i in order[:rem]:
        base[i] += 1
    chosen = []
    for j, q in zip(ids, base):
        members = np.flatnonzero(lab == j)
        top = members[np.argsort(-s[members], kind="stable")[:q]]
        chosen.extend(top.tolist())
    return np.asarray(chosen)


def embedding_log_vendi(E):
    X = _unit(E)
    return float(np.log(vendi_score_from_kernel(X @ X.T)))


class PoolNull:
    """The random-k reference for one pool, computed once and shared.

    Mirrors `pipeline._endpoint`: selection is over all N, the `keep` filter is
    applied to whatever was chosen, and the null draws that fall below the
    common count m are counted, not silently dropped.
    """

    def __init__(self, labels, keep, E, k, m, n_null, rng, n_emb=200):
        self.labels, self.keep, self.k, self.m = list(labels), list(keep), k, m
        n = len(self.labels)
        logd, bad = [], 0
        for _ in range(n_null):
            idx = random_selector(n, k, rng)
            nl = [self.labels[i] for i in idx if self.keep[i]]
            v = rarefy_to_common_count(nl, m, rng=rng)
            if v["non_estimable"]:
                bad += 1
            else:
                logd.append(np.log(v["value"]))
        self.insufficient = bad / float(n_null)
        self.mean_logd = float(np.mean(logd)) if logd else float("nan")
        self.E = None if E is None else np.asarray(E)
        if E is not None:
            self.mean_emb = float(np.mean([embedding_log_vendi(self.E[random_selector(n, k, rng)])
                                           for _ in range(n_emb)]))

    def delta(self, idx, rng):
        sel = [self.labels[i] for i in idx if self.keep[i]]
        v = rarefy_to_common_count(sel, self.m, rng=rng)
        if v["non_estimable"] or self.insufficient > MAX_NULL_INSUFFICIENT:
            return float("nan")
        return float(np.log(v["value"]) - self.mean_logd)

    def emb_delta(self, idx):
        return embedding_log_vendi(self.E[np.asarray(idx)]) - self.mean_emb


def _std_gain(v, idx):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return float((v[np.asarray(idx)].mean() - v.mean()) / sd) if sd > 0 else 0.0


def _raw_gain(v, idx):
    v = np.asarray(v, dtype=np.float64)
    return float(v[np.asarray(idx)].mean() - v.mean())


def evaluate_selection(idx, null, sel_scores, judge_scores, rng):
    # `gain_*` is standardised by the pool's own score SD (the registered A3
    # quality unit); `raw_gain_judge` is the same gain in the judge's native
    # units, reported beside it (experiment audit, 2026-09-24).
    return {"delta": null.delta(idx, rng), "emb_delta": null.emb_delta(idx),
            "gain_selector": _std_gain(sel_scores, idx),
            "gain_judge": _std_gain(judge_scores, idx),
            "raw_gain_judge": _raw_gain(judge_scores, idx)}


def _select(name, param, scores, E, k, seed):
    if name == "topk":
        return select_topk(scores, k)
    if name == "stk":
        return stratified_topk(scores, E, k, param, seed=seed)
    if name == "mmr":
        return mmr_select(scores, E, k, lam=param)
    if name == "dpp":
        return dpp_map_select(scores, E, k, alpha=param)
    raise ValueError(name)


def run_selectors(labels, keep, E, sel_scores, judge_scores, k, m, n_null, seed,
                  grid=None):
    """Every selector configuration on one pool.  Returns {(name, param): metrics}."""
    rng = np.random.default_rng(seed)
    null = PoolNull(labels, keep, E, k, m, n_null, rng)
    out = {("random", None): {"delta": 0.0, "emb_delta": 0.0,
                              "gain_selector": 0.0, "gain_judge": 0.0}}
    for name, params in (grid or SELECTOR_GRID).items():
        for p in params:
            idx = _select(name, p, sel_scores, E, k, seed)
            out[(name, p)] = evaluate_selection(idx, null, sel_scores, judge_scores, rng)
    out["_null_insufficient"] = null.insufficient
    return out


def choose_on_tuning(tuning_rows, margin_log=np.log(0.9)):
    """Per selector family, the frozen hyper-parameter, chosen on TUNING pools only.

    Rule (fixed before any evaluation pool was analysed): the configuration
    with the highest held-out-judge gain among those whose mean tuning-pool
    diversity contrast is inside the pre-registered margin (delta >= log 0.9).
    If none is inside, the one with the least negative delta.

    Means are EQUAL-WEIGHT COUNTRY means when rows carry `_country`, and a
    configuration is eligible only if every tuning country has an estimable
    delta for it.  Round-3 code review, P1: `nanmean` over pools tuned each
    configuration on whichever countries happened to survive, so two
    configurations could be compared on different country sets.
    """
    countries = sorted({r.get("_country") for r in tuning_rows})

    def country_mean(name, p, metric):
        per = []
        for c in countries:
            v = [r[(name, p)][metric] for r in tuning_rows
                 if r.get("_country") == c and np.isfinite(r[(name, p)][metric])]
            per.append(np.mean(v) if v else np.nan)
        per = np.asarray(per, dtype=float)
        return (float(per.mean()) if np.all(np.isfinite(per)) else float("nan"),
                [c for c, x in zip(countries, per) if not np.isfinite(x)])

    fams = {}
    for key in tuning_rows[0]:
        if not isinstance(key, tuple) or key[0] == "random":
            continue
        fams.setdefault(key[0], []).append(key[1])
    choice = {}
    for name, params in fams.items():
        best, excluded = None, {}
        for p in params:
            d, miss_d = country_mean(name, p, "delta")
            g, miss_g = country_mean(name, p, "gain_judge")
            if miss_d or miss_g:
                excluded[str(p)] = sorted(set(miss_d) | set(miss_g))
                continue
            inside = d >= margin_log
            key = (inside, g if inside else d)
            if best is None or key > best[0]:
                best = (key, p, d, g)
        if best is None:
            choice[name] = {"param": None, "tuning_delta": float("nan"),
                            "tuning_gain_judge": float("nan"), "inside_margin": False,
                            "non_estimable": True, "excluded_incomplete": excluded,
                            "reason": "no configuration is estimable in every "
                                      "tuning country"}
            continue
        choice[name] = {"param": best[1], "tuning_delta": float(best[2]),
                        "tuning_gain_judge": float(best[3]),
                        "inside_margin": bool(best[0][0]),
                        "excluded_incomplete": excluded}
    return choice
