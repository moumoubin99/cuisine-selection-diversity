"""Shared helpers for the post hoc A3 reanalyses in this folder (CPU only).

`pool_rows` replays `analyze.frontier`'s per-pool selector runs (same pools,
seeds, k, m, frontier n_null and SELECTOR_GRID) and caches them, so the
analyses here can form pool-level paired contrasts that the stored JSON only
keeps as summaries.  `fixed_design` and `paired_ratio` copy the estimators of
`analyze.frontier` (equal-weight country means of template means; country x
template stratified bootstrap; fixed-country t interval), so their intervals
use the same bootstrap unit as the reported A3 results.
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

from analyze import Cell, _main_prompts  # noqa: E402
from src.frontier import run_selectors  # noqa: E402
from src.config import load_config  # noqa: E402
from src.store import load_vocab_file, tune_prompt_list  # noqa: E402

CONFIG = os.path.join(EXP, "configs", "amend7.yaml")
VOCAB = os.path.join(EXP, "data", "cspace_cuisine_vocab.json")
FRONTIER_N_NULL = 2000  # analyze.py default, as in every stored A3 result


def pool_rows(root, gen, sel, judge, cache_dir, n_null=FRONTIER_N_NULL):
    """(tuning rows, main rows) exactly as `analyze.frontier` builds them."""
    path = os.path.join(cache_dir, f"{gen}__{sel}__{judge}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    cfg = load_config(CONFIG)
    full, _ = load_vocab_file(VOCAB)
    cell = Cell(root, cfg, gen, full, reader="qwen", scorers=(sel, judge))
    k, m = cfg["k"], cfg["m_auth"]

    def rows_for(prompts, seed_base, n_pools):
        rows = []
        for i, p in enumerate(prompts):
            for j in range(n_pools):
                seed = seed_base + 1000 * i + j
                ims, labels = cell.pool(p, seed)
                keep = [l is not None for l in labels]
                E = np.stack([cell.emb[im.image_id] for im in ims])
                s = cell.scorers[sel].score(ims, p.text)
                jv = cell.scorers[judge].score(ims, p.text)
                r = run_selectors([l or "" for l in labels], keep, E, s, jv, k, m,
                                  n_null, seed)
                r["_country"], r["_template"] = p.country, p.template
                rows.append(r)
        return rows

    tu = cfg["tune"]
    tune = rows_for(tune_prompt_list(cfg), tu["seed_base"], tu["pools_per_country"])
    main = rows_for(_main_prompts(cfg), cfg["seed"], cfg["n_pools"])
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump((tune, main), fh)
    return tune, main


def _by_country(main, values_of):
    """{country: [per-template arrays of finite per-pool values]}; None if thin."""
    cells = sorted({(r["_country"], r["_template"]) for r in main})
    bc = {c: [] for c in cells}
    for r in main:
        v = values_of(r)
        if np.all(np.isfinite(v)):
            bc[(r["_country"], r["_template"])].append(v)
    if any(len(v) < 2 for v in bc.values()):
        return None
    out = {}
    for (c, t), v in bc.items():
        out.setdefault(c, []).append(np.asarray(v, dtype=float))
    return out


def fixed_design(main, values_of, seed, n_boot=2000):
    """Copy of `analyze.frontier.fixed_design` (scalar per-pool values)."""
    from src.analysis import _fixed_country_moments, _student_t_ppf
    by = _by_country(main, lambda r: np.float64(values_of(r)))
    if by is None:
        return {"non_estimable": True, "mean": float("nan"), "ci": [float("nan")] * 2}
    by_country = {c: {i: v for i, v in enumerate(vs)} for c, vs in by.items()}
    theta, se, df, _ = _fixed_country_moments(by_country)
    rng = np.random.default_rng(seed)
    boots = [np.mean([np.mean([v[rng.integers(0, v.size, v.size)].mean() for v in vs])
                      for vs in by.values()]) for _ in range(n_boot)]
    out = {"non_estimable": False,
           "mean": float(np.mean([np.mean([v.mean() for v in vs]) for vs in by.values()])),
           "bootstrap_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
    if theta is None:
        out.update(ci=[float("nan")] * 2)
        return out
    h = _student_t_ppf(0.975, df) * se
    out.update(ci=[float(theta - h), float(theta + h)])
    return out


def cmean(by, rows_idx=None):
    """Equal-weight country mean of template means, vector-valued."""
    return np.mean([np.mean([v.mean(axis=0) for v in vs], axis=0) for vs in by.values()],
                   axis=0)


def boot_stat(main, values_of, stat, seed, n_boot=2000):
    """Point value and 95% percentile interval of `stat(country-mean vector)`,
    resampling pools within each country x template cell (A3's unit), with a
    pool's values kept together (paired).  NaN statistics are dropped and the
    interval is withheld if more than 10% of replicates are NaN."""
    by = _by_country(main, lambda r: np.asarray(values_of(r), dtype=float))
    if by is None:
        return {"value": float("nan"), "ci": [float("nan")] * 2, "non_estimable": True}
    point = stat(cmean(by))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        rb = {c: [v[rng.integers(0, len(v), len(v))] for v in vs] for c, vs in by.items()}
        boots.append(stat(cmean(rb)))
    boots = np.asarray(boots, dtype=float)
    ok = np.isfinite(boots)
    ci = ([float(np.quantile(boots[ok], .025)), float(np.quantile(boots[ok], .975))]
          if ok.mean() >= 0.9 else [float("nan")] * 2)
    return {"value": float(point), "ci": ci, "boot_finite_frac": float(ok.mean()),
            "non_estimable": False}
