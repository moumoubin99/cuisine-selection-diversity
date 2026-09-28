"""E13 (post hoc, descriptive): four more re-selector families next to STK and DPP.

Reviewers of diversity-aware selection usually ask for the standard subset
selection baselines, not only MMR and DPP.  Four families are added, all on the
same DINOv2 features and z-scored selector scores as the registered ones:

  thr     random k among the top q fraction of the pool (quality filter, then
          random), q in THR_Q; q = k/n is top-k and q = 1 is random-k.
  fl      greedy facility location: coverage of the pool (mean over pool images
          of the best cosine similarity, mapped to [0, 1]) + beta * mean z-score.
  vendi   greedy log Vendi(S) (DINOv2 cosine kernel) + gamma * mean z-score.
  kcenter greedy k-center: start at the top-scored image, then add the image
          maximising (cosine distance to the chosen set) + eta * z-score.

Every family is tuned on the tuning pools with the registered rule
(`choose_on_tuning`) and evaluated on the 64 main pools, exactly as A3 treats
STK, MMR and DPP.  The registered grid runs first on the same rng stream, so the
STK / MMR / DPP rows reproduce the stored A3 rows (checked below).  Quality is
the held-out scorer; label diversity uses the pre-registered reader (Qwen).

    python extra/e13_more_selectors.py --root <store> --cache <dir> --jobs 8
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _rows import CONFIG, EXP, FRONTIER_N_NULL, VOCAB, boot_stat, fixed_design  # noqa: E402
from build_rows import DIRS, GENS  # noqa: E402
import src.frontier as F  # noqa: E402
from src.cube_metrics import vendi_score_from_kernel  # noqa: E402

THR_Q = [0.375, 0.5, 0.625, 0.75]
FL_BETA = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05]
VENDI_GAMMA = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
KC_ETA = [1.0, 0.5, 0.25, 0.1, 0.05]
NEW_GRID = {"thr": THR_Q, "fl": FL_BETA, "vendi": VENDI_GAMMA, "kcenter": KC_ETA}
GRID = {**F.SELECTOR_GRID, **NEW_GRID}
NEW = tuple(NEW_GRID)
REF = ("topk", "stk", "mmr", "dpp")


def _z(s):
    s = np.asarray(s, dtype=np.float64)
    return (s - s.mean()) / (s.std() + 1e-12)


def _cos(E):
    X = F._unit(np.asarray(E, dtype=np.float64))
    return X @ X.T


def thr_select(scores, k, q, seed):
    n = len(scores)
    m = max(k, int(round(q * n)))
    top = np.argsort(-np.asarray(scores), kind="stable")[:m]
    rng = np.random.default_rng(seed + 7_000_003)
    return np.sort(rng.choice(top, size=k, replace=False))


def fl_select(scores, E, k, beta):
    z, S = _z(scores), (_cos(E) + 1.0) / 2.0
    n = len(z)
    best = np.zeros(n)
    chosen = []
    for _ in range(k):
        cov = np.maximum(S, best[:, None]).mean(axis=0) - best.mean()
        obj = cov + beta * z / k
        obj[chosen] = -np.inf
        j = int(np.argmax(obj))
        chosen.append(j)
        best = np.maximum(best, S[:, j])
    return np.asarray(chosen)


def _logvendi(K):
    return float(np.log(vendi_score_from_kernel(K)))


def vendi_select(scores, E, k, gamma):
    z, K = _z(scores), _cos(E)
    chosen = [int(np.argmax(z))]
    while len(chosen) < k:
        best, bj = -np.inf, None
        for j in range(len(z)):
            if j in chosen:
                continue
            idx = chosen + [j]
            v = _logvendi(K[np.ix_(idx, idx)]) + gamma * z[idx].mean()
            if v > best:
                best, bj = v, j
        chosen.append(bj)
    return np.asarray(chosen)


def kcenter_select(scores, E, k, eta):
    z, D = _z(scores), 1.0 - _cos(E)
    chosen = [int(np.argmax(z))]
    dmin = D[:, chosen[0]].copy()
    while len(chosen) < k:
        obj = dmin + eta * z
        obj[chosen] = -np.inf
        j = int(np.argmax(obj))
        chosen.append(j)
        dmin = np.minimum(dmin, D[:, j])
    return np.asarray(chosen)


_orig_select = F._select


def _select(name, param, scores, E, k, seed):
    if name == "thr":
        return thr_select(scores, k, param, seed)
    if name == "fl":
        return fl_select(scores, E, k, param)
    if name == "vendi":
        return vendi_select(scores, E, k, param)
    if name == "kcenter":
        return kcenter_select(scores, E, k, param)
    return _orig_select(name, param, scores, E, k, seed)


F._select = _select


def rows(root, gen, sel, judge, cache):
    path = os.path.join(cache, f"{gen}__{sel}__{judge}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    from analyze import Cell, _main_prompts
    from src.config import load_config
    from src.store import load_vocab_file, tune_prompt_list
    cfg = load_config(CONFIG)
    full, _ = load_vocab_file(VOCAB)
    cell = Cell(root, cfg, gen, full, reader="qwen", scorers=(sel, judge))
    k, m = cfg["k"], cfg["m_auth"]

    def rows_for(prompts, seed_base, n_pools):
        out = []
        for i, p in enumerate(prompts):
            for j in range(n_pools):
                seed = seed_base + 1000 * i + j
                ims, labels = cell.pool(p, seed)
                keep = [l is not None for l in labels]
                E = np.stack([cell.emb[im.image_id] for im in ims])
                s = cell.scorers[sel].score(ims, p.text)
                jv = cell.scorers[judge].score(ims, p.text)
                r = F.run_selectors([l or "" for l in labels], keep, E, s, jv, k, m,
                                    FRONTIER_N_NULL, seed, grid=GRID)
                r["_country"], r["_template"] = p.country, p.template
                out.append(r)
        return out

    tu = cfg["tune"]
    res = (rows_for(tune_prompt_list(cfg), tu["seed_base"], tu["pools_per_country"]),
           rows_for(_main_prompts(cfg), cfg["seed"], cfg["n_pools"]))
    os.makedirs(cache, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(res, fh)
    return res


def _one(a):
    root, cache, gen, sel, judge = a
    rows(root, gen, sel, judge, cache)
    return gen, sel, judge


def retention(main, key):
    def vals(r):
        return [r[key]["gain_judge"], r[("topk", None)]["gain_judge"]]
    return boot_stat(main, vals, lambda v: 100 * v[0] / v[1], 11)


def cell_report(root, cache, a3cache, gen, sel, judge):
    tune, main = rows(root, gen, sel, judge, cache)
    ch = F.choose_on_tuning(tune)
    rep = {"generator": gen, "selector": sel, "judge": judge, "families": {},
           "reproduction": {}}
    # reproduction of the registered rows (same seeds, same rng stream)
    with open(os.path.join(a3cache, f"{gen}__{sel}__{judge}.pkl"), "rb") as fh:
        _, ref_main = pickle.load(fh)
    diff = max(abs(a[(f, p)][m] - b[(f, p)][m])
               for a, b in zip(main, ref_main) for f in REF for p in F.SELECTOR_GRID[f]
               for m in ("emb_delta", "gain_judge")
               if np.isfinite(a[(f, p)][m]) and np.isfinite(b[(f, p)][m]))
    rep["reproduction"]["max_abs_diff_vs_a3_rows"] = float(diff)
    ps, pd = ch["stk"]["param"], ch["dpp"]["param"]
    for fam in REF[1:] + NEW:
        p = ch[fam]["param"]
        key = (fam, p)
        e = {"param": p, "inside_margin": ch[fam]["inside_margin"]}
        for i, met in enumerate(("emb_delta", "delta", "gain_judge")):
            e[met] = fixed_design(main, lambda r: r[key][met], 50 + i)
        e["retained_pct"] = retention(main, key)
        if fam in NEW:
            for ref, pr in (("stk", ps), ("dpp", pd)):
                e[f"minus_{ref}"] = {
                    met: fixed_design(main, lambda r: r[key][met] - r[(ref, pr)][met], 60 + i)
                    for i, met in enumerate(("emb_delta", "delta", "gain_judge"))}
        rep["families"][fam] = e
    return rep


def pct(x):
    return f"{100 * (np.exp(x) - 1):+.1f}"


def ci_pct(d):
    lo, hi = d["ci"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/e"
    return f"{pct(d['mean'])} [{pct(lo)}, {pct(hi)}]"


def ci_num(d, f="{:+.3f}"):
    lo, hi = d["ci"]
    return f"{f.format(d['mean'])} [{f.format(lo)}, {f.format(hi)}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--a3cache", required=True, help="build_rows.py cache (reproduction)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(EXP, "results", "extra", "e13"))
    a = ap.parse_args()
    todo = [(a.root, a.cache, g, s, j) for g in GENS for s, j in DIRS]
    with ProcessPoolExecutor(a.jobs) as ex:
        for g, s, j in ex.map(_one, todo):
            print("rows", g, s, j, flush=True)
    reps = [cell_report(a.root, a.cache, a.a3cache, g, s, j) for _, _, g, s, j in todo]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "e13.json"), "w") as fh:
        json.dump({"status": "post hoc, descriptive", "grid": {k: v for k, v in NEW_GRID.items()},
                   "cells": reps}, fh, indent=2)
    L = ["# E13: more re-selector families (post hoc, descriptive)", "",
         "Tuned on tuning pools with the registered rule; 64 main pools; fixed-country t "
         "intervals. DINOv2 and Qwen-label contrasts vs random-16 in %; judge gain in "
         "pool-SD units; retention = judge gain / top-16 judge gain.", ""]
    for r in reps:
        L += [f"## {r['generator']}: {r['selector']} selects, {r['judge']} judges "
              f"(reproduction max diff {r['reproduction']['max_abs_diff_vs_a3_rows']:.2e})", "",
              "| family | param | DINOv2 | Qwen label | judge gain | retained % | "
              "DINOv2 minus STK | DINOv2 minus DPP |", "|---|---|---|---|---|---|---|---|"]
        for fam, e in r["families"].items():
            ret = e["retained_pct"]
            L.append(f"| {fam} | {e['param']} | {ci_pct(e['emb_delta'])} | {ci_pct(e['delta'])} | "
                     f"{ci_num(e['gain_judge'])} | {ret['value']:.1f} [{ret['ci'][0]:.1f}, "
                     f"{ret['ci'][1]:.1f}] | "
                     + (f"{ci_pct(e['minus_stk']['emb_delta'])} | {ci_pct(e['minus_dpp']['emb_delta'])} |"
                        if "minus_stk" in e else "— | — |"))
        L.append("")
    with open(os.path.join(a.out, "e13.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
