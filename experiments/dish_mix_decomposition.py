"""Post hoc dish-mix decomposition of the R1 verifier keep-rate drop.

POST HOC (added at paper planning, 2026-09-24, after every result was seen;
not pre-registered).  CPU only; reads the store mirror, no model is run.

For each generator x scorer x reader, the known-label pools are ranked within
pool exactly as analyze.py does (stable double argsort / n; quartile bin =
min(int(4 r), 3)).  The Q4 - Q1 difference of the verifier keep rate is then
split as

    Q4 - Q1 = sum_d (w4_d - w1_d) * rbar_d            (composition)
            + [ (Q4 - Q1) - composition ]             (within-dish remainder)

where w{1,4}_d is dish d's share of quartile 1 / 4 and rbar_d is dish d's
keep rate over all four quartiles.  The composition term is the drop that the
change in dish mix alone would produce if each dish kept its average
recognisability.  Intervals: pool-cluster bootstrap (2,000 resamples, the
resampled pools recompute every term, seed 0), as in known_label.py.
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "results", "store_mirror")
READERS = {"qwen": "", "pixtral": "__pixtral", "siglip": "__siglip"}
REPORTED = {"qwen": "real", "pixtral": "reader_pixtral", "siglip": "reader_siglip"}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def pool_table(gen, scorer, reader):
    man = [r for r in load(os.path.join(M, "manifest", gen + ".jsonl")) if r["kind"] == "known"]
    sc = {r["image_id"]: r["score"] for r in load(os.path.join(M, "scores", f"{gen}__{scorer}.jsonl"))}
    ver = {r["image_id"]: bool(r["kept"]) for r in
           load(os.path.join(M, "verify", gen + READERS[reader] + ".jsonl"))}
    pools = defaultdict(list)
    for r in man:
        pools[(r["prompt_id"], r["pool_seed"])].append(r)
    rows = []  # (pool, dish, bin, kept)
    for key in sorted(pools):
        ims = sorted(pools[key], key=lambda r: r["index"])
        s = np.array([sc[r["image_id"]] for r in ims])
        rank = np.argsort(np.argsort(s, kind="stable"), kind="stable") / float(len(s))
        b = np.minimum((rank * 4).astype(int), 3)
        for r, bb in zip(ims, b):
            rows.append((key, r["dish"], int(bb), ver[r["image_id"]]))
    return rows


def terms(rows):
    dishes = sorted({r[1] for r in rows})
    di = {d: i for i, d in enumerate(dishes)}
    n = np.zeros((len(dishes), 4))
    k = np.zeros((len(dishes), 4))
    for _, d, b, kept in rows:
        n[di[d], b] += 1
        k[di[d], b] += kept
    q1, q4 = k[:, 0].sum() / n[:, 0].sum(), k[:, 3].sum() / n[:, 3].sum()
    rbar = k.sum(1) / n.sum(1)
    w1, w4 = n[:, 0] / n[:, 0].sum(), n[:, 3] / n[:, 3].sum()
    comp = float(((w4 - w1) * rbar).sum())
    return {"total": float(q4 - q1), "composition": comp, "within": float(q4 - q1 - comp)}


def analyse(rows, n_boot=2000, seed=0):
    est = terms(rows)
    by_pool = defaultdict(list)
    for r in rows:
        by_pool[r[0]].append(r)
    ids = sorted(by_pool)
    rng = np.random.default_rng(seed)
    boots = {t: [] for t in est}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        t = terms([r for i in pick for r in by_pool[ids[i]]])
        for name in boots:
            boots[name].append(t[name])
    out = {}
    for name, v in est.items():
        lo, hi = np.quantile(boots[name], [0.025, 0.975])
        out[name] = {"est": v, "ci": [float(lo), float(hi)]}
    out["composition_share"] = est["composition"] / est["total"] if est["total"] else None
    out["n_images"], out["n_pools"] = len(rows), len(ids)
    return out


def main():
    res = {"status": "post_hoc", "interval": "pool_cluster_bootstrap_2000",
           "decomposition": "composition = sum_d (w4_d - w1_d) rbar_d; within = total - composition",
           "cells": {}}
    lines = ["# Post hoc dish-mix decomposition of the Q4 - Q1 verifier keep rate", "",
             "POST HOC, not pre-registered. Points in percentage points; 95% pool-cluster "
             "bootstrap (2,000). `check` = total reported by analyze.py.", "",
             "| generator | scorer | reader | total | composition | within-dish | check |",
             "|---|---|---|---|---|---|---|"]
    for gen in ("sdxl", "flux-schnell"):
        for scorer in ("pickscore", "imagereward"):
            for reader in READERS:
                out = analyse(pool_table(gen, scorer, reader))
                rep = json.load(open(os.path.join(HERE, "results", REPORTED[reader], gen + ".json")))
                chk = rep[f"r1__{scorer}"]["verifier_keep_by_score_quartile"]["top_minus_bottom"]
                out["reported_total"] = chk
                assert abs(chk - out["total"]["est"]) < 1e-9, (gen, scorer, reader, chk, out["total"])
                res["cells"][f"{gen}/{scorer}/{reader}"] = out
                f = lambda t: "{:+.1f} [{:+.1f}, {:+.1f}]".format(
                    100 * out[t]["est"], *(100 * x for x in out[t]["ci"]))
                lines.append(f"| {gen} | {scorer} | {reader} | {f('total')} | {f('composition')} "
                             f"| {f('within')} | {100 * chk:+.1f} |")
    os.makedirs(os.path.join(HERE, "results", "posthoc"), exist_ok=True)
    with open(os.path.join(HERE, "results", "posthoc", "dish_mix_decomposition.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(HERE, "results", "posthoc", "dish_mix_decomposition.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
