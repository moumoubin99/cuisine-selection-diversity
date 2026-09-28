"""E11 sensitivity (post hoc, descriptive): does the real-photograph result rest
on one country or on dishes with many photographs?

  * leave-one-country-out (LOCO): the equal-weight country mean without each
    country in turn, with the same half-sampling intervals as E11;
  * dish cap: every dish keeps at most C photographs (a fixed random subset),
    so dishes with many photographs cannot dominate a pool.

Uses the E11 design (pool_max, frac, n_pools, n_rand) and its data unchanged.

    python extra/e11_sensitivity.py --n-boot 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import e11_realphoto_selection as e11  # noqa: E402

SCORERS = ("pickscore", "imagereward", "laion_aes", "hpsv21")


def _capped(base, cap, seed):
    if cap is None:
        return base
    rng = np.random.default_rng(seed)
    return {c: [d if len(d) <= cap else d[rng.permutation(len(d))[:cap]] for d in ds]
            for c, ds in base.items()}


def _variant(base, spec):
    kind, v = spec
    if kind == "loco":
        return {c: ds for c, ds in base.items() if c != v}
    if kind == "cap":
        return _capped(base, v, 20260927)
    return base


def _job(args):
    root, vocab, scorer, spec, seed, n_boot, a = args
    D = e11.Data(root, vocab)
    g = _variant(e11._groups(D), spec)
    rng = np.random.default_rng(seed)
    _, point = e11.estimate(D, scorer, g, rng, a)
    boots = []
    for _ in range(n_boot):
        h = {c: [ds[i] for i in rng.permutation(len(ds))[:(len(ds) + 1) // 2]] for c, ds in g.items()}
        boots.append(e11.estimate(D, scorer, h, rng, a)[1])
    out = {}
    for m in point:
        v = np.array([b[m] for b in boots])
        v = v[np.isfinite(v)]
        ci = e11._ci(point[m], v) if len(v) else [float("nan")] * 2
        out[m] = {"log_delta": point[m], "pct": 100 * (np.exp(point[m]) - 1),
                  "ci_pct": [100 * (np.exp(x) - 1) for x in ci]}
    return scorer, spec, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(e11.EXP, "results", "store_mirror_a7", "realimg"))
    ap.add_argument("--vocab", default=os.path.join(e11.EXP, "data", "cspace_cuisine_vocab.json"))
    ap.add_argument("--out", default=os.path.join(e11.EXP, "results", "extra", "e11"))
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--caps", default="2,3")
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args(argv)
    with open(os.path.join(a.out, "e11.json")) as fh:
        design = json.load(fh)["design"]
    d = argparse.Namespace(**{k: design[k] for k in ("pool_max", "frac", "n_pools", "n_rand")})
    D = e11.Data(a.root, a.vocab)
    specs = [("loco", c) for c in D.countries] + [("cap", int(c)) for c in a.caps.split(",")]
    jobs = [(a.root, a.vocab, s, sp, 7919 * i + 13, a.n_boot, d)
            for i, (s, sp) in enumerate((s, sp) for s in SCORERS for sp in specs)]
    res = {"status": "post hoc, descriptive", "design": {**design, "n_boot": a.n_boot},
           "results": {s: {} for s in SCORERS}}
    with ProcessPoolExecutor(a.workers) as ex:
        for s, sp, out in ex.map(_job, jobs):
            res["results"][s][f"{sp[0]}:{sp[1]}"] = out
            print(s, sp, {m: round(out[m]["pct"], 1) for m in out}, flush=True)
    with open(os.path.join(a.out, "e11_sensitivity.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    L = ["# E11 sensitivity (post hoc): leave-one-country-out and dish cap", "",
         "Dataset-label Vendi change of the top quarter vs random (%, 95% half-sampling CI).", "",
         "| scorer | " + " | ".join(f"{k}:{v}" for k, v in specs) + " |",
         "|---" * (len(specs) + 1) + "|"]
    for s in SCORERS:
        r = res["results"][s]
        L.append(f"| {s} | " + " | ".join(
            "{:+.1f} [{:+.1f}, {:+.1f}]".format(r[f"{k}:{v}"]["dataset"]["pct"], *r[f"{k}:{v}"]["dataset"]["ci_pct"])
            for k, v in specs) + " |")
    with open(os.path.join(a.out, "e11_sensitivity.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
