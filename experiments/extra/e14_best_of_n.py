"""E14 (post hoc, descriptive): selection intensity.  How does the diversity of a
kept set of k=16 change as the number of candidates N it is chosen from grows?

For each main pool (64 images) and each N in NS, draw R random N-subsets of the
pool, keep the top-16 of each under the scorer, and average

    emb_delta(N)  = log Vendi_DINOv2(top-16 of N) - mean log Vendi_DINOv2(random-16)
    gain(N)       = held-out-judge gain of the kept set (pool-SD units)

N = 16 is random selection (delta 0 by construction) and N = 64 is the paper's
top-16.  The random-16 reference is the pool's own (random 16 of a random
N-subset is random 16 of the pool).  Label-free and reader-free.  Statistics:
fixed-country t over countries x templates, as in the main analysis; plus the
per-pool least-squares slope of emb_delta on log2(N/16).

    python extra/e14_best_of_n.py --root <store>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

from src.analysis import _fixed_country_moments, _student_t_ppf  # noqa: E402
from src.config import load_config  # noqa: E402
from src.frontier import embedding_log_vendi  # noqa: E402
from src.selectors import random_selector  # noqa: E402
from src.store import CachedGenerator, CachedScorer, _main_prompts, load_embeddings  # noqa: E402

GENS = ("sdxl", "flux-schnell", "pixart-sigma", "sd35m")
SCORERS = (("pickscore", "imagereward"), ("imagereward", "pickscore"),
           ("laion_aes", "pickscore"), ("hpsv21", "pickscore"))
NS = (16, 20, 24, 32, 40, 48, 64)
R = 50
N_EMB = 200


def fixed(values):
    by_c = {}
    for (c, t), v in values.items():
        by_c.setdefault(c, {})[t] = np.asarray(v, dtype=float)
    theta, se, df, _ = _fixed_country_moments(by_c)
    if theta is None:
        return {"mean": float("nan"), "ci": [float("nan")] * 2}
    h = _student_t_ppf(0.975, df) * se
    return {"mean": float(theta), "ci": [float(theta - h), float(theta + h)]}


def _std_gain(v, idx):
    sd = v.std()
    return float((v[idx].mean() - v.mean()) / sd) if sd > 0 else 0.0


def run_gen(args):
    root, config, tag = args
    cfg = load_config(config)
    k = cfg["k"]
    gen = CachedGenerator(root, tag, known_keep="verified", reader="qwen")
    E = load_embeddings(root, tag, "dinov2")
    scorers = {s: CachedScorer(root, tag, s) for s in {x for p in SCORERS for x in p}}
    emb = {(s, N): {} for s, _ in SCORERS for N in NS}
    gain = {(s, N): {} for s, _ in SCORERS for N in NS}
    slope = {s: {} for s, _ in SCORERS}
    x = np.log2(np.asarray(NS, dtype=float) / k)
    for i, p in enumerate(_main_prompts(cfg)):
        for j in range(cfg["n_pools"]):
            seed = cfg["seed"] + 1000 * i + j
            ims = gen.generate(p.text, cfg["n"], seed, p.country)
            X = np.stack([E[im.image_id] for im in ims])
            rng = np.random.default_rng(seed + 14)
            ref = float(np.mean([embedding_log_vendi(X[random_selector(len(ims), k, rng)])
                                 for _ in range(N_EMB)]))
            sc = {s: np.asarray(scorers[s].score(ims, p.text), dtype=float) for s in scorers}
            cell = (p.country, p.template)
            for s, judge in SCORERS:
                ys = []
                for N in NS:
                    ev, gv = [], []
                    reps = 1 if N == len(ims) else R
                    for _ in range(reps):
                        sub = rng.choice(len(ims), size=N, replace=False)
                        top = sub[np.argsort(-sc[s][sub], kind="stable")[:k]]
                        ev.append(embedding_log_vendi(X[top]) - ref)
                        gv.append(_std_gain(sc[judge], top))
                    emb[(s, N)].setdefault(cell, []).append(float(np.mean(ev)))
                    gain[(s, N)].setdefault(cell, []).append(float(np.mean(gv)))
                    ys.append(float(np.mean(ev)))
                slope[s].setdefault(cell, []).append(float(np.polyfit(x, ys, 1)[0]))
    out = {}
    for s, judge in SCORERS:
        out[s] = {"judge": judge,
                  "by_N": {str(N): {"emb_delta": fixed(emb[(s, N)]),
                                    "gain_judge": fixed(gain[(s, N)])} for N in NS},
                  "slope_per_doubling": fixed(slope[s])}
    return tag, out


def pct(x):
    return f"{100 * (np.exp(x) - 1):+.1f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default=os.path.join(EXP, "configs", "amend7.yaml"))
    ap.add_argument("--out", default=os.path.join(EXP, "results", "extra", "e14"))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for tag, out in ex.map(run_gen, [(a.root, a.config, g) for g in GENS]):
            res[tag] = out
            print("done", tag, flush=True)
    with open(os.path.join(a.out, "e14.json"), "w") as fh:
        json.dump({"status": "post hoc, descriptive", "NS": NS, "R": R, "results": res},
                  fh, indent=2)
    L = ["# E14: best-of-N selection intensity (k=16; DINOv2 Vendi vs random-16, %)", ""]
    for g in GENS:
        L += [f"## {g}", "", "| scorer | " + " | ".join(f"N={N}" for N in NS)
              + " | slope per doubling |", "|---" * (len(NS) + 2) + "|"]
        for s, _ in SCORERS:
            r = res[g][s]
            sl = r["slope_per_doubling"]
            L.append(f"| {s} | " + " | ".join(pct(r["by_N"][str(N)]["emb_delta"]["mean"])
                                               for N in NS)
                     + f" | {pct(sl['mean'])} [{pct(sl['ci'][0])}, {pct(sl['ci'][1])}] |")
        L.append("")
    with open(os.path.join(a.out, "e14.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
