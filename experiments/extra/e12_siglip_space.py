"""E12: the frozen selections re-measured in a second, independent feature space.

Every diversity-aware selector (STK, MMR, DPP) picks its images in DINOv2 space,
and the paper's embedding endpoint is also DINOv2 Vendi, so a selector's
embedding gain is partly the quantity it optimised.  Here the SAME selections
(same pools, same scores, same frozen hyper-parameters as `analyze.frontier`)
are measured in SigLIP so400m image space as well:

    emb_delta_<space> = log Vendi_<space>(selected) - mean_r log Vendi_<space>(random-k)

The DINOv2 column reproduces `a3*.chosen_on_tuning.<sel>.evaluation.emb_delta`
up to the random-k reference draws.  Statistics: the fixed-country t
construction of the main analysis (`_fixed_country_moments`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

from src.analysis import _fixed_country_moments, _student_t_ppf  # noqa: E402
from src.config import load_config  # noqa: E402
from src.frontier import _select, embedding_log_vendi  # noqa: E402
from src.selectors import random_selector  # noqa: E402
from src.store import CachedGenerator, CachedScorer, _main_prompts, load_embeddings  # noqa: E402

PAIRS = [("pickscore", "imagereward", "a3__pickscore"),
         ("imagereward", "pickscore", "a3__imagereward"),
         ("laion_aes", "pickscore", "a3x__laion_aes__pickscore"),
         ("hpsv21", "pickscore", "a3x__hpsv21__pickscore")]
SPACES = ("dinov2", "siglip")


def fixed(values):
    """values: {(country, template): [per-pool]} -> mean, t CI."""
    by_c = {}
    for (c, t), v in values.items():
        by_c.setdefault(c, {})[t] = np.asarray(v, dtype=float)
    theta, se, df, _ = _fixed_country_moments(by_c)
    if theta is None:
        return {"mean": float("nan"), "ci": [float("nan")] * 2}
    h = _student_t_ppf(0.975, df) * se
    return {"mean": float(theta), "ci": [float(theta - h), float(theta + h)],
            "pct": float(100 * (np.exp(theta) - 1))}


def run_gen(root, cfg, tag, frozen_path, n_emb):
    with open(frozen_path) as fh:
        frozen = json.load(fh)
    gen = CachedGenerator(root, tag, known_keep="verified", reader="qwen")
    E = {s: load_embeddings(root, tag, s) for s in SPACES}
    k = cfg["k"]
    out = {}
    for sel, judge, key in PAIRS:
        if key not in frozen:
            continue
        ch = frozen[key]["chosen_on_tuning"]
        sels = {name: c["param"] for name, c in ch.items() if not c.get("non_estimable")}
        sc = CachedScorer(root, tag, sel)
        vals = {(name, sp): {} for name in sels for sp in SPACES}
        paired = {(name, sp): {} for name in sels if name != "topk" for sp in SPACES}
        for i, p in enumerate(_main_prompts(cfg)):
            for j in range(cfg["n_pools"]):
                seed = cfg["seed"] + 1000 * i + j
                ims = gen.generate(p.text, cfg["n"], seed, p.country)
                s = sc.score(ims, p.text)
                X = {sp: np.stack([E[sp][im.image_id] for im in ims]) for sp in SPACES}
                rng = np.random.default_rng(seed)
                ref = {}
                draws = [random_selector(len(ims), k, rng) for _ in range(n_emb)]
                for sp in SPACES:
                    ref[sp] = float(np.mean([embedding_log_vendi(X[sp][d]) for d in draws]))
                cell = (p.country, p.template)
                got = {}
                for name, prm in sels.items():
                    idx = np.asarray(_select(name, prm, s, X["dinov2"], k, seed))
                    for sp in SPACES:
                        v = embedding_log_vendi(X[sp][idx]) - ref[sp]
                        got[(name, sp)] = v
                        vals[(name, sp)].setdefault(cell, []).append(v)
                for name in sels:
                    if name == "topk":
                        continue
                    for sp in SPACES:
                        paired[(name, sp)].setdefault(cell, []).append(
                            got[(name, sp)] - got[("topk", sp)])
        res = {"selector": sel, "judge": judge, "params": sels, "evaluation": {}, "vs_topk": {}}
        for (name, sp), v in vals.items():
            res["evaluation"].setdefault(name, {})[sp] = fixed(v)
        for (name, sp), v in paired.items():
            res["vs_topk"].setdefault(name, {})[sp] = fixed(v)
        # reference: the paper's DINOv2 number for the same configuration
        res["paper_dinov2_emb_delta"] = {
            name: ch[name]["evaluation"]["emb_delta"]["mean"] for name in sels}
        out[key] = res
        e = res["evaluation"]
        print(tag, key, {n: (round(e[n]["dinov2"]["pct"], 1), round(e[n]["siglip"]["pct"], 1),
                             [round(100 * (np.exp(x) - 1), 1) for x in e[n]["siglip"]["ci"]])
                         for n in e}, flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default=os.path.join(EXP, "configs", "amend7.yaml"))
    ap.add_argument("--gens", default="sdxl,flux-schnell,pixart-sigma")
    ap.add_argument("--frozen-dir", default=os.path.join(EXP, "results", "a7", "qwen"))
    ap.add_argument("--out", default=os.path.join(EXP, "results", "extra", "e12"))
    ap.add_argument("--n-emb", type=int, default=200)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    os.makedirs(a.out, exist_ok=True)
    for tag in a.gens.split(","):
        if not os.path.exists(os.path.join(a.root, "embed", f"{tag}__siglip.npz")):
            print("skip (no siglip embeddings):", tag)
            continue
        res = run_gen(a.root, cfg, tag, os.path.join(a.frozen_dir, f"{tag}.json"), a.n_emb)
        with open(os.path.join(a.out, f"{tag}.json"), "w") as fh:
            json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
