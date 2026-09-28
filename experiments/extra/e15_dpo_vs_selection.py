"""E15 (post hoc, descriptive): preference optimization at generation time versus
preference-score selection at the output layer.

SDXL and SDXL with the public Diffusion-DPO UNet (Wallace et al., 2024) share
prompts, pool seeds and sampler settings, so their main pools are paired.  Per
pool and generator:

  rand_emb   mean DINOv2 log Vendi of 200 random 16-subsets
  top_emb[s] DINOv2 log Vendi of the top-16 under scorer s
  rand_lab   mean log rarefied (m=8) dish-label Vendi of random 16-subsets
  top_lab[s] the same for the top-16 (NaN when fewer than m labels resolve)
  score[j]   pool mean of scorer j in native units; top16 mean likewise

Contrasts, all paired by pool, fixed-country t intervals as in the main analysis:
  * generation:  rand(DPO) - rand(SDXL)                  (does DPO narrow the pool?)
  * selection:   top16(SDXL) - rand(SDXL), top16(DPO) - rand(DPO)
  * both:        top16(DPO) - rand(SDXL)
and the same differences in each scorer's native unit.  Diffusion-DPO was trained
on Pick-a-Pic, PickScore's training data, so PickScore is not a held-out judge
for it; ImageReward, LAION-Aes and HPSv2.1 are reported beside it.

    python extra/e15_dpo_vs_selection.py --root <store> --reader qwen
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

from analyze import Cell  # noqa: E402
from src.analysis import _fixed_country_moments, _student_t_ppf, rarefy_to_common_count  # noqa: E402
from src.config import load_config  # noqa: E402
from src.frontier import embedding_log_vendi  # noqa: E402
from src.selectors import random_selector  # noqa: E402
from src.store import _main_prompts, load_vocab_file  # noqa: E402

GENS = ("sdxl", "sdxl-dpo")
SEL = ("pickscore", "imagereward")
JUDGES = ("pickscore", "imagereward", "laion_aes", "hpsv21")
N_RAND = 200


def fixed(values):
    by_c = {}
    for (c, t), v in values.items():
        v = [x for x in v if np.isfinite(x)]
        if v:
            by_c.setdefault(c, {})[t] = np.asarray(v, dtype=float)
    theta, se, df, _ = _fixed_country_moments(by_c)
    if theta is None:
        return {"mean": float("nan"), "ci": [float("nan")] * 2}
    h = _student_t_ppf(0.975, df) * se
    return {"mean": float(theta), "ci": [float(theta - h), float(theta + h)]}


def _lab(labels, idx, m, rng):
    v = rarefy_to_common_count([labels[i] for i in idx if labels[i] is not None], m, rng=rng)
    return float("nan") if v["non_estimable"] else float(np.log(v["value"]))


def pool_metrics(cell, p, seed, k, m):
    ims, labels = cell.pool(p, seed)
    E = np.stack([cell.emb[im.image_id] for im in ims])
    sc = {j: np.asarray(cell.scorers[j].score(ims, p.text), dtype=float) for j in JUDGES}
    rng = np.random.default_rng(seed + 15)
    rs = [random_selector(len(ims), k, rng) for _ in range(N_RAND)]
    out = {"rand_emb": float(np.mean([embedding_log_vendi(E[r]) for r in rs])),
           "rand_lab": float(np.nanmean([_lab(labels, r, m, rng) for r in rs])),
           "pool_score": {j: float(sc[j].mean()) for j in JUDGES},
           "resolved": float(np.mean([l is not None for l in labels]))}
    for s in SEL:
        top = np.argsort(-sc[s], kind="stable")[:k]
        out[f"top_emb:{s}"] = embedding_log_vendi(E[top])
        out[f"top_lab:{s}"] = _lab(labels, top, m, rng)
        out[f"top_score:{s}"] = {j: float(sc[j][top].mean()) for j in JUDGES}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default=os.path.join(EXP, "configs", "e15_dpo.yaml"))
    ap.add_argument("--vocab", default=os.path.join(EXP, "data", "cspace_cuisine_vocab.json"))
    ap.add_argument("--reader", default="qwen")
    ap.add_argument("--out", default=os.path.join(EXP, "results", "extra", "e15"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    k, m = cfg["k"], cfg["m_auth"]
    full, _ = load_vocab_file(a.vocab)
    cells = {g: Cell(a.root, cfg, g, full, reader=a.reader, scorers=JUDGES) for g in GENS}
    per = {g: {} for g in GENS}
    for i, p in enumerate(_main_prompts(cfg)):
        for j in range(cfg["n_pools"]):
            seed = cfg["seed"] + 1000 * i + j
            for g in GENS:
                per[g].setdefault((p.country, p.template), []).append(
                    pool_metrics(cells[g], p, seed, k, m))
    base, dpo = per["sdxl"], per["sdxl-dpo"]

    def paired(f):
        return fixed({c: [f(x, y) for x, y in zip(base[c], dpo[c])] for c in base})

    res = {"status": "post hoc, descriptive", "reader": a.reader, "k": k, "m": m,
           "resolved": {g: float(np.mean([z["resolved"] for c in per[g] for z in per[g][c]]))
                        for g in GENS}}
    for meas in ("emb", "lab"):
        r = {"generation_dpo_minus_sdxl": paired(lambda x, y: y[f"rand_{meas}"] - x[f"rand_{meas}"])}
        for s in SEL:
            r[f"selection_sdxl:{s}"] = paired(lambda x, y: x[f"top_{meas}:{s}"] - x[f"rand_{meas}"])
            r[f"selection_dpo:{s}"] = paired(lambda x, y: y[f"top_{meas}:{s}"] - y[f"rand_{meas}"])
            r[f"both_dpo_top_minus_sdxl_rand:{s}"] = paired(
                lambda x, y: y[f"top_{meas}:{s}"] - x[f"rand_{meas}"])
        res[meas] = r
    q = {}
    for jd in JUDGES:
        q[jd] = {"generation_dpo_minus_sdxl": paired(lambda x, y: y["pool_score"][jd] - x["pool_score"][jd])}
        for s in SEL:
            q[jd][f"selection_sdxl:{s}"] = paired(lambda x, y: x[f"top_score:{s}"][jd] - x["pool_score"][jd])
            q[jd][f"selection_dpo:{s}"] = paired(lambda x, y: y[f"top_score:{s}"][jd] - y["pool_score"][jd])
            q[jd][f"both_dpo_top_minus_sdxl_rand:{s}"] = paired(
                lambda x, y: y[f"top_score:{s}"][jd] - x["pool_score"][jd])
    res["score_native"] = q
    os.makedirs(os.path.join(a.out, a.reader), exist_ok=True)
    with open(os.path.join(a.out, a.reader, "e15.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    def pct(d):
        f = lambda x: 100 * (np.exp(x) - 1)
        return f"{f(d['mean']):+.1f} [{f(d['ci'][0]):+.1f}, {f(d['ci'][1]):+.1f}]"

    def nat(d):
        return f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"

    keys = ["generation_dpo_minus_sdxl"] + [f"{c}:{s}" for s in SEL for c in
            ("selection_sdxl", "selection_dpo", "both_dpo_top_minus_sdxl_rand")]
    L = [f"# E15: Diffusion-DPO vs preference-score selection (reader {a.reader}; post hoc)", "",
         "| contrast | DINOv2 Vendi % | dish-label Vendi % | " + " | ".join(JUDGES) + " (native) |",
         "|---" * (3 + len(JUDGES)) + "|"]
    for kk in keys:
        L.append(f"| {kk} | {pct(res['emb'][kk])} | {pct(res['lab'][kk])} | "
                 + " | ".join(nat(q[jd][kk]) for jd in JUDGES) + " |")
    L += ["", "resolved share: " + ", ".join(f"{g} {res['resolved'][g]:.3f}" for g in GENS)]
    with open(os.path.join(a.out, a.reader, "e15.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
