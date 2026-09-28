#!/usr/bin/env python3
"""Post-hoc, DESCRIPTIVE (Codex review item 2): what does DINOv2 space encode
on the known-label pools, where every image's prompted dish is known?

  dish gap     within a pool (one country, 64 images, several prompted
               dishes): mean cosine of same-prompted-dish pairs minus mean
               cosine of different-dish pairs.
  country gap  for each pool's images, mean cosine to DIFFERENT-dish images of
               the same country minus mean cosine to images of other countries
               (dish held different on both sides).

Each is a per-pool value (16 pools per generator); intervals are percentile
bootstraps over pools stratified by country.  Repeated on the images the
pre-registered Qwen forced-choice verifier kept (`kept`), and per country.

    python extra/dino_dish_similarity.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from _common import GENS, OUT, RES, merged_root, pct

from src.labels import normalize
from src.store import load_embeddings, read_jsonl

N_BOOT = 2000


def unit(E):
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def pool_values(rows, E, kept):
    """rows: all known rows of the generator (possibly verifier-filtered)."""
    idx = {r["image_id"]: i for i, r in enumerate(rows)}
    S = E @ E.T
    dish = np.asarray([normalize(r["dish"]) for r in rows], dtype=object)
    ctry = np.asarray([r["country"] for r in rows], dtype=object)
    pool = np.asarray([r["pool_seed"] for r in rows])
    out = []
    for ps in sorted(set(pool.tolist())):
        m = np.where(pool == ps)[0]
        sub = S[np.ix_(m, m)]
        same = dish[m][:, None] == dish[m][None, :]
        off = ~np.eye(len(m), dtype=bool)
        sd, dd = sub[same & off], sub[~same]
        dish_gap = float(sd.mean() - dd.mean()) if sd.size and dd.size else float("nan")
        c = ctry[m[0]]
        same_c = (ctry == c)
        vals_sc, vals_oc = [], []
        for i in m:
            sc = same_c & (dish != dish[i])
            sc[i] = False
            if sc.any():
                vals_sc.append(S[i, sc].mean())
            vals_oc.append(S[i, ~same_c].mean())
        out.append({"pool_seed": int(ps), "country": c, "n": int(len(m)),
                    "n_dishes": int(len(set(dish[m]))),
                    "same_dish_cos": float(sd.mean()) if sd.size else float("nan"),
                    "diff_dish_cos": float(dd.mean()) if dd.size else float("nan"),
                    "dish_gap": dish_gap,
                    "same_country_diff_dish_cos": float(np.mean(vals_sc)),
                    "other_country_cos": float(np.mean(vals_oc)),
                    "country_gap": float(np.mean(vals_sc) - np.mean(vals_oc))})
    return out


def summarize(pools, key, rng):
    by_c = {}
    for p in pools:
        if np.isfinite(p[key]):
            by_c.setdefault(p["country"], []).append(p[key])
    by_c = {c: np.asarray(v) for c, v in by_c.items()}
    point = float(np.mean(np.concatenate(list(by_c.values()))))
    boots = [np.mean(np.concatenate([v[rng.integers(0, v.size, v.size)] for v in by_c.values()]))
             for _ in range(N_BOOT)]
    return {"mean": point, "ci": pct(boots), "n_pools": int(sum(v.size for v in by_c.values())),
            "by_country": {c: float(v.mean()) for c, v in sorted(by_c.items())}}


def main():
    root = merged_root()
    rng = np.random.default_rng(3)
    res = {"status": "post hoc, descriptive", "space": "DINOv2 (cosine)", "generators": {}}
    for gen in GENS:
        rows = [r for r in read_jsonl(os.path.join(root, "manifest", f"{gen}.jsonl"))
                if r["kind"] == "known"]
        emb = load_embeddings(root, gen)
        ver = {v["image_id"]: v for v in read_jsonl(os.path.join(root, "verify", f"{gen}.jsonl"))}
        g = res["generators"][gen] = {}
        for scope in ("all", "qwen_kept"):
            sel = rows if scope == "all" else [r for r in rows if ver.get(r["image_id"], {}).get("kept")]
            E = unit(np.stack([emb[r["image_id"]] for r in sel]).astype(np.float64))
            pv = pool_values(sel, E, None)
            g[scope] = {"n_images": len(sel),
                        **{k: summarize(pv, k, rng) for k in
                           ("same_dish_cos", "diff_dish_cos", "dish_gap",
                            "same_country_diff_dish_cos", "other_country_cos", "country_gap")},
                        "pools": pv}
        print(f"[{gen}] dish_gap {g['all']['dish_gap']['mean']:+.3f} "
              f"country_gap {g['all']['country_gap']['mean']:+.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "dino_dish_similarity.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    L = ["# What DINOv2 space separates on known-label pools (post hoc, descriptive)", "",
         "Cosine gaps; 95% percentile bootstrap over the 16 pools per generator, "
         "stratified by country.", "",
         "| gen | scope | imgs | same-dish cos | diff-dish cos | dish gap | same-country (diff dish) cos | other-country cos | country gap |",
         "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: f"{v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
    for gen, g in res["generators"].items():
        for scope, v in g.items():
            L.append(f"| {gen} | {scope} | {v['n_images']} | {v['same_dish_cos']['mean']:.3f} | "
                     f"{v['diff_dish_cos']['mean']:.3f} | {f(v['dish_gap'])} | "
                     f"{v['same_country_diff_dish_cos']['mean']:.3f} | {v['other_country_cos']['mean']:.3f} | "
                     f"{f(v['country_gap'])} |")
    L += ["", "## Per-country dish gap (all images)", "", "| gen | " +
          " | ".join(sorted(res["generators"]["sdxl"]["all"]["dish_gap"]["by_country"])) + " |",
          "|---|" + "---|" * 8]
    for gen, g in res["generators"].items():
        L.append(f"| {gen} | " + " | ".join(f"{x:+.3f}" for x in
                                            g["all"]["dish_gap"]["by_country"].values()) + " |")
    L.append("")
    with open(os.path.join(OUT, "dino_dish_similarity.md"), "w") as fh:
        fh.write("\n".join(L))
    print("wrote", os.path.join(OUT, "dino_dish_similarity.{json,md}"))


if __name__ == "__main__":
    main()
