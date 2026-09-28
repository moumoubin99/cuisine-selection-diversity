"""Fig 1 left panel: which images each selector keeps in ONE pool fixed by rule.

Rule (PAPER_PLAN.md, fixed before looking at the images): SDXL, PickScore
selector, the first main pool in sorted prompt order (cuisine|Brazil|A, pool
seed 0).  Selectors: top-16 by PickScore; STK with the SDXL frozen c = 4 (the
configuration chosen on the tuning pools, `a3__pickscore.chosen_on_tuning`),
k-means seed = pool seed as in analyze.frontier; one random-16 draw from
numpy default_rng(0).  Illustration only.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP = os.path.join(ROOT, "experiments")
sys.path.insert(0, EXP)
from src.frontier import stratified_topk  # noqa: E402
from src.selectors import select_topk  # noqa: E402

M = os.path.join(EXP, "results", "store_mirror")
man = [json.loads(l) for l in open(os.path.join(M, "manifest", "sdxl.jsonl"))]
pool = sorted((r for r in man if r["kind"] == "main" and r["prompt_id"] == "cuisine|Brazil|A"
               and r["pool_seed"] == 0), key=lambda r: r["index"])
sc = {json.loads(l)["image_id"]: json.loads(l)["score"]
      for l in open(os.path.join(M, "scores", "sdxl__pickscore.jsonl"))}
z = np.load(os.path.join(M, "embed", "sdxl__dinov2.npz"), allow_pickle=True)
row = {i: n for n, i in enumerate(z["ids"].tolist())}
E = np.stack([z["E"][row[r["image_id"]]] for r in pool])
s = np.array([sc[r["image_id"]] for r in pool])
c = json.load(open(os.path.join(EXP, "results", "real", "sdxl.json")))[
    "a3__pickscore"]["chosen_on_tuning"]["stk"]["param"]
out = {"pool": "sdxl cuisine|Brazil|A seed 0", "stk_c": c,
       "paths": [r["path"] for r in pool], "pickscore": s.tolist(),
       "topk": sorted(select_topk(s, 16).tolist()),
       "stk": sorted(stratified_topk(s, E, 16, c, seed=0).tolist()),
       "random": sorted(np.random.default_rng(0).choice(64, 16, replace=False).tolist())}
json.dump(out, open(os.path.join(ROOT, "paper", "figures", "data", "hero_pool.json"), "w"), indent=1)
print(c, out["topk"], out["stk"], out["random"], sep="\n")
