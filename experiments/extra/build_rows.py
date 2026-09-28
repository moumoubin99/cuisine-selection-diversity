"""Replay A3 per-pool selector rows for the six post hoc cells (CPU, parallel)
and check them against the stored A3 summaries."""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _rows import EXP, fixed_design, pool_rows  # noqa: E402

GENS = ("sdxl", "flux-schnell", "pixart-sigma", "sd35m")
DIRS = (("pickscore", "imagereward"), ("imagereward", "pickscore"))


def _one(args):
    root, cache, gen, sel, judge = args
    pool_rows(root, gen, sel, judge, cache)
    return gen, sel, judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="store with manifest/scores/tags/embed")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    todo = [(a.root, a.cache, g, s, j) for g in GENS for s, j in DIRS]
    with ProcessPoolExecutor(a.jobs) as ex:
        for g, s, j in ex.map(_one, todo):
            print("rows", g, s, j, flush=True)
    # Reproduction check: tuned configurations and their emb_delta means.
    from src.frontier import choose_on_tuning
    for g, s, j in [(t[2], t[3], t[4]) for t in todo]:
        tune, main_rows = pool_rows(a.root, g, s, j, a.cache)
        ch = choose_on_tuning(tune)
        with open(os.path.join(EXP, "results", "a7", "qwen", f"{g}.json")) as fh:
            ref = json.load(fh)[f"a3__{s}"]["chosen_on_tuning"]
        for fam in ("stk", "mmr", "dpp"):
            p = ch[fam]["param"]
            got = fixed_design(main_rows, lambda r: r[(fam, p)]["emb_delta"], 1)["mean"]
            want = ref[fam]["evaluation"]["emb_delta"]["mean"]
            print(f"check {g}/{s} {fam}: param {p} vs {ref[fam]['param']}; "
                  f"emb {got:.6f} vs {want:.6f}", flush=True)


if __name__ == "__main__":
    main()
