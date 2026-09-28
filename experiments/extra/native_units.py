"""Post hoc, descriptive (Codex research review run01, item 6): the A3 retention
rule re-read in the judge's native units.

The registered retention divides the tuned selector's held-out judge gain by
top-k's, each pool's gain standardized by the judge's SD over that pool
(`analyze.frontier.paired`).  Here the same ratio is formed from raw gains
(mean judge score of the set minus the pool mean, in ImageReward or PickScore
units), which weights pools by their judge SD.  Both ratios use the same
paired bootstrap as the stored `judge_gain_retained_ci` (pools resampled within
country x template cells; a pool's top-k and selector gains kept together;
equal-weight country means; seed 7; replicates with a non-positive top-k gain
dropped, interval withheld below 90% valid), so the standardized row
reproduces the stored value and interval.

    python extra/build_rows.py --root <store> --cache <dir>   # first
    python extra/native_units.py --root <store> --cache <dir>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _rows import EXP, boot_stat, fixed_design, pool_rows  # noqa: E402
from build_rows import DIRS, GENS  # noqa: E402
from src.frontier import choose_on_tuning  # noqa: E402

RULE = 0.90


def ratio(m):
    return m[1] / m[0] if m[0] > 0 else np.nan


def retention(main, key, gain):
    tk = fixed_design(main, lambda r: r[("topk", None)][gain], 3)
    gate = (not tk["non_estimable"]) and tk["ci"][0] > 0
    b = boot_stat(main, lambda r: (r[("topk", None)][gain], r[key][gain]), ratio, 7)
    return {"topk_gain": tk["mean"], "topk_gain_ci": tk["ci"], "gate_topk_ci_above_0": gate,
            "retained": b["value"] if gate else float("nan"),
            "retained_ci": b["ci"] if gate else [float("nan")] * 2,
            "meets_90_point": bool(gate and b["value"] >= RULE),
            "ci_above_90": bool(gate and b["ci"][0] >= RULE)}


def cell(root, cache, gen, sel, judge):
    tune, main = pool_rows(root, gen, sel, judge, cache)
    ch = choose_on_tuning(tune)
    with open(os.path.join(EXP, "results", "a7", "qwen", f"{gen}.json")) as fh:
        ref = json.load(fh)[f"a3__{sel}"]["chosen_on_tuning"]
    out = {"generator": gen, "selector": sel, "judge": judge, "families": {}}
    for fam in ("stk", "mmr", "dpp"):
        key = (fam, ch[fam]["param"])
        std, raw = retention(main, key, "gain_judge"), retention(main, key, "raw_gain_judge")
        out["families"][fam] = {
            "param": ch[fam]["param"], "standardized": std, "native": raw,
            "stored_standardized": [ref[fam]["vs_topk"]["judge_gain_retained"],
                                    ref[fam]["vs_topk"]["judge_gain_retained_ci"]],
            "native_minus_standardized": raw["retained"] - std["retained"],
            "rule_verdict_changes": std["meets_90_point"] != raw["meets_90_point"]}
    return out


def markdown(rows):
    L = ["# A3 retention in native judge units (post hoc, descriptive)", "",
         "Retained share of top-16's held-out judge gain by the tuned configuration; "
         "95% paired bootstrap (A3's unit).  `std` is the registered SD-standardized "
         "ratio (reproduces the stored value), `native` uses raw judge units.  "
         "n/e: top-16's own gain interval is not above zero (registered gate).", "",
         "| gen | sel->judge | family (param) | std retained | native retained | "
         "top-16 raw gain | >=90% (std / native) |",
         "|---|---|---|---|---|---|---|"]

    def f(x, ci):
        return ("n/e" if not np.isfinite(x) else
                f"{100 * x:.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")

    for r in rows:
        for fam, v in r["families"].items():
            s, n = v["standardized"], v["native"]
            yn = lambda d: "n/e" if not d["gate_topk_ci_above_0"] else (  # noqa: E731
                "yes" if d["meets_90_point"] else "no")
            L.append(f"| {r['generator']} | {r['selector']}->{r['judge']} | {fam} ({v['param']}) | "
                     f"{f(s['retained'], s['retained_ci'])} | {f(n['retained'], n['retained_ci'])} | "
                     f"{n['topk_gain']:.4f} [{n['topk_gain_ci'][0]:.4f}, {n['topk_gain_ci'][1]:.4f}] | "
                     f"{yn(s)} / {yn(n)}{' **changes**' if v['rule_verdict_changes'] else ''} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.join(EXP, "results", "extra"))
    a = ap.parse_args()
    rows = [cell(a.root, a.cache, g, s, j) for g in GENS for s, j in DIRS]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "native_units.json"), "w") as fh:
        json.dump({"status": "post hoc, descriptive", "rule": RULE, "cells": rows}, fh,
                  indent=1, default=float)
    with open(os.path.join(a.out, "native_units.md"), "w") as fh:
        fh.write(markdown(rows))
    print(markdown(rows))
    for r in rows:
        for fam, v in r["families"].items():
            print("repro", r["generator"], r["selector"], fam, v["standardized"]["retained"],
                  v["standardized"]["retained_ci"], "stored", v["stored_standardized"])


if __name__ == "__main__":
    main()
