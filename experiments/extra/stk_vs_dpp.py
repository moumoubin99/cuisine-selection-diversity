"""Post hoc, descriptive (Codex research review run01, item 4): is STK better
than DPP (and MMR) once held-out judge gain is matched?

Two paired, pool-level comparisons per generator x scorer direction, on the 64
main pools, with A3's unit of resampling (pools within country x template
cells; equal-weight country means of template means):

1. Tuned-on-tuning.  The configurations `choose_on_tuning` froze for each
   family; per-pool STK - DPP (and STK - MMR) in held-out judge gain
   (SD-standardized and native), DINOv2 log-Vendi and Qwen dish-label
   log-Vendi.  Fixed-country t interval (A3's primary) and percentile
   bootstrap.  Dominance is read from the t intervals.
2. Matched judge gain.  For each STK grid point, the other family's DINOv2 and
   Qwen-label contrast at the same mean judge gain, by linear interpolation
   along its grid (anchored at top-k, the family's quality-only limit); where
   segments overlap the best one is used, which favours the comparator.
   Undefined (NaN) outside the comparator's gain range.  Bootstrap interval,
   re-interpolating in every replicate.  The reverse (each comparator grid
   point against STK's curve, anchored at top-k) is also reported, because
   MMR's grid barely leaves top-k's gain.

Selections do not depend on the reader, so DINOv2 contrasts are reader-free;
the label contrast is Qwen's (pre-registered reader).

    python extra/build_rows.py --root <store> --cache <dir>   # first
    python extra/stk_vs_dpp.py --root <store> --cache <dir>
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
from src.frontier import SELECTOR_GRID, choose_on_tuning  # noqa: E402

METRICS = ("gain_judge", "raw_gain_judge", "emb_delta", "delta")
DIV = ("emb_delta", "delta")


def verdict(ci):
    if not np.all(np.isfinite(ci)):
        return "n/e"
    return "better" if ci[0] > 0 else "worse" if ci[1] < 0 else "tie"


def dominance(v):
    """STK vs comparator on (judge gain, one diversity metric), from t intervals."""
    s = set(v)
    if "n/e" in s:
        return "n/e"
    if "worse" not in s and "better" in s:
        return "STK dominates"
    if "better" not in s and "worse" in s:
        return "STK dominated"
    return "neither (tie)" if s == {"tie"} else "neither (trade-off)"


def tuned(main, ch, fam):
    ps, pf = ch["stk"]["param"], ch[fam]["param"]
    out = {"stk_param": ps, f"{fam}_param": pf}
    for i, m in enumerate(METRICS):
        out[m] = fixed_design(main, lambda r: r[("stk", ps)][m] - r[(fam, pf)][m], 20 + i)
    for d in DIV:
        out[f"dominance_{d}"] = dominance([verdict(out["gain_judge"]["ci"]),
                                           verdict(out[d]["ci"])])
    return out


def interp_best(g, xs, ys):
    """Best comparator value at gain g along the piecewise-linear grid curve."""
    best = np.nan
    for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        lo, hi = min(x0, x1), max(x0, x1)
        if lo <= g <= hi:
            y = y0 if hi == lo else y0 + (y1 - y0) * (g - x0) / (x1 - x0)
            best = y if not np.isfinite(best) else max(best, y)
    return best


def curve_keys(fam):
    """Family grid in the order that runs away from top-k, anchored at top-k."""
    return [("topk", None)] + [(fam, p) for p in SELECTOR_GRID[fam]]


def matched(main, point, curve, metric, seed, sign):
    """sign * (point's `metric` - the curve's best `metric` at point's judge
    gain); sign=+1 when `point` is STK, -1 when the curve is STK, so a
    positive value always favours STK."""
    n = len(curve)

    def values(r):
        return ([r[point]["gain_judge"], r[point][metric]]
                + [r[k]["gain_judge"] for k in curve] + [r[k][metric] for k in curve])

    def stat(v):
        return sign * (v[1] - interp_best(v[0], v[2:2 + n], v[2 + n:]))

    out = boot_stat(main, values, stat, seed)
    gains = [fixed_design(main, lambda r, k=k: r[k]["gain_judge"], 1)["mean"] for k in curve]
    out["curve_gain_range"] = [float(min(gains)), float(max(gains))]
    return out


def cell_report(root, cache, gen, sel, judge):
    tune, main = pool_rows(root, gen, sel, judge, cache)
    ch = choose_on_tuning(tune)
    rep = {"generator": gen, "selector": sel, "judge": judge,
           "chosen": {f: ch[f]["param"] for f in ("stk", "mmr", "dpp")},
           "tuned": {f: tuned(main, ch, f) for f in ("dpp", "mmr")},
           "matched": {}, "matched_reverse": {}}
    for fam in ("dpp", "mmr"):
        # (a) STK grid points on the comparator's curve; (b) the comparator's
        # grid points on STK's curve (MMR's grid barely leaves top-k's gain).
        rep["matched"][fam], rep["matched_reverse"][fam] = {}, {}
        for c in SELECTOR_GRID["stk"]:
            rep["matched"][fam][str(c)] = {
                "stk_gain_judge": fixed_design(main, lambda r: r[("stk", c)]["gain_judge"], 1)["mean"],
                **{d: matched(main, ("stk", c), curve_keys(fam), d, 30 + j, +1)
                   for j, d in enumerate(DIV)}}
        for p in SELECTOR_GRID[fam]:
            rep["matched_reverse"][fam][str(p)] = {
                "gain_judge": fixed_design(main, lambda r: r[(fam, p)]["gain_judge"], 1)["mean"],
                **{d: matched(main, (fam, p), curve_keys("stk"), d, 40 + j, -1)
                   for j, d in enumerate(DIV)}}
    return rep


def pct(x):
    return f"{100 * (np.exp(x) - 1):+.1f}%"


def pct_ci(ci):
    return "[n/e]" if not np.all(np.isfinite(ci)) else f"[{pct(ci[0])}, {pct(ci[1])}]"


def markdown(reps):
    L = ["# STK vs DPP / MMR at matched held-out judge gain (post hoc, descriptive)", "",
         "Replayed from the stored pools; per-pool rows reproduce every stored tuned",
         "configuration and DINOv2 mean exactly.  Diversity contrasts are "
         "100(exp(x)-1)% of STK over the comparator; gains are SD-standardized.",
         "95% fixed-country t intervals (tuned) or country x template stratified",
         "bootstrap (matched gain).", "",
         "## 1. Tuned-on-tuning configurations, paired per pool", "",
         "| gen | sel->judge | vs | params (STK / other) | d judge gain | d raw gain | "
         "d DINOv2 | d Qwen label | dominance (DINOv2 / label) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in reps:
        for f, t in r["tuned"].items():
            g, rg, e, d = (t[m] for m in METRICS)
            L.append(f"| {r['generator']} | {r['selector']}->{r['judge']} | {f} | "
                     f"{t['stk_param']} / {t[f + '_param']} | "
                     f"{g['mean']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] | "
                     f"{rg['mean']:+.4f} [{rg['ci'][0]:+.4f}, {rg['ci'][1]:+.4f}] | "
                     f"{pct(e['mean'])} {pct_ci(e['ci'])} | {pct(d['mean'])} {pct_ci(d['ci'])} | "
                     f"{t['dominance_emb_delta']} / {t['dominance_delta']} |")
    L += ["", "## 2. STK minus comparator at STK's judge gain (interpolated)", "",
          "| gen | sel->judge | vs | STK c (gain) | DINOv2 | Qwen label |",
          "|---|---|---|---|---|---|"]
    for r in reps:
        for f, per in r["matched"].items():
            for c, m in per.items():
                star = " *" if int(c) == r["chosen"]["stk"] else ""
                L.append(f"| {r['generator']} | {r['selector']}->{r['judge']} | {f} | "
                         f"{c}{star} ({m['stk_gain_judge']:.3f}) | "
                         f"{pct(m['emb_delta']['value'])} {pct_ci(m['emb_delta']['ci'])} | "
                         f"{pct(m['delta']['value'])} {pct_ci(m['delta']['ci'])} |")
    L += ["", "## 3. STK's curve at the comparator's judge gain (positive favours STK)", "",
          "| gen | sel->judge | vs | param (gain) | DINOv2 | Qwen label |",
          "|---|---|---|---|---|---|"]
    for r in reps:
        for f, per in r["matched_reverse"].items():
            for p, m in per.items():
                star = " *" if float(p) == r["chosen"][f] else ""
                L.append(f"| {r['generator']} | {r['selector']}->{r['judge']} | {f} | "
                         f"{p}{star} ({m['gain_judge']:.3f}) | "
                         f"{pct(m['emb_delta']['value'])} {pct_ci(m['emb_delta']['ci'])} | "
                         f"{pct(m['delta']['value'])} {pct_ci(m['delta']['ci'])} |")
    L += ["", "`*` = configuration frozen on the tuning pools (STK in section 2, the "
          "comparator in section 3).  n/e = the matched gain is outside the curve's grid "
          "range (no extrapolation) or fewer than 90% of bootstrap replicates are "
          "inside it."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.join(EXP, "results", "extra"))
    a = ap.parse_args()
    reps = [cell_report(a.root, a.cache, g, s, j) for g in GENS for s, j in DIRS]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "stk_vs_dpp.json"), "w") as fh:
        json.dump({"status": "post hoc, descriptive", "frontier_n_null": 2000,
                   "cells": reps}, fh, indent=1)
    with open(os.path.join(a.out, "stk_vs_dpp.md"), "w") as fh:
        fh.write(markdown(reps))
    print(markdown(reps))


if __name__ == "__main__":
    main()
