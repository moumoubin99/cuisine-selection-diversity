#!/usr/bin/env python3
"""Second-reader sensitivity summary (PREREG_AMENDMENT.md amendment 6).

Reads the per-generator outputs of `analyze.py` for each reader and writes one
markdown file with the cross-reader tables and the robustness rules the
amendment fixed before any second-reader output existed:

  1. "measured dish-label diversity falls under top-k" is reader-robust in a
     cell only if every reader's 95% interval lies below 0;
  2. the faithfulness statement is reader-robust in a cell only if every
     reader's own verifier gives a Q4 - Q1 keep-rate interval below 0;
  3. a second reader's A1 verdict is a sensitivity result, never a
     replacement of the pre-registered one;
  4. agreement on a country's modal dish is reported, and does not separate
     generator from reader.

"Every reader" means the three registered ones (`REGISTERED`).  With fewer,
a cell can already be "no" (one reader's interval reaches 0) but never
"yes": it is "pending" until the missing readers exist (experiment audit,
2026-09-24).

Plus the descriptive generator interaction for repetition (FLUX - SDXL, Welch
interval over the eight country values).  Nothing here is confirmatory.

    python sensitivity.py --reader qwen=results/real --reader pixtral=results/reader_pixtral \
        --reader siglip=results/reader_siglip --out results/sensitivity.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from make_report import fmt, fmt_ci, pct  # noqa: E402

SCORERS = ("pickscore", "imagereward")
GENS = ("sdxl", "flux-schnell")
REGISTERED = ("qwen", "pixtral", "siglip")


def load(readers):
    out = {}
    for name, d in readers:
        for g in GENS:
            p = os.path.join(d, f"{g}.json")
            if os.path.exists(p):
                with open(p) as fh:
                    out[(name, g)] = json.load(fh)
    return out


def _below0(ci):
    return bool(ci) and all(isinstance(c, (int, float)) and math.isfinite(c) for c in ci) \
        and ci[1] < 0


def _complete(names):
    return set(REGISTERED) <= set(names)


def _robust_cell(ok, names):
    if not ok:
        return "no"
    return "yes" if _complete(names) else (
        f"pending ({len(set(names) & set(REGISTERED))}/{len(REGISTERED)} readers)")


def diversity_table(res, names):
    head = "| Generator | Scorer | " + " | ".join(
        f"{n}: change [95% CI], TOST verdict" for n in names) + " | Reader-robust decrease |"
    rows = [head, "|" + "---|" * (3 + len(names))]
    robust = {}
    for g in GENS:
        for s in SCORERS:
            cells, ok = [], True
            for n in names:
                r = res.get((n, g), {}).get(f"a1__{s}")
                if not r:
                    cells.append("n/a")
                    ok = False
                    continue
                e = r["raw"]
                ci = e["marginal"].get("ci")
                ok &= _below0(ci)
                cells.append(f"{fmt(pct(e['mean_delta']))}% {fmt_ci(ci)}, "
                             f"{e['equivalence']['verdict']}")
            robust[(g, s)] = ok and _complete(names)
            rows.append(f"| {g} | {s} | " + " | ".join(cells) + f" | {_robust_cell(ok, names)} |")
    return "\n".join(rows), robust


def faithfulness_table(res, names, key="verifier_keep_by_score_quartile"):
    head = "| Generator | Scorer | " + " | ".join(
        f"{n}: keep Q1→Q4, Q4−Q1 [95% CI]" for n in names) + " | Reader-robust |"
    rows = [head, "|" + "---|" * (3 + len(names))]
    robust = {}
    for g in GENS:
        for s in SCORERS:
            cells, ok = [], True
            for n in names:
                v = ((res.get((n, g), {}).get(f"r1__{s}") or {}).get(key))
                if not v:
                    cells.append("n/a")
                    ok = False
                    continue
                ci = v.get("ci")
                ok &= _below0(ci)
                cells.append(" / ".join(fmt(100 * x, 0, False) for x in v["accuracy_by_bin"])
                             + f", {fmt(100 * v['top_minus_bottom'], 1)} "
                               f"{fmt_ci(ci, f=lambda x: 100 * x)}")
            robust[(g, s)] = ok and _complete(names)
            rows.append(f"| {g} | {s} | " + " | ".join(cells) + f" | {_robust_cell(ok, names)} |")
    return "\n".join(rows), robust


def r1_table(res, names):
    rows = ["| Reader | Generator | Scorer | Verified gap (log) [90% CI] | "
            "A1 verdict under this reader | Verifier AUC | Forced-choice keep | Tagger resolved |",
            "|---|---|---|---|---|---|---|---|"]
    for n in names:
        for g in GENS:
            for s in SCORERS:
                r = res.get((n, g), {}).get(f"r1__{s}")
                a = res.get((n, g), {}).get(f"a1__{s}")
                if not r or not a:
                    continue
                gv = r["verified"]["gap"]
                vd = r.get("verifier_diagnostics") or {}
                td = r.get("tagger_diagnostics") or {}
                # Amendment 6, rule 3: a second reader whose gate passes gives
                # a SENSITIVITY verdict; the pre-registered one is unchanged.
                gate = gv.get("verdict") == "equivalent"
                verdict = a["raw"]["equivalence"]["verdict"] if gate else "not claimable (gate)"
                rows.append(
                    f"| {n} | {g} | {s} | {gv.get('verdict')} {fmt(gv.get('mean'), 3)} "
                    f"{fmt_ci(gv.get('ci'), f=lambda x: x, nd=3)} | {verdict} | "
                    f"{fmt(vd.get('auc_true_vs_decoy'), 3, False)} | "
                    f"{fmt(100 * vd.get('forced_choice_keep_rate', float('nan')), 1, False)}% | "
                    f"{fmt(100 * td.get('resolved_rate', float('nan')), 1, False)}% |")
    return "\n".join(rows)


def _welch(x, y):
    from src.analysis import _student_t_ppf
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan"), [float("nan")] * 2
    vx, vy = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (x.size - 1) + vy ** 2 / (y.size - 1)) if se > 0 else 1
    d = float(x.mean() - y.mean())
    h = _student_t_ppf(0.975, df) * se
    return d, [d - h, d + h]


def repetition_table(res, names):
    rows = ["| Reader | Scorer | Jaccard excess SDXL [95% CI] | FLUX [95% CI] | "
            "FLUX − SDXL (Welch over countries) [95% CI] | Matched excess SDXL | Matched FLUX | "
            "Matched FLUX − SDXL [95% CI] |",
            "|---|---|---|---|---|---|---|---|"]
    for n in names:
        for s in SCORERS:
            rs = res.get((n, "sdxl"), {}).get(f"rep__{s}")
            rf = res.get((n, "flux-schnell"), {}).get(f"rep__{s}")
            if not rs or not rf:
                continue

            def vals(r, k):
                return [v.get(k, float("nan")) for v in r["by_country"].values()]
            d, ci = _welch(vals(rf, "repetition_excess"), vals(rs, "repetition_excess"))
            dm, cim = _welch(vals(rf, "repetition_excess_matched"),
                             vals(rs, "repetition_excess_matched"))
            rows.append(
                f"| {n} | {s} | {fmt(rs['mean_repetition_excess'], 3)} "
                f"{fmt_ci(rs.get('repetition_ci'), f=lambda x: x, nd=3)} | "
                f"{fmt(rf['mean_repetition_excess'], 3)} "
                f"{fmt_ci(rf.get('repetition_ci'), f=lambda x: x, nd=3)} | "
                f"{fmt(d, 3)} {fmt_ci(ci, f=lambda x: x, nd=3)} | "
                f"{fmt(rs.get('mean_repetition_excess_matched'), 3)} "
                f"{fmt_ci(rs.get('repetition_matched_ci'), f=lambda x: x, nd=3)} | "
                f"{fmt(rf.get('mean_repetition_excess_matched'), 3)} "
                f"{fmt_ci(rf.get('repetition_matched_ci'), f=lambda x: x, nd=3)} | "
                f"{fmt(dm, 3)} {fmt_ci(cim, f=lambda x: x, nd=3)} |")
    return "\n".join(rows)


def modal_table(res, names, scorer="pickscore"):
    rows = ["| Generator | Country | " + " | ".join(names) + " |",
            "|" + "---|" * (2 + len(names))]
    for g in GENS:
        countries = sorted({c for n in names for c in
                            ((res.get((n, g), {}).get(f"rep__{scorer}") or {})
                             .get("by_country") or {})})
        for c in countries:
            cells = []
            for n in names:
                v = ((res.get((n, g), {}).get(f"rep__{scorer}") or {})
                     .get("by_country") or {}).get(c)
                cells.append(f"{v['modal_dish']} ({fmt(100 * v['modal_share_pool'], 0, False)}%)"
                             if v and v.get("modal_dish") else "n/a")
            rows.append(f"| {g} | {c} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _raw(d, ci=False):
    """Native judge-score units (unstandardised); n/a in outputs that predate them."""
    if not d or d.get("non_estimable"):
        return "n/a"
    m = f"{d['mean']:+.3f}"
    return m + (f" {fmt_ci(d.get('ci'), f=lambda x: x, nd=3)}" if ci else "")


def retention_table(res, name="qwen"):
    rows = ["| Generator | Selector → judge | Method | Retained [95% bootstrap CI] "
            "| Δ diversity vs top-k [95% t CI] "
            "| Top-k raw judge gain | Raw judge gain vs top-k [95% t CI] |",
            "|---|---|---|---|---|---|---|"]
    for g in GENS:
        for s in SCORERS:
            a = res.get((name, g), {}).get(f"a3__{s}")
            if not a:
                continue
            for fam in ("stk", "mmr", "dpp"):
                ch = a["chosen_on_tuning"].get(fam) or {}
                vs = ch.get("vs_topk")
                if not vs:
                    continue
                rows.append(
                    f"| {g} | {a['selector']} → {a['judge']} | {fam} | "
                    f"{fmt(vs.get('judge_gain_retained'), 3, False)} "
                    f"{fmt_ci(vs.get('judge_gain_retained_ci'), f=lambda x: x, nd=3)} | "
                    f"{fmt(pct(vs['delta'].get('mean')))}% {fmt_ci(vs['delta'].get('ci'))} | "
                    f"{_raw(vs.get('topk_raw_judge_gain'))} | {_raw(vs.get('raw_gain_judge'), True)} |")
    return "\n".join(rows)


def a3_cross_reader_table(res, names):
    """Post hoc, descriptive (after experiment audit run 02): the configuration
    frozen on the pre-registered reader's tuning pools selects the same images
    under every reader, so only the tag-diversity gain over top-k changes.
    Reads `curves_vs_topk`; the embedding gain is reader-free."""
    head = ("| Generator | Selector → judge | Method (frozen on " + names[0] + ") | "
            + " | ".join(f"{n}: tag Δ vs top-k [95% t CI]" for n in names)
            + " | DINOv2 Δ vs top-k [95% t CI] | CI > 0 under all readers |")
    rows = [head, "|" + "---|" * (5 + len(names))]
    robust = {}
    for g in GENS:
        for s in SCORERS:
            a0 = res.get((names[0], g), {}).get(f"a3__{s}")
            if not a0:
                continue
            for fam in ("stk", "mmr", "dpp"):
                ch = a0["chosen_on_tuning"].get(fam) or {}
                if ch.get("non_estimable") or "param" not in ch:
                    continue
                key = f"{fam}|{ch['param']}"
                cells, ok, emb = [], True, None
                for n in names:
                    cv = (res.get((n, g), {}).get(f"a3__{s}") or {}).get("curves_vs_topk")
                    d = (cv or {}).get(key, {}).get("delta")
                    if not d or d.get("non_estimable"):
                        cells.append("n/a")
                        ok = False
                        continue
                    ci = d.get("ci")
                    ok &= bool(ci) and all(math.isfinite(c) for c in ci) and ci[0] > 0
                    cells.append(f"{fmt(pct(d['mean']))}% {fmt_ci(ci)}")
                    if emb is None:
                        emb = cv[key].get("emb_delta")
                ec = (f"{fmt(pct(emb['mean']))}% {fmt_ci(emb.get('ci'))}"
                      if emb and not emb.get("non_estimable") else "n/a")
                robust[(g, s, fam)] = ok and _complete(names)
                rows.append(f"| {g} | {a0['selector']} → {a0['judge']} | {fam} ({ch['param']}) | "
                            + " | ".join(cells) + f" | {ec} | {_robust_cell(ok, names)} |")
    return "\n".join(rows), robust


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", action="append", required=True,
                    help="name=results_dir; the first one is the pre-registered reader")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    readers = [tuple(x.split("=", 1)) for x in a.reader]
    names = [n for n, _ in readers]
    res = load(readers)
    div, div_ok = diversity_table(res, names)
    a3x, a3x_ok = a3_cross_reader_table(res, names)
    fa, fa_ok = faithfulness_table(res, names)
    fw, fw_ok = faithfulness_table(res, names, "verifier_keep_by_score_quartile_within_dish")
    parts = [
        "# Second-reader sensitivity (amendment 6; descriptive, not confirmatory)",
        f"Readers: {', '.join(names)}. The first is pre-registered; its A1/A2/A3 decisions "
        "are unchanged by anything here.",
        "## Tag-based diversity change, top-k vs random-k (rule 1)", div,
        "## Faithfulness: verifier keep rate by selector-score quartile (rule 2)", fa,
        "### Within-dish quartiles", fw,
        "### Same images as the within-dish rows, ranked within the pool (post hoc)",
        faithfulness_table(res, names,
                           "verifier_keep_by_score_quartile_within_dish_subset_pool_rank")[0],
        "## Known-label gate per reader (rule 3)", r1_table(res, names),
        "## Repetition and the generator interaction", repetition_table(res, names),
        "## Modal dish per country, PickScore pools (rule 4)", modal_table(res, names),
        "## A3 retention with interval (pre-registered reader)",
        "t CI: fixed-design Student-t over countries. Bootstrap CI: paired percentile bootstrap "
        "stratified by country and template. Raw gains are in the judge's native units.",
        retention_table(res, names[0]),
        "## A3 configuration frozen on the pre-registered reader, read by every reader (post hoc)",
        "The frozen configuration selects the same images under every reader, so only the "
        "tag-diversity gain changes; retention and the DINOv2 gain are reader-free. Descriptive; "
        "added after experiment audit run 02.",
        a3x,
        "## Summary",
        ("" if _complete(names) else
         f"Only {', '.join(names)} supplied: no cell can be called reader-robust until all of "
         f"{', '.join(REGISTERED)} exist; the lists below are therefore empty by rule."),
        "- Reader-robust diversity decrease: "
        + (", ".join(f"{g}/{s}" for (g, s), ok in div_ok.items() if ok) or "none"),
        "- Reader-robust faithfulness decrease: "
        + (", ".join(f"{g}/{s}" for (g, s), ok in fa_ok.items() if ok) or "none"),
        "- Reader-robust within-dish faithfulness decrease: "
        + (", ".join(f"{g}/{s}" for (g, s), ok in fw_ok.items() if ok) or "none"),
        "- Frozen A3 configuration with a tag-diversity gain over top-k under all readers: "
        + (", ".join(f"{g}/{s}/{f}" for (g, s, f), ok in a3x_ok.items() if ok) or "none"),
    ]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n\n".join(parts) + "\n")
    with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
        json.dump({"readers": names, "complete_reader_set": _complete(names),
                   "diversity_robust": {f"{g}|{s}": v for (g, s), v in div_ok.items()},
                   "faithfulness_robust": {f"{g}|{s}": v for (g, s), v in fa_ok.items()},
                   "faithfulness_within_dish_robust":
                       {f"{g}|{s}": v for (g, s), v in fw_ok.items()},
                   "a3_frozen_config_gain_robust":
                       {f"{g}|{s}|{f}": v for (g, s, f), v in a3x_ok.items()}}, fh, indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
