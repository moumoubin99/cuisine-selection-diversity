"""Tables and figures from the analysis output.

    python3 make_report.py --in results/real --out results/real/report

Reads every `<generator>.json` written by analyze.py and writes

    tables.md              A1/A2, A3, R1, repetition and provenance tables
    fig_country_delta.pdf  per-country proportional change, top-k vs random-k
    fig_frontier.pdf       judge gain vs diversity change for STK / MMR / DPP
    fig_r1_quartile.pdf    tagger accuracy by selector-score quartile

(each figure is also saved as a PNG).  Nothing here recomputes a statistic:
every number is read from analyze.py's output, so the report cannot drift from
the pre-registered analysis.
"""
import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Categorical slots 1-3 of the reference palette, validated all-pairs in light
# mode (worst CVD dE 9.2).  Slot 3 is below 3:1 on white, so every series is
# also direct-labelled.
SERIES = {"stk": "#2a78d6", "mmr": "#eb6834", "dpp": "#1baf7a"}
SERIES_NAME = {"stk": "STK (stratified top-k)", "mmr": "MMR", "dpp": "DPP"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"
UNFILTERED, VERIFIED = "#2a78d6", "#eb6834"


def pct(x):
    """Log contrast -> proportional change in percent."""
    return 100.0 * math.expm1(x) if _finite(x) else float("nan")


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def fmt(x, nd=1, sign=True):
    if not _finite(x):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def fmt_ci(ci, f=pct, nd=1):
    if not ci or not all(_finite(c) for c in ci):
        return "n/a"
    return f"[{fmt(f(ci[0]), nd)}, {fmt(f(ci[1]), nd)}]"


def fmt_p(p):
    if not _finite(p):
        return "n/a"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def load(indir):
    out = {}
    for path in sorted(glob.glob(os.path.join(indir, "*.json"))):
        with open(path) as f:
            res = json.load(f)
        if isinstance(res, dict) and "generator" in res:
            out[res["generator"]] = res
    if not out:
        raise SystemExit(f"no analyze.py output in {indir}")
    return out


def cells(results):
    for gen, res in results.items():
        for sc in res.get("scorers", []):
            yield gen, sc, res


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- tables

def table_a1(results):
    rows = ["| Generator | Scorer | Endpoint | Change (top-k vs random-k) | 90% CI (TOST) | "
            "95% CI | Verdict | Deciding known-label gap, log [90% CI] | A2 status | Pools |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for gen, sc, res in cells(results):
        a = res.get(f"a1__{sc}")
        if not a:
            continue
        for ep in ("raw", "conditional"):
            e = a[ep]
            eq, m = e["equivalence"], e["marginal"]
            # Round-5 code review, P2: the interval that gates A1 sits beside it.
            # Only the verified gap decides (amendment 4), and only the raw
            # endpoint is confirmatory.
            gap = "-"
            if ep == "raw" and res.get(f"r1__{sc}"):
                g = res[f"r1__{sc}"]["verified"]["gap"]
                gap = (f"{g['verdict']} {fmt(g.get('mean'), 3)} "
                       f"{fmt_ci(g.get('ci'), f=lambda x: x, nd=3)} (verified)")
            rows.append(
                f"| {gen} | {sc} | {ep} | {fmt(pct(e['mean_delta']))}% | "
                f"{fmt_ci(eq.get('ci'))} | {fmt_ci(m.get('ci'))} | {eq['verdict']} | "
                f"{gap} | {e['a2']['status']} | {e['n_estimable']}/{a['n_pools_total']} |")
    return "\n".join(rows)


def table_claims(results):
    rows = ["| Generator | Scorer | Claim | Status | Holm p | Known-label gap (basis) | "
            "Withheld because |",
            "|---|---|---|---|---|---|---|"]
    for gen, sc, res in cells(results):
        c = res.get(f"claims__{sc}")
        if not c:
            continue
        for name in ("A1", "A2", "A3"):
            d = c.get(name, {})
            status = d.get("status", "claimed" if d.get("claim") else "not_claimed")
            if c.get("confirmatory") is False:
                status = ("rule met (sensitivity reader, not a claim)"
                          if d.get("rule_met_sensitivity_only")
                          else "rule not met (sensitivity reader)")
            gap = ""
            if name == "A1":
                gap = f"{d.get('known_label_gap', 'n/a')} ({d.get('known_label_gap_basis', 'n/a')})"
            why = "; ".join(d.get("withheld_because", [])) or "-"
            rows.append(f"| {gen} | {sc} | {name} | {'**claimed**' if d.get('claim') else status} | "
                        f"{fmt_p(d.get('p_holm'))} | {gap} | {why} |")
    return "\n".join(rows)


def table_a3(results):
    # The label-free secondary endpoint (DINOv2 embedding Vendi, prereg §6)
    # sits beside the tag-based diversity; it needs no tagger.
    rows = ["| Generator | Selector → judge | Method | Param | Diversity vs random-k | "
            "Embedding Vendi vs random-k [95% CI] | "
            "Δ vs top-k [95% CI] | Judge gain (SD units) | Top-k judge gain (SD units) | "
            "Retained [95% bootstrap CI] | Raw judge gain | Raw Δ vs top-k [95% CI] |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for gen, sc, res in cells(results):
        a = res.get(f"a3__{sc}")
        if not a:
            continue
        for fam in ("topk", "stk", "mmr", "dpp"):
            ch = a["chosen_on_tuning"].get(fam)
            if not ch or "evaluation" not in ch:
                continue
            ev, vs = ch["evaluation"], ch.get("vs_topk", {})
            dvs = vs.get("delta", {}) if fam != "topk" else {}
            rows.append(
                f"| {gen} | {a['selector']} → {a['judge']} | {fam} | {ch.get('param')} | "
                f"{fmt(pct(ev['delta'].get('mean')))}% | "
                f"{fmt(pct((ev.get('emb_delta') or {}).get('mean')))}% "
                f"{fmt_ci((ev.get('emb_delta') or {}).get('ci'))} | "
                + (f"{fmt(pct(dvs.get('mean')))}% {fmt_ci(dvs.get('ci'))} | " if dvs else "- | ")
                + f"{fmt(ev['gain_judge'].get('mean'), 3, False)} | "
                f"{fmt(vs.get('topk_judge_gain', {}).get('mean'), 3, False) if fam != 'topk' else '-'} | "
                + (f"{fmt(vs.get('judge_gain_retained'), 3, False)} "
                   f"{fmt_ci(vs.get('judge_gain_retained_ci'), f=lambda x: x, nd=3)} |"
                   if fam != 'topk' else "- |")
                + f" {fmt((ev.get('raw_gain_judge') or {}).get('mean'), 3, False)} | "
                + (f"{fmt((vs.get('raw_gain_judge') or {}).get('mean'), 3, False)} "
                   f"{fmt_ci((vs.get('raw_gain_judge') or {}).get('ci'), f=lambda x: x, nd=3)} |"
                   if fam != 'topk' and vs.get('raw_gain_judge') else "- |"))
    return "\n".join(rows)


def table_r1(results):
    rows = ["| Generator | Scorer | Gap, verified — decides (log) | 90% CI | "
            "Gap, unfiltered — sensitivity (log) | 90% CI | "
            "Tagger acc. Q1→Q4 (verified) | Q4−Q1 [95% CI] | Verifier keep | "
            "Resolved | Native script | Verifier keep Q1→Q4 | Q4−Q1 [95% CI] | "
            "Within-dish keep Q1→Q4 | Q4−Q1 [95% CI] |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for gen, sc, res in cells(results):
        r = res.get(f"r1__{sc}")
        if not r:
            continue
        gv, gu = r["verified"]["gap"], r["unfiltered"]["gap"]
        q = r["tagger_accuracy_by_score_quartile"]["verified"]
        acc = " / ".join(fmt(100 * x, 0, False) for x in q["accuracy_by_bin"])
        td = r.get("tagger_diagnostics", {})
        rows.append(
            f"| {gen} | {sc} | {gv['verdict']}"
            f"{' (exact agreement)' if gv.get('exact_agreement') else ''} "
            f"({fmt(gv.get('mean'), 3)}) | "
            f"{fmt_ci(gv.get('ci'), f=lambda x: x, nd=3)} | "
            f"{gu['verdict']} ({fmt(gu.get('mean'), 3)}) | "
            f"{fmt_ci(gu.get('ci'), f=lambda x: x, nd=3)} | "
            f"{acc} | {fmt(100 * q['top_minus_bottom'], 1)} "
            f"{fmt_ci(q.get('ci'), f=lambda x: 100 * x)} | "
            f"{fmt(100 * r.get('verifier_keep_rate', float('nan')), 1, False)}% | "
            f"{fmt(100 * td.get('resolved_rate', float('nan')), 1, False)}% | "
            f"{fmt(100 * td.get('native_script_rate', float('nan')), 1, False)}% | "
            f"{_keep_by_quartile(r)} | "
            f"{_keep_by_quartile(r, 'verifier_keep_by_score_quartile_within_dish')} |")
    return "\n".join(rows)


def _keep_by_quartile(r, key="verifier_keep_by_score_quartile"):
    """Verifier keep rate by within-pool selector-score quartile: does the
    scorer prefer images that depict the named dish less faithfully?  With the
    `_within_dish` key, quartiles are taken within each prompted dish of a pool
    (amendment 6)."""
    v = r.get(key)
    if not v:
        return "n/a | n/a"
    return (" / ".join(fmt(100 * x, 0, False) for x in v["accuracy_by_bin"])
            + f" | {fmt(100 * v['top_minus_bottom'], 1)} "
              f"{fmt_ci(v.get('ci'), f=lambda x: 100 * x)}")


def table_rep(results):
    rows = ["| Generator | Scorer | Jaccard excess (top-k − random) [95% CI] | "
            "Countries > 0 | Modal-share excess [95% CI] | "
            "Cardinality-matched Jaccard excess [95% CI] | Countries > 0 |",
            "|---|---|---|---|---|---|---|"]
    for gen, sc, res in cells(results):
        r = res.get(f"rep__{sc}")
        if not r:
            continue
        rows.append(
            f"| {gen} | {sc} | {fmt(r['mean_repetition_excess'], 3)} "
            f"{fmt_ci(r.get('repetition_ci'), f=lambda x: x, nd=3)} | "
            f"{r['n_countries_positive']}/{len(r['by_country'])} | "
            f"{fmt(r['mean_modal_share_excess'], 3)} "
            f"{fmt_ci(r.get('modal_share_ci'), f=lambda x: x, nd=3)} | "
            f"{fmt(r.get('mean_repetition_excess_matched'), 3)} "
            f"{fmt_ci(r.get('repetition_matched_ci'), f=lambda x: x, nd=3)} | "
            f"{r.get('n_countries_positive_matched', 'n/a')}/{len(r['by_country'])} |")
    return "\n".join(rows)


def table_dose(results):
    """Selection fraction (descriptive): every k on the primary contrast's
    estimand (log selected - mean log null) with one null count; the secondary
    k also as a paired per-pool difference from the primary k."""
    rows = ["| Generator | Scorer | k | Pools estimable | Change vs random-k [95% CI] | "
            "Difference from primary k, log units [95% CI] |",
            "|---|---|---|---|---|---|"]
    for gen, sc, res in cells(results):
        for k, d in sorted(((res.get(f"a1__{sc}") or {}).get("dose_response_summary")
                            or {}).items(), key=lambda t: int(t[0])):
            v = d.get("vs_primary_k")
            diff = (f"{fmt(v.get('mean_diff'), 3, False)} "
                    f"{fmt_ci(v.get('ci'), f=lambda x: x, nd=3)} (vs k={v['k_ref']}, "
                    f"{v['n_pools']} pools)" if v and "mean_diff" in v else "-")
            rows.append(f"| {gen} | {sc} | {k} | {d['n_estimable']}/{d['n_pools']} | "
                        f"{fmt(pct(d.get('mean_log_contrast')))}% {fmt_ci(d.get('ci'))} | {diff} |")
    return "\n".join(rows)


def table_provenance(results):
    rows = ["| Generator | Config | Vocabulary | Store (manifest) | Code | n_null | "
            "Frontier n_null |", "|---|---|---|---|---|---|---|"]
    for gen, res in results.items():
        p = res.get("provenance", {})
        rows.append(f"| {gen} | `{p.get('config_sha')}` | `{p.get('vocab_sha')}` | "
                    f"`{p.get('store_sha', {}).get('manifest')}` | `{p.get('code_sha')}` | "
                    f"{p.get('n_null')} | {p.get('frontier_n_null')} |")
    return "\n".join(rows)


def write_tables(results, outdir):
    readers = {r.get("provenance", {}).get("reader") for r in results.values()} - {None}
    claims_title = ("## Decision rules applied to a sensitivity reader ({}; amendment 6, "
                    "not confirmatory)".format(", ".join(sorted(readers))) if readers
                    else "## Pre-registered claims (Holm over {A1, A2, A3})")
    parts = [
        "# Results tables",
        "",
        "Generated by `make_report.py` from analyze.py output. Changes are proportional "
        "(exp(Δ) − 1) in exact-match Vendi at m = 8, top-k minus random-k, "
        "equal-weight country marginal with template-balanced country values. "
        "The equivalence margin is −10% / +10%. Known-label gaps are in log units "
        "(tagger minus prompted-label contrast) with 90% TOST intervals against a "
        "±5% margin (log 0.95 = −0.051, log 1.05 = 0.049); only the verified gap "
        "decides A1, and the unfiltered gap is a sensitivity analysis.",
        "",
        "## A1 / A2: diversity change under aesthetic top-k", "", table_a1(results), "",
        claims_title, "", table_claims(results), "",
        "## A3: diversity-aware selection (configurations chosen on tuning pools)", "",
        "Judge gains are standardised by the judge's SD over the 64-candidate pool (the "
        "pre-registered estimand); raw gains are selected mean minus pool mean in the judge's "
        "native units. Unlabelled 95% CIs are fixed-design Student-t intervals over countries; "
        "retention uses a paired stratified percentile bootstrap.", "",
        table_a3(results), "",
        "## R1: known-label stress test (named prompts)", "", table_r1(results), "",
        "## Repetition (descriptive)", "", table_rep(results), "",
        "## Secondary k (dose response, descriptive)", "", table_dose(results), "",
        "## Provenance", "", table_provenance(results), "",
    ]
    with open(os.path.join(outdir, "tables.md"), "w") as f:
        f.write("\n".join(parts))


# -------------------------------------------------------------------- figures

def fig_country_delta(results, outdir):
    panels = [(g, s, r) for g, s, r in cells(results) if r.get(f"a1__{s}")]
    if not panels:
        return
    countries = sorted({c for g, s, r in panels
                        for c in r[f"a1__{s}"]["raw"]["marginal"]["country_means"]})
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels) + 0.9, 3.4),
                             sharey=True, squeeze=False)
    for ax, (gen, sc, res) in zip(axes[0], panels):
        a = res[f"a1__{sc}"]
        m = a["raw"]["marginal"]
        ci = m.get("ci") or [float("nan")] * 2
        if all(_finite(c) for c in ci):
            ax.axvspan(pct(ci[0]), pct(ci[1]), color=GRID, zorder=0, linewidth=0)
        ax.axvline(pct(m["mean_delta"]), color=INK, linewidth=1.5, zorder=1)
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        for x in (-10, 10):
            ax.axvline(x, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)), zorder=1)
        for i, c in enumerate(countries):
            pools = [pct(p["raw"]["delta"]) for p in a["pools"]
                     if p["country"] == c and not p["raw"].get("non_estimable")]
            ys = [i + 0.18 * ((j % 5) - 2) / 2 for j in range(len(pools))]
            ax.scatter(pools, ys, s=10, color=MUTED, alpha=0.55, linewidths=0, zorder=2)
            cm = m["country_means"].get(c)
            if _finite(cm):
                ax.scatter([pct(cm)], [i], s=36, color=INK, edgecolors="white",
                           linewidths=1.2, zorder=3)
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries)
        ax.set_title(f"{gen} · {sc}", fontsize=9, color=INK)
        ax.set_xlabel("Change in dish diversity, top-k vs random-k (%)", fontsize=8, color=INK)
        _style(ax)
    axes[0][0].invert_yaxis()                 # once: the y axis is shared
    fig.text(0.01, -0.04, "Dots: pools. Circles: template-balanced country means. Line and band: "
             "equal-weight marginal and 95% CI. Dotted: ±10% equivalence margin.",
             fontsize=7, color=MUTED)
    _save(fig, outdir, "fig_country_delta")


def _curve_points(curves, fam):
    pts = []
    for key, v in curves.items():
        f, _, param = key.partition("|")
        if f != fam:
            continue
        d, g = v["delta"].get("mean"), v["gain_judge"].get("mean")
        if _finite(d) and _finite(g):
            pts.append((pct(d), g, param))
    # Grid order is the order of selector strength (weakest diversity push first).
    return pts


def fig_frontier(results, outdir):
    panels = [(g, s, r) for g, s, r in cells(results) if r.get(f"a3__{s}")]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 3.4), squeeze=False,
                             layout="constrained")
    for ax, (gen, sc, res) in zip(axes[0], panels):
        a = res[f"a3__{sc}"]
        curves, chosen = a["curves"], a["chosen_on_tuning"]
        for fam in ("stk", "mmr", "dpp"):
            pts = _curve_points(curves, fam)
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, color=SERIES[fam], linewidth=1.5, marker="o", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3, label=SERIES_NAME[fam])
            ch = chosen.get(fam, {})
            ev = ch.get("evaluation")
            if ev:
                cx, cy = pct(ev["delta"]["mean"]), ev["gain_judge"]["mean"]
                ci = ev["delta"].get("ci")
                if ci and all(_finite(c) for c in ci):
                    ax.plot([pct(ci[0]), pct(ci[1])], [cy, cy], color=SERIES[fam],
                            linewidth=1.0, alpha=0.6, zorder=2)
                ax.scatter([cx], [cy], s=70, facecolors="none", edgecolors=SERIES[fam],
                           linewidths=1.5, zorder=4)
            # Direct label at the strongest setting, the end of each curve.
            ax.annotate(fam.upper(), (xs[-1], ys[-1]), xytext=(0, -9),
                        textcoords="offset points", fontsize=7.5, color=INK, ha="center",
                        va="top")
        tk = curves.get("topk|None")
        if tk:
            x, y = pct(tk["delta"]["mean"]), tk["gain_judge"]["mean"]
            ax.scatter([x], [y], s=40, marker="s", color=INK, zorder=5)
            ax.annotate("top-k", (x, y), xytext=(0, 7), textcoords="offset points",
                        fontsize=7.5, color=INK, ha="center")
        ax.scatter([0], [0], s=40, marker="D", color=MUTED, zorder=5)
        ax.annotate("random-k", (0, 0), xytext=(5, 3), textcoords="offset points",
                    fontsize=7.5, color=MUTED)
        ax.axvline(-10, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)))
        ax.margins(x=0.08, y=0.1)
        ax.set_title(f"{gen} · select {a['selector']}, judge {a['judge']}", fontsize=9, color=INK)
        ax.set_xlabel("Diversity change vs random-k (%)", fontsize=8, color=INK)
        ax.set_ylabel(f"Held-out judge gain, SD units ({a['judge']})", fontsize=8, color=INK)
        _style(ax)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    fig.text(0.01, -0.03, "Rings: configuration chosen on tuning pools, with its 95% CI on "
             "diversity. Dotted: −10% margin.", fontsize=7, color=MUTED)
    _save(fig, outdir, "fig_frontier")


def fig_r1_quartile(results, outdir):
    panels = [(g, s, r) for g, s, r in cells(results) if r.get(f"r1__{s}")]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(2.8 * len(panels), 2.9),
                             sharey=True, squeeze=False, layout="constrained")
    for ax, (gen, sc, res) in zip(axes[0], panels):
        q = res[f"r1__{sc}"]["tagger_accuracy_by_score_quartile"]
        for name, color, dx in (("unfiltered", UNFILTERED, -0.06), ("verified", VERIFIED, 0.06)):
            acc = [100 * x if _finite(x) else float("nan") for x in q[name]["accuracy_by_bin"]]
            xs = [i + 1 + dx for i in range(len(acc))]
            ax.plot(xs, acc, color=color, linewidth=1.5, marker="o", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8, label=name)
            ax.annotate(name, (xs[-1], acc[-1]), xytext=(5, 0), textcoords="offset points",
                        fontsize=7.5, color=INK, va="center")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlim(0.6, 4.9)
        ax.set_title(f"{gen} · {sc}", fontsize=9, color=INK)
        ax.set_xlabel(f"{sc} score quartile", fontsize=8, color=INK)
        _style(ax)
    axes[0][0].set_ylabel("Tagger accuracy on named prompts (%)", fontsize=8, color=INK)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=2, frameon=False, fontsize=8)
    _save(fig, outdir, "fig_r1_quartile")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", default=os.path.join(HERE, "results", "real"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    outdir = args.out or os.path.join(args.indir, "report")
    os.makedirs(outdir, exist_ok=True)
    results = load(args.indir)
    write_tables(results, outdir)
    fig_country_delta(results, outdir)
    fig_frontier(results, outdir)
    fig_r1_quartile(results, outdir)
    print(f"wrote {outdir} ({', '.join(sorted(results))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
