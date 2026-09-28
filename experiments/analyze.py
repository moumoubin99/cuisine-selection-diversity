#!/usr/bin/env python3
"""CPU analysis of a real run, replayed from the store through `run_audit`.

    python analyze.py --root /root/autodl-tmp/store --out results/real

Per generator and selecting scorer:

  A1/A2/A4  `run_audit` on the main pools: raw and conditional endpoints,
            country marginal, TOST at the 10% margin, Welch + simultaneous
            country intervals and the A4 decision rule.
  R1        the same, on the KNOWN-LABEL pools with `audit=True`: the tagger's
            contrast against the contrast on the prompted labels, gap tested
            for equivalence at 5%; tagger accuracy by within-pool score
            quartile.  Verifier-filtered is primary; unfiltered is reported.
  REP       stereotype repetition: do independent pools of one country select
            the SAME dishes more than random selection would?
  A3        the frontier (src/frontier.py), hyper-parameters frozen on the
            tuning pools, judged by the held-out scorer.

Nothing here loads a model.  Everything the GPU produced is read through
`src/store.py`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.frontier import SELECTOR_GRID, choose_on_tuning, run_selectors
from src.known_label import tagger_accuracy_by_score_quantile
from src.labels import normalize
from src.pipeline import TAGGER_ERROR, run_audit
from src.selectors import random_selector, select_topk
from src.store import (CachedGenerator, CachedScorer, CachedTagger, _main_prompts,
                       gen_tag, known_label_prompt_list, load_embeddings,
                       load_vocab_file, read_jsonl, reader_file, tune_prompt_list)

HERE = os.path.dirname(os.path.abspath(__file__))
SCORERS = ("pickscore", "imagereward")
# Amendment 7: aesthetic / preference scorers added as selectors only.  They
# are judged by each registered scorer (both are held out from them); the
# registered A3 pairs stay PickScore <-> ImageReward.
EXTRA_SCORERS = ("laion_aes", "hpsv21")


def _jsonable(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k))): _jsonable(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


class Cell:
    """Everything replayable for one generator."""

    def __init__(self, root, cfg, tag, full_vocab, reader="qwen", scorers=SCORERS):
        self.root, self.cfg, self.tag, self.reader = root, cfg, tag, reader
        self.full_vocab = full_vocab
        self.gen = CachedGenerator(root, tag, known_keep="verified", reader=reader)
        self.gen_all = CachedGenerator(root, tag, known_keep="all", reader=reader)
        # By country, so the matcher's demonym rule can check the country.
        self.tagger = CachedTagger(root, tag, full_vocab, reader=reader)
        self.scorers = {s: CachedScorer(root, tag, s) for s in scorers
                        if os.path.exists(os.path.join(root, "scores", f"{tag}__{s}.jsonl"))}
        p = os.path.join(root, "embed", f"{tag}__dinov2.npz")
        self.emb = load_embeddings(root, tag) if os.path.exists(p) else None

    def pool(self, prompt, pool_seed):
        ims = self.gen.generate(prompt.text, self.cfg["n"], pool_seed, prompt.country)
        tags = self.tagger.tag(ims)
        labels = [None if t[2] == TAGGER_ERROR else t[2] for t in tags]
        return ims, labels


def _strip(out):
    return {k: v for k, v in out.items() if k != "rows"}


def _pool_table(out):
    keep = ("prompt_id", "country", "template", "pool_seed", "status_counts",
            "unresolved_rate", "authenticity_deficit", "machine_minus_audit")
    rows = []
    for r in out["rows"]:
        d = {k: r[k] for k in keep if k in r}
        for ep in ("raw", "conditional", "audit"):
            if ep in r:
                d[ep] = {k: r[ep].get(k) for k in ("delta", "non_estimable", "reason",
                                                   "n_kept_selected", "kept_shift",
                                                   "null_insufficient_frac",
                                                   "delta_bounds")}
        if "dose_response" in r:
            d["dose_response"] = r["dose_response"]
        if "dose_response_matched" in r:
            d["dose_response_matched"] = r["dose_response_matched"]
        if "stereotype_shift" in r:
            d["stereotype_shift"] = r["stereotype_shift"]
            d["stereotype_shift_permutation_null"] = r.get("stereotype_shift_permutation_null")
        rows.append(d)
    return rows


def dose_summary(rows, n_boot=2000, seed=11):
    """Selection fraction (FINAL_PROPOSAL: secondary k = 32), descriptive.

    Uses `dose_response_matched`: every k, the primary one included, on the
    primary contrast's estimand (log selected minus mean log null) with the
    same null count.  Per k, the mean over estimable pools with a percentile
    interval over pools; and the paired per-pool difference of each secondary
    k from the primary k, over pools estimable at both.  Experiment audit
    run 02 found the earlier summary compared k = 32 on a different estimand
    (and 60 null draws) with the k = 16 primary contrast."""
    by_pool = []
    for r in rows:
        dr = r.get("dose_response_matched")
        if dr:
            by_pool.append({int(d["k"]): (float(d["log_contrast"])
                                         if not d.get("non_estimable") else float("nan"))
                            for d in dr})
    if not by_pool:
        return {}
    ks = sorted({k for p in by_pool for k in p})
    k0 = ks[0]
    rng = np.random.default_rng(seed)

    def boot(x):
        bs = x[rng.integers(0, x.size, (n_boot, x.size))].mean(axis=1)
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    out = {}
    for k in ks:
        x = np.asarray([p.get(k, np.nan) for p in by_pool])
        x = x[np.isfinite(x)]
        row = {"k": k, "n_pools": len(by_pool), "n_estimable": int(x.size),
               "estimand": "log selected - mean log null (random-k)"}
        if x.size >= 2:
            row.update(mean_log_contrast=float(x.mean()), ci=boot(x))
        if k != k0:
            dx = np.asarray([p.get(k, np.nan) - p.get(k0, np.nan) for p in by_pool])
            dx = dx[np.isfinite(dx)]
            row["vs_primary_k"] = {"k_ref": k0, "n_pools": int(dx.size)}
            if dx.size >= 2:
                row["vs_primary_k"].update(mean_diff=float(dx.mean()), ci=boot(dx))
        out[str(k)] = row
    return out


def a1(cell, scorer, proto, n_null):
    cfg = cell.cfg
    out = run_audit(cell.gen, cell.scorers[scorer], cell.tagger, _main_prompts(cfg),
                    cell.full_vocab, n=cfg["n"], k=cfg["k"], m_auth=cfg["m_auth"],
                    n_pools=cfg["n_pools"], prototypicality_by_country=proto,
                    seed=cfg["seed"], n_null=n_null,
                    secondary_ks=cfg.get("secondary_k") or (),
                    margin_proportional=cfg["analysis"]["margin_proportional"])
    res = {**_strip(out), "pools": _pool_table(out)}
    res["dose_response_summary"] = dose_summary(out["rows"])
    for ep in ("raw", "conditional"):
        if ep in out:
            res[ep]["a2"] = a2_rule(out[ep])
    return res


A2_MIN_SPREAD = float(-np.log(0.9))     # PREREG_AMENDMENT.md section 4
ALPHA = 0.05


def a2_rule(ep):
    """A2: the A4 decision rule says heterogeneous AND the spread of country
    mean deltas reaches the A1 margin, on the COMPLETE eight-country,
    two-template design.

    Round-3 code review, P1: seven surviving countries could produce a claim
    about eight, and an observed spread of 0.106 does not establish that the
    underlying spread reaches 0.105.  The confirmatory rule stays the
    pre-registered observed-spread rule; `spread_lcb` is the lower bound implied
    by the simultaneous country intervals (theta_a - theta_b >= lo_a - hi_b for
    every pair, jointly at 95%), and only `spread_lcb >= 0.105` licenses the
    wording "a spread of at least 10%".
    """
    cm = ep["equivalence"].get("country_means") or {}
    vals = [v for v in cm.values() if np.isfinite(v)]
    spread = float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
    verdict = ep["heterogeneity_decision"].get("verdict")
    complete = bool((ep.get("design") or {}).get("complete_design", False))
    iv = (ep.get("country_intervals") or {}).get("intervals") or {}
    lcb = (float(max(iv[a][0] - iv[b][1] for a in iv for b in iv if a != b))
           if len(iv) >= 2 else float("nan"))
    if not complete:
        status = "non_estimable_incomplete_design"
    elif verdict == "non_estimable":
        # Round-6 code review, P1: Welch can be estimable while the
        # simultaneous intervals are not; A2 is then unavailable and enters
        # Holm at p = 1 rather than lending A3 a smaller multiplier.
        status = "non_estimable_heterogeneity_decision"
    elif verdict == "heterogeneous" and spread >= A2_MIN_SPREAD:
        status = ("heterogeneous_spread_established" if lcb >= A2_MIN_SPREAD
                  else "heterogeneous_spread_size_not_established")
    else:
        status = "heterogeneity_not_established"
    return {"spread": spread, "spread_lcb": lcb, "decision_rule": verdict,
            "complete_design": complete, "status": status,
            "claim": bool(complete and verdict == "heterogeneous"
                          and spread >= A2_MIN_SPREAD),
            "welch_p": ep["heterogeneity"].get("p_value"),
            "holm_p": _a2_holm_p(ep["heterogeneity"])}


def _a2_holm_p(het):
    """Round-6 code review, P2: Welch's F is liberal under skewed pool deltas
    (measured with 4 pools per template cell, 8 countries, 1,500 replicates:
    5.3% null rejection with normal errors, 13.1% with Beta(1,5)).  The
    within-cell bootstrap with re-estimated variances held 0.8% and 1.7%, so
    A2 enters Holm with max(Welch p, conservative bootstrap p) -- valid in both
    cases, at a measured loss of power (0.24 -> 0.05 at a 0.25 shift in one
    country).  Round-7 code review, P2: a result without both p-values is
    unavailable (None -> p = 1 in Holm) rather than falling back to Welch."""
    pw = het.get("p_value")
    pb = het.get("p_value_bootstrap_conservative")
    if pw is None or pb is None or not (np.isfinite(pw) and np.isfinite(pb)):
        return None
    return float(max(pw, pb))


def holm(pvals):
    """Holm step-down; returns adjusted p-values in input order.  A missing
    (non-finite) p-value counts as 1: the family is fixed by the
    pre-registration, and dropping an untestable hypothesis would shrink the
    multiplier for the others (round-3 code review, P2)."""
    p = np.asarray(pvals, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(p, kind="stable")
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def r1_disposition(res, scorer):
    """The pre-registered R1 disposition of the known-label gap (verified).

    equivalent    -> A1 may be claimed
    harmful       -> stop rule 4: the machine endpoint exaggerates the loss by
                     more than the 5% margin; the paper is a measurement paper
    inconclusive  -> A1 is reported as unvalidated at 5%, not claimed
    missing / non_estimable -> not claimed

    Only the VERIFIED gap decides (amendment 4, round-5 code review, P0).
    Amendment 3 let the unfiltered gap decide when the verified one was
    non-estimable, calling it the stricter check.  It is not: on images the
    verifier rejects the prompted label may be wrong, and score-dependent
    errors there can offset errors on kept images, so an `equivalent`
    unfiltered gap does not validate the tagger on the images the verified gap
    is about.  The unfiltered gap is reported as a sensitivity analysis
    (`r1_sensitivity`) and never clears A1.

    Returns (verdict, reason, basis).
    """
    r = res.get(f"r1__{scorer}")
    if not r:
        return "missing", "no known-label analysis in this output", None
    g = (r.get("verified") or {}).get("gap") or {}
    return g.get("verdict", "non_estimable"), g.get("reason"), "verified"


def r1_sensitivity(res, scorer):
    """The unfiltered known-label gap: reported beside A1, never decisive."""
    u = (((res.get(f"r1__{scorer}") or {}).get("unfiltered") or {}).get("gap")
         or {})
    return {"verdict": u.get("verdict", "missing"), "mean": u.get("mean"),
            "ci": u.get("ci"), "reason": u.get("reason")}


def claims(res, scorer="pickscore", endpoint="raw", family="stk"):
    """The three confirmatory claims of one cell, Holm-corrected over a FIXED
    family of three.

    A1: one-sided t test that the equal-weight country mean delta is below
        log 0.9, from the TOST moments; claimed only if the verdict is
        `harmful`, Holm p < .05 AND the known-label gap is `equivalent`.
    A2: the Welch omnibus p, claimed only with the spread rule on the complete
        design (`a2_rule`).
    A3: one-sided fixed-country t test that the frozen selector's paired delta
        gain over top-k is <= 0; claimed only if Holm p < .05, the 95% interval
        lies above 0, top-k's held-out judge gain is positive (its lower
        bound), and the selector keeps >= 90% of it.

    Every withheld claim says why.
    """
    from src.analysis import _student_t_cdf
    a1r = (res.get(f"a1__{scorer}") or {}).get(endpoint)
    a3r = ((res.get(f"a3__{scorer}") or {}).get("chosen_on_tuning") or {}).get(family)
    if a1r is None:
        return None
    eq = a1r["equivalence"]
    # Round-5 code review, P0: a family member that is not available enters
    # Holm at p = 1.  `run_audit` keeps the observed-only mean and SE of an
    # incomplete design (labelled `equivalence_on_observed`), and testing it
    # here gave A3 a smaller Holm multiplier on an A1 that cannot be claimed.
    p1 = (float(_student_t_cdf((eq["mean_delta"] - np.log(0.9)) / eq["se"], eq["df"]))
          if eq.get("se") and eq.get("verdict") != "non_estimable"
          else float("nan"))
    a2 = a1r.get("a2") or a2_rule(a1r)
    p2a = a2["holm_p"] if "holm_p" in a2 else a2.get("welch_p")
    p2 = (p2a if p2a is not None
          and not str(a2.get("status", "")).startswith("non_estimable")
          else float("nan"))
    vs = (a3r or {}).get("vs_topk") or {}
    d3 = vs.get("delta") or {}
    p3 = (float("nan") if d3.get("non_estimable")
          else d3.get("p_le_0", float("nan")))
    adj = holm([p1, p2, p3])

    r1v, r1why, r1basis = r1_disposition(res, scorer)
    a1_why = []
    if eq["verdict"] != "harmful":
        a1_why.append(f"A1 verdict is {eq['verdict']}")
    if not adj[0] < ALPHA:
        a1_why.append(f"Holm p = {adj[0]:.3g}")
    if r1v != "equivalent":
        a1_why.append(f"known-label gap is {r1v}"
                      + (" (stop rule 4)" if r1v == "harmful" else "")
                      + (f": {r1why}" if r1why else ""))
    a1_status = ("claimed" if not a1_why else
                 "stop_rule_4" if r1v == "harmful" else
                 "unvalidated_at_5pct" if (eq["verdict"] == "harmful" and adj[0] < ALPHA
                                           and r1v == "inconclusive")
                 else "not_claimed")

    a2_why = [] if a2["claim"] else [a2["status"]]
    if not adj[1] < ALPHA:
        a2_why.append(f"Holm p = {adj[1]:.3g}")

    a3 = None
    if a3r:
        ci = d3.get("ci") or [float("nan")] * 2
        tk = vs.get("topk_judge_gain") or {}
        ret = vs.get("judge_gain_retained", float("nan"))
        a3_why = []
        if d3.get("non_estimable"):
            a3_why.append(f"paired delta non-estimable: {d3.get('reason')}")
        if not adj[2] < ALPHA:
            a3_why.append(f"Holm p = {adj[2]:.3g}")
        if not (np.isfinite(ci[0]) and ci[0] > 0):
            a3_why.append(f"95% CI lower bound {ci[0]:.3g} is not > 0")
        if not (np.isfinite(tk.get("ci_lo", np.nan)) and tk["ci_lo"] > 0):
            a3_why.append("top-k's held-out judge gain is not positive, so a "
                          "retained share is not interpretable")
        if not (np.isfinite(ret) and ret >= 0.9):
            a3_why.append(f"judge gain retained {ret:.3g} < 0.9")
        a3 = {"family": family, "param": a3r["param"], "delta_gain": d3,
              "topk_judge_gain": tk, "judge_gain_retained": ret,
              "p_holm": adj[2], "claim": not a3_why, "withheld_because": a3_why}
    return {
        "cell": f"{scorer} x {res['generator']} x {endpoint}",
        "holm_family": ["A1", "A2", "A3"], "p_raw": [p1, p2, p3],
        "A1": {"verdict": eq["verdict"], "p_loss_beyond_margin": p1, "p_holm": adj[0],
               "known_label_gap": r1v, "known_label_gap_basis": r1basis,
               "known_label_gap_unfiltered_sensitivity": r1_sensitivity(res, scorer),
               "status": a1_status,
               "claim": not a1_why, "withheld_because": a1_why},
        "A2": {**a2, "p_holm": adj[1], "claim": not a2_why, "withheld_because": a2_why},
        "A3": a3 if a3 is not None else {
            "claim": False, "p_holm": adj[2],
            "withheld_because": ["no frontier result for this cell"]},
    }


def r1(cell, scorer, n_null):
    """The known-label audit: tagger contrast vs prompted-label contrast."""
    cfg = cell.cfg
    kl = cfg["known_label"]
    res = {}
    for mode, gen in (("verified", cell.gen), ("unfiltered", cell.gen_all)):
        out = run_audit(gen, cell.scorers[scorer], cell.tagger, known_label_prompt_list(cfg),
                        cell.full_vocab, n=cfg["n"], k=cfg["k"], m_auth=cfg["m_auth"],
                        n_pools=kl["pools_per_country"], seed=kl["seed_base"],
                        n_null=n_null, audit=True)
        res[mode] = {
            "raw_mean_delta": out.get("raw", {}).get("mean_delta"),
            "audit_mean_delta": out.get("audit", {}).get("mean_delta"),
            "audit_equivalence": out.get("audit", {}).get("equivalence"),
            # machine minus known-label contrast on ONE image set (tagger-
            # resolved AND verifier-kept), TOST at the 5% margin on the fixed
            # eight-country marginal (`pipeline._gap_test`).
            "gap": out.get("machine_minus_audit", {"verdict": "non_estimable"}),
            # the endpoint-level difference, each side on its own images
            "gap_unmatched": out.get("machine_minus_audit_unmatched",
                                     {"verdict": "non_estimable"}),
            "audit_sources": sorted({r.get("audit_source") for r in out["rows"]
                                     if r.get("audit_source")}),
            "pools": _pool_table(out),
        }
    # Tagger accuracy vs within-pool score rank, on verifier-kept images.
    correct, ranks, vkeep, pool_id = [], [], [], []
    wd_keep, wd_rank, wd_pool, wd_prank = [], [], [], []
    for i, p in enumerate(known_label_prompt_list(cfg)):
        for j in range(kl["pools_per_country"]):
            seed = kl["seed_base"] + 1000 * i + j
            ims_all = cell.gen_all.generate(p.text, cfg["n"], seed, p.country)
            ims_ver = cell.gen.generate(p.text, cfg["n"], seed, p.country)
            tags = cell.tagger.tag(ims_all)
            s = cell.scorers[scorer].score(ims_all, p.text)
            r = np.argsort(np.argsort(s, kind="stable"), kind="stable") / float(len(s))
            # Amendment 6: ranks within each prompted dish of the pool, so a
            # dish that is both prettier and harder to recognise cannot make
            # the score-faithfulness association by itself.
            by_dish = {}
            for t_i, im in enumerate(ims_all):
                by_dish.setdefault(im.artifact, []).append(t_i)
            for idx in by_dish.values():
                if len(idx) < 4:
                    continue
                sd = s[idx]
                rd = np.argsort(np.argsort(sd, kind="stable"), kind="stable") / float(len(sd))
                for t_i, rr_d in zip(idx, rd):
                    wd_keep.append(ims_ver[t_i].artifact is not None)
                    wd_rank.append(rr_d)
                    wd_pool.append(seed)
                    wd_prank.append(r[t_i])
            m = cell.tagger.matcher
            for im, iv, t, rr in zip(ims_all, ims_ver, tags, r):
                truth, _ = m.match(im.artifact)
                got = None if t[2] == TAGGER_ERROR else t[2]
                correct.append(got is not None and normalize(got) == normalize(truth))
                ranks.append(rr)
                vkeep.append(iv.artifact is not None)
                pool_id.append(seed)
    res["tagger_accuracy_by_score_quartile"] = {
        "verified": tagger_accuracy_by_score_quantile(correct, ranks, keep=vkeep,
                                                      groups=pool_id),
        "unfiltered": tagger_accuracy_by_score_quantile(correct, ranks, groups=pool_id),
    }
    res["verifier_keep_rate"] = float(np.mean(vkeep))
    res["verifier_diagnostics"] = verifier_diagnostics(cell)
    res["tagger_diagnostics"] = tagger_diagnostics(cell.tagger)
    res["verifier_keep_by_score_quartile"] = tagger_accuracy_by_score_quantile(
        vkeep, ranks, groups=pool_id)
    res["verifier_keep_by_score_quartile_within_dish"] = (
        {**tagger_accuracy_by_score_quantile(wd_keep, wd_rank, groups=wd_pool),
         "n_images": len(wd_keep)} if wd_keep else None)
    # Post hoc (after the within-dish result was seen; amendment 6 log): the
    # ordinary within-POOL ranks on the same images, so a within-dish null
    # cannot be an artefact of which images have a dish group of >= 4.
    res["verifier_keep_by_score_quartile_within_dish_subset_pool_rank"] = (
        {**tagger_accuracy_by_score_quantile(wd_keep, wd_prank, groups=wd_pool),
         "n_images": len(wd_keep)} if wd_keep else None)
    return res


def _has_non_latin_letter(s):
    return any(ch.isalpha() and ord(ch) >= 0x250 for ch in s or "")


def tagger_diagnostics(tagger):
    """Over every stored tag (main, tuning and known-label pools): the share of
    open answers that resolve to a CSpace label, and the share written in a
    non-Latin script.  Native-script answers are kept but not romanised
    (round-3 code review, P1 #8), so a native-script answer and the romanised
    CSpace name of the same dish count as two labels; this share bounds how
    many images that can affect."""
    ids = list(tagger._r)
    if not ids:
        return None
    from src.labels import parse_tagger_json
    native = resolved = 0
    for i in ids:
        r = tagger._r[i]
        dish = parse_tagger_json(r["reply"])[0] if r.get("reply") else r.get("dish")
        native += _has_non_latin_letter(dish)
        resolved += tagger.tag_one(i)[2] != TAGGER_ERROR
    return {"n": len(ids), "resolved_rate": resolved / len(ids),
            "in_vocab_rate": float(np.mean([bool(tagger.in_vocab[i]) for i in ids])),
            "native_script_rate": native / len(ids)}


def verifier_diagnostics(cell):
    """How discriminating the verifier is, on its own records: the yes/no
    probability for the prompted dish vs a same-pool decoy (AUC; 0.5 = no
    discrimination), and the forced-choice outcome distribution."""
    rows = read_jsonl(reader_file(cell.root, "verify", cell.tag, cell.reader))
    rows = [r for r in rows if "p_yes_decoy" in r]
    if not rows:
        return None
    t = np.array([r["p_yes"] for r in rows])
    d = np.array([r["p_yes_decoy"] for r in rows])
    auc = float(np.mean((t[:, None] > d[None, :]) + 0.5 * (t[:, None] == d[None, :])))
    chosen = [r.get("chosen") for r in rows]
    none = sum(1 for r, c in zip(rows, chosen) if c == len(r.get("options", [])) - 1)
    return {"n": len(rows), "yes_rate_true": float(np.mean(t >= .5)),
            "yes_rate_decoy": float(np.mean(d >= .5)), "auc_true_vs_decoy": auc,
            "forced_choice_keep_rate": float(np.mean([r.get("kept", False) for r in rows])),
            "forced_choice_none_rate": none / len(rows)}


def repetition(cell, scorer, n_rand=200, seed=0):
    """Cross-pool stereotype repetition.  DESCRIPTIVE ONLY.

    For each country, the mean pairwise Jaccard similarity between the DISTINCT
    dish sets selected from independent pools, top-k minus random-k.  Positive
    means aesthetic selection keeps returning the same dishes for a country
    across independent draws, beyond what the generator's own repetition
    produces.  Also the share of selected images carrying the country's modal
    dish (its single most frequent tagger label across all its pools).

    Jaccard moves with set cardinality: smaller selected dish sets overlap
    differently for purely mechanical reasons, so top-k minus random-k overlap
    partly restates the diversity loss itself rather than isolating a separate
    repetition mechanism (round-3 code review, P2).  The mean distinct-set size
    and the number of resolved images in each selected set are therefore
    reported beside it, and neither measure is confirmatory.
    """
    cfg = cell.cfg
    rng = np.random.default_rng(seed)
    k = cfg["k"]
    by_c = {}
    for i, p in enumerate(_main_prompts(cfg)):
        for j in range(cfg["n_pools"]):
            ims, labels = cell.pool(p, cfg["seed"] + 1000 * i + j)
            s = cell.scorers[scorer].score(ims, p.text)
            by_c.setdefault(p.country, []).append((labels, s))
    out = {}
    for c, pools in by_c.items():
        allab = [l for labels, _ in pools for l in labels if l is not None]
        vals, cnt = np.unique(np.asarray(allab, dtype=object), return_counts=True)
        mode = vals[np.argmax(cnt)] if len(vals) else None

        def sets(labels, idx):
            return {labels[t] for t in idx if labels[t] is not None}

        top_idx = [select_topk(s, k) for _, s in pools]
        top = [sets(l, t) for (l, _), t in zip(pools, top_idx)]
        rand_idx = [[random_selector(len(l), k, rng) for _ in range(n_rand)]
                    for l, _ in pools]
        rand = [[sets(l, r) for r in ri] for (l, _), ri in zip(pools, rand_idx)]
        n_res_top = [sum(l[t] is not None for t in ti) for (l, _), ti in zip(pools, top_idx)]
        n_res_rand = [np.mean([sum(l[t] is not None for t in r) for r in ri])
                      for (l, _), ri in zip(pools, rand_idx)]

        def jac(a, b):
            return len(a & b) / len(a | b) if (a | b) else float("nan")

        pairs = list(itertools.combinations(range(len(pools)), 2))
        jt = np.nanmean([jac(top[a], top[b]) for a, b in pairs])
        jr = np.nanmean([np.nanmean([jac(rand[a][r], rand[b][r]) for r in range(n_rand)])
                         for a, b in pairs])
        # Amendment 6: both sets of a pool subsampled to the smaller of the
        # two sizes, per random draw, so set cardinality cannot drive the gap.
        mrng = np.random.default_rng(seed + 1)

        def sub(x, d):
            x = sorted(x, key=str)
            return set(mrng.choice(np.asarray(x, dtype=object), d, replace=False)) if d else set()

        jm = []
        for r in range(n_rand):
            d = [min(len(top[a]), len(rand[a][r])) for a in range(len(pools))]
            ts = [sub(top[a], d[a]) for a in range(len(pools))]
            rs = [sub(rand[a][r], d[a]) for a in range(len(pools))]
            jm.append(np.nanmean([jac(ts[a], ts[b]) for a, b in pairs])
                      - np.nanmean([jac(rs[a], rs[b]) for a, b in pairs]))
        head_top = np.mean([np.mean([l[t] == mode for t in select_topk(s, k)]) for l, s in pools])
        head_pool = np.mean([np.mean([x == mode for x in l]) for l, _ in pools])
        out[c] = {"jaccard_topk": float(jt), "jaccard_random": float(jr),
                  "repetition_excess": float(jt - jr), "modal_dish": mode,
                  "repetition_excess_matched": float(np.nanmean(jm)),
                  "modal_share_topk": float(head_top), "modal_share_pool": float(head_pool),
                  "modal_share_excess": float(head_top - head_pool),
                  "mean_set_size_topk": float(np.mean([len(x) for x in top])),
                  "mean_set_size_random": float(np.mean([len(x) for r in rand for x in r])),
                  "mean_resolved_topk": float(np.mean(n_res_top)),
                  "mean_resolved_random": float(np.mean(n_res_rand)),
                  "n_pools": len(pools)}
    ex = np.array([v["repetition_excess"] for v in out.values()])
    hx = np.array([v["modal_share_excess"] for v in out.values()])
    mx = np.array([v["repetition_excess_matched"] for v in out.values()])
    # Countries are fixed by design; the interval is a t interval over the
    # eight country values, reported as descriptive, not confirmatory.
    def tci(x):
        from src.analysis import _student_t_ppf
        x = x[np.isfinite(x)]
        if x.size < 2:
            return [float("nan")] * 2
        h = _student_t_ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
        return [float(x.mean() - h), float(x.mean() + h)]
    return {"by_country": out, "status": "descriptive",
            "mean_repetition_excess": float(np.nanmean(ex)), "repetition_ci": tci(ex),
            "mean_modal_share_excess": float(np.nanmean(hx)), "modal_share_ci": tci(hx),
            "n_countries_positive": int(np.sum(ex > 0)),
            "mean_repetition_excess_matched": float(np.nanmean(mx)),
            "repetition_matched_ci": tci(mx),
            "n_countries_positive_matched": int(np.sum(mx > 0))}


def frontier(cell, sel, judge, n_null):
    cfg = cell.cfg
    k, m = cfg["k"], cfg["m_auth"]

    def rows_for(prompts, seed_base, n_pools):
        rows = []
        for i, p in enumerate(prompts):
            for j in range(n_pools):
                seed = seed_base + 1000 * i + j
                ims, labels = cell.pool(p, seed)
                keep = [l is not None for l in labels]
                E = np.stack([cell.emb[im.image_id] for im in ims])
                s = cell.scorers[sel].score(ims, p.text)
                jv = cell.scorers[judge].score(ims, p.text)
                r = run_selectors([l or "" for l in labels], keep, E, s, jv, k, m, n_null, seed)
                r["_country"], r["_template"] = p.country, p.template
                rows.append(r)
        return rows

    tu = cfg["tune"]
    tune = rows_for(tune_prompt_list(cfg), tu["seed_base"], tu["pools_per_country"])
    chosen = choose_on_tuning(tune)
    main = rows_for(_main_prompts(cfg), cfg["seed"], cfg["n_pools"])

    cells = sorted({(r["_country"], r["_template"]) for r in main})

    def by_cell(values_of):
        """{(country, template): finite per-pool values}; the design is FIXED."""
        out = {c: [] for c in cells}
        for r in main:
            v = values_of(r)
            if np.isfinite(v):
                out[(r["_country"], r["_template"])].append(float(v))
        return {c: np.asarray(v) for c, v in out.items()}

    def fixed_design(values_of, seed):
        """Equal-weight country mean with a country x template stratified
        bootstrap, and the fixed-country t moments as the primary test.

        Round-3 code review, P1: resampling all of a country's pools together
        randomised the A/B template proportions, and `nanmean` silently dropped
        a country with no estimable value.  Now every (country, template) cell
        must hold >= 2 estimable pools, or the result is non-estimable.  The
        test is the same fixed-country t construction as A1 (its calibration
        was simulated in round 2: 4.42% at a nominal 5%); the percentile
        bootstrap is reported beside it, uncalibrated at this pool count.
        """
        from src.analysis import _fixed_country_moments, _student_t_cdf, _student_t_ppf
        bc = by_cell(values_of)
        thin = [f"{c}/{t}" for (c, t), v in bc.items() if v.size < 2]
        if thin:
            return {"non_estimable": True, "mean": float("nan"),
                    "ci": [float("nan")] * 2, "p_le_0": float("nan"),
                    "reason": "cells with < 2 estimable pools: " + ", ".join(thin)}
        by_country = {}
        for (c, t), v in bc.items():
            by_country.setdefault(c, {})[t] = v
        # Country value = mean of its template means.  Round-4 code review:
        # concatenating a country's pools weighted its templates by how many
        # estimable pools each kept (0.337 reported vs 0.503 balanced on the
        # reviewer's example), although the design fixes them at equal counts.
        theta, se, df, _ = _fixed_country_moments(by_country)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(2000):
            boots.append(np.mean([np.mean([v[rng.integers(0, v.size, v.size)].mean()
                                           for v in vs.values()])
                                  for vs in by_country.values()]))
        out = {"non_estimable": False,
               "mean": float(np.mean([np.mean([v.mean() for v in vs.values()])
                                      for vs in by_country.values()])),
               "bootstrap_ci": [float(np.quantile(boots, .025)),
                                float(np.quantile(boots, .975))],
               "n_pools": int(sum(v.size for v in bc.values()))}
        if theta is None:
            return {**out, "non_estimable": True, "ci": [float("nan")] * 2,
                    "p_le_0": float("nan"),
                    "reason": "zero pool-level variance in every country"}
        h = _student_t_ppf(0.975, df) * se
        out.update({"ci": [float(theta - h), float(theta + h)], "se": se, "df": df,
                    "ci_lo": float(theta - h),
                    "p_le_0": float(1.0 - _student_t_cdf(theta / se, df))})
        return out

    def summarize(rows, key):
        assert rows is main
        return {metric: fixed_design(lambda r: r[key][metric], 1)
                for metric in ("delta", "emb_delta", "gain_selector", "gain_judge",
                               "raw_gain_judge")}

    def paired(key):
        """Per-pool (selector - top-k) on the same pools, so the difference is
        paired; fixed-design test and interval (`fixed_design`)."""
        out = {metric: fixed_design(
                   lambda r: r[key][metric] - r[("topk", None)][metric], 2)
               for metric in ("delta", "emb_delta", "gain_judge")}
        # Share of top-k's held-out quality gain the selector retains, only
        # meaningful when top-k's gain is itself positive (round-3 code review,
        # P0: a negative denominator let a worse selector pass the 90% rule).
        tk = fixed_design(lambda r: r[("topk", None)]["gain_judge"], 3)
        out["topk_judge_gain"] = tk
        out["raw_gain_judge"] = fixed_design(
            lambda r: r[key]["raw_gain_judge"] - r[("topk", None)]["raw_gain_judge"], 4)
        out["topk_raw_judge_gain"] = fixed_design(
            lambda r: r[("topk", None)]["raw_gain_judge"], 5)
        ok = (not tk["non_estimable"]) and tk.get("ci_lo", -1) > 0
        out["judge_gain_retained"] = (
            float((tk["mean"] + out["gain_judge"]["mean"]) / tk["mean"])
            if ok else float("nan"))
        out["judge_gain_retained_ci"] = retention_ci(key) if ok else [float("nan")] * 2
        return out

    def retention_ci(key, n_boot=2000, seed=7):
        """Amendment 6, descriptive: 95% percentile interval for the retained
        share, resampling pools within each (country, template) cell and
        keeping each pool's top-k and selector gains together (paired)."""
        cellv = {c: [] for c in cells}
        for r in main:
            t, v = r[("topk", None)]["gain_judge"], r[key]["gain_judge"]
            if np.isfinite(t) and np.isfinite(v):
                cellv[(r["_country"], r["_template"])].append((t, v))
        if any(len(v) < 2 for v in cellv.values()):
            return [float("nan")] * 2
        by_country = {}
        for (c, t), v in cellv.items():
            by_country.setdefault(c, []).append(np.asarray(v))
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            m = np.mean([np.mean([v[rng.integers(0, len(v), len(v))].mean(axis=0)
                                  for v in vs], axis=0) for vs in by_country.values()], axis=0)
            if m[0] > 0:
                boots.append(m[1] / m[0])
        if len(boots) < 0.9 * n_boot:
            return [float("nan")] * 2
        return [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]

    curves = {}
    for name, params in SELECTOR_GRID.items():
        for p in params:
            curves[f"{name}|{p}"] = summarize(main, (name, p))
    frozen = {name: (ch if ch.get("non_estimable") else
                     {**ch, "evaluation": summarize(main, (name, ch["param"])),
                      "vs_topk": paired((name, ch["param"]))})
              for name, ch in chosen.items()}
    # Post hoc, descriptive (after experiment audit run 02): the paired
    # difference from top-k for every grid point, so a configuration frozen on
    # one reader's tuning pools can be read under another reader's labels.
    # The selections do not depend on the reader; only `delta` does.
    curves_vs_topk = {
        f"{name}|{p}": {metric: fixed_design(
            lambda r, k=(name, p), mt=metric: r[k][mt] - r[("topk", None)][mt], 2)
            for metric in ("delta", "emb_delta")}
        for name, params in SELECTOR_GRID.items() for p in params if name != "topk"}
    return {"selector": sel, "judge": judge, "chosen_on_tuning": frozen, "curves": curves,
            "curves_vs_topk": curves_vs_topk,
            "n_tuning_pools": len(tune), "n_eval_pools": len(main),
            "design_cells": [list(c) for c in cells]}


def validate_frontier_config(cfg):
    """The frozen config must describe the grid the code runs (round-3 code
    review, P2: pilot.yaml still listed random/top-k/MMR and 'tune_on:
    pilot_pools' while the code ran STK and DPP on separate tuning pools)."""
    fr = cfg.get("frontier") or {}
    want = {"selectors": ["random"] + list(SELECTOR_GRID),
            "stk_clusters": SELECTOR_GRID["stk"], "mmr_lambda": SELECTOR_GRID["mmr"],
            "dpp_alpha": SELECTOR_GRID["dpp"], "tune_on": "tune_pools"}
    bad = {k: (fr.get(k), v) for k, v in want.items()
           if (sorted(map(str, fr.get(k))) if isinstance(v, list) and fr.get(k) is not None
               else fr.get(k)) != (sorted(map(str, v)) if isinstance(v, list) else v)}
    if bad:
        raise ValueError("configs frontier block does not match src/frontier.py "
                         "SELECTOR_GRID: " + "; ".join(f"{k}: config {a!r}, code {b!r}"
                                                       for k, (a, b) in bad.items()))


def normalized_prominence(kb):
    """Knowledge-base prominence keyed by normalised name, the form the
    tagger's labels take (round-6 code review, P1).  Names that normalise
    alike within a country get the mean of their prominences."""
    from src.labels import normalize
    out = {}
    for c, d in kb.items():
        acc = {}
        for name, v in d.items():
            acc.setdefault(normalize(name), []).append(float(v))
        out[c] = {k: float(np.mean(v)) for k, v in acc.items() if k}
    return out


def provenance(root, config_path, vocab_path, tag, n_null, frontier_n_null, reader="qwen",
               scorers=SCORERS):
    """What an output was computed from.  Outputs are merged only with outputs
    of identical provenance (round-3 code review, P1).  Round-5 code review,
    P2: the frontier's Monte Carlo setting and the analysis code are part of
    it, so a frontier-only rerun at 500 draws, or after a matcher change,
    cannot be merged into an output computed otherwise."""
    import hashlib

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read())
        return h.hexdigest()[:16]

    store = {}
    for sub in ("manifest", "verify", "tags"):
        pth = (os.path.join(root, sub, f"{tag}.jsonl") if sub == "manifest"
               else reader_file(root, sub, tag, reader))
        store[sub] = sha(pth) if os.path.exists(pth) else None
    for sc in scorers:
        pth = os.path.join(root, "scores", f"{tag}__{sc}.jsonl")
        store[f"scores/{sc}"] = sha(pth) if os.path.exists(pth) else None
    pth = os.path.join(root, "embed", f"{tag}__dinov2.npz")
    store["embed"] = sha(pth) if os.path.exists(pth) else None
    code = hashlib.sha256()
    for f in ["analyze.py"] + sorted(os.path.join("src", x) for x in
                                     os.listdir(os.path.join(HERE, "src"))
                                     if x.endswith(".py")):
        code.update(f.encode() + b"\0" + sha(os.path.join(HERE, f)).encode())
    out = {"config_sha": sha(config_path), "vocab_sha": sha(vocab_path),
            "store_sha": store, "n_null": n_null,
            "frontier_n_null": frontier_n_null,
            "code_sha": code.hexdigest()[:16]}
    if reader not in (None, "qwen"):
        # Only non-default readers carry the key, so provenance of the
        # pre-registered outputs is unchanged and still merges.
        out["reader"] = reader
    return out


def rebuild_claims(res):
    """Claims are derived data: always recomputed from the merged results, and
    removed where their inputs are absent, so a partial rerun cannot leave a
    stale claim beside a fresh part."""
    for key in [k for k in res if k.startswith("claims__")]:
        del res[key]
    reader = (res.get("provenance") or {}).get("reader")
    for s in SCORERS:
        if f"a1__{s}" in res:
            c = claims(res, scorer=s)
            if reader:
                # Experiment audit run 02: amendment 6 makes every reader but
                # Qwen a sensitivity analysis.  Its decision rules are still
                # evaluated, but nothing it produces is a claim.
                c = {n: ({**d, "claim": False,
                          "rule_met_sensitivity_only": bool(d.get("claim"))}
                         if n in ("A1", "A2", "A3") else d) for n, d in c.items()}
                c.update(confirmatory=False, reader=reader)
            res[f"claims__{s}"] = c
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("ARIS_STORE", "/root/autodl-tmp/store"))
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "pilot.yaml"))
    ap.add_argument("--vocab", default=os.path.join(HERE, "data", "cspace_cuisine_vocab.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "real"))
    ap.add_argument("--gens", default="")
    ap.add_argument("--n-null", type=int, default=0, help="override config n_null")
    ap.add_argument("--frontier-n-null", type=int, default=2000)
    ap.add_argument("--parts", default="a1,r1,rep,frontier")
    ap.add_argument("--reader", default="qwen", choices=["qwen", "pixtral", "siglip"],
                    help="qwen is pre-registered; others are amendment 6 sensitivity readers "
                         "(use a separate --out)")
    ap.add_argument("--scorers", default=",".join(SCORERS),
                    help="amendment 7: also laion_aes,hpsv21 (use a separate --out)")
    ap.add_argument("--replace", action="store_true",
                    help="discard an existing output of different provenance")
    a = ap.parse_args(argv)
    if a.reader != "qwen" and os.path.abspath(a.out) == os.path.join(HERE, "results", "real"):
        raise SystemExit("a sensitivity reader writes to its own --out, never over the "
                         "pre-registered results")
    scorers = tuple(a.scorers.split(","))
    if (set(scorers) - set(SCORERS)) and os.path.abspath(a.out) == os.path.join(HERE, "results", "real"):
        raise SystemExit("amendment 7 scorers write to their own --out")
    cfg = load_config(a.config)
    validate_frontier_config(cfg)
    full, _ = load_vocab_file(a.vocab)
    with open(a.vocab) as fh:
        kb = normalized_prominence(json.load(fh)["kb_prominence"])
    n_null = a.n_null or cfg["n_null"]
    parts = set(a.parts.split(","))
    tags = [t for t in (a.gens.split(",") if a.gens else
                        [gen_tag(g) for g in cfg["generators"]])
            if os.path.exists(os.path.join(a.root, "manifest", f"{t}.jsonl"))]
    os.makedirs(a.out, exist_ok=True)
    for tag in tags:
        t0 = time.time()
        cell = Cell(a.root, cfg, tag, full, reader=a.reader, scorers=scorers)
        res = {"generator": tag, "generator_id": cell.gen.generator_id,
               "scorers": sorted(cell.scorers), "n_null": n_null,
               "provenance": provenance(a.root, a.config, a.vocab, tag, n_null,
                                         a.frontier_n_null, a.reader, scorers)}
        for s in cell.scorers:
            if "a1" in parts:
                res[f"a1__{s}"] = a1(cell, s, kb, n_null)
                print(f"[{tag}/{s}] A1 done {time.time()-t0:.0f}s", flush=True)
            if "r1" in parts and cfg.get("known_label"):
                res[f"r1__{s}"] = r1(cell, s, n_null)
                print(f"[{tag}/{s}] R1 done {time.time()-t0:.0f}s", flush=True)
            if "rep" in parts:
                res[f"rep__{s}"] = repetition(cell, s)
        if ("frontier" in parts and cell.emb is not None
                and all(s in cell.scorers for s in SCORERS)):
            for s in SCORERS:
                j = [x for x in SCORERS if x != s][0]
                res[f"a3__{s}"] = frontier(cell, s, j, a.frontier_n_null)
                print(f"[{tag}/{s}] A3 done {time.time()-t0:.0f}s", flush=True)
            for s in [x for x in EXTRA_SCORERS if x in cell.scorers]:
                for j in SCORERS:
                    res[f"a3x__{s}__{j}"] = frontier(cell, s, j, a.frontier_n_null)
                    print(f"[{tag}/{s}->{j}] A3x done {time.time()-t0:.0f}s", flush=True)
        path = os.path.join(a.out, f"{tag}.json")
        prev = {}
        if os.path.exists(path):
            with open(path) as fh:
                prev = json.load(fh)
            if prev.get("provenance") != _jsonable(res["provenance"]):
                if not a.replace:
                    raise SystemExit(
                        f"{path} was computed from a different config, vocabulary, "
                        "store, n_null, frontier n_null or analysis code; rerun "
                        "all parts with --replace")
                prev = {}
        prev.update(_jsonable(res))
        prev = _jsonable(rebuild_claims(prev))
        with open(path, "w") as fh:
            json.dump(prev, fh, indent=1)
        print(f"wrote {path} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
