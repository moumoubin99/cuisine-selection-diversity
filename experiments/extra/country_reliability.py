#!/usr/bin/env python3
"""Post-hoc, DESCRIPTIVE (Codex review item 3): per-country reader reliability
and the per-country selection effect.

  (a) real photos (E8 set): per country and reader, open-tag dish accuracy and
      country accuracy (the `realimg.e8a` definitions) and forced-choice
      accuracy (`e8b`), each with a Wilson interval over images and a
      percentile bootstrap over dishes (images of one dish are not
      independent).
  (b) generated main pools: per-country top-16 minus random-16 effect on the
      DINOv2 log Vendi (reader-free), and the stored per-country label-Vendi
      means with their simultaneous intervals from the a7 JSONs.
  (c) the ESTABLISHED share (reader's country = prompted country and label in
      that country's CSpace list; `pipeline.classify_tags`) in the top-16
      minus its pool share (the exact random-16 expectation), per pool;
      equal-weight country mean with a country x template stratified pool
      bootstrap, per generator x scorer x reader, plus per country.

    python extra/country_reliability.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from _common import (CONFIG, GENS, M7, OUT, RES, VOCAB, merged_root, pct,
                     strat_boot_mean, wilson)

import realimg
from analyze import Cell
from src.config import load_config
from src.frontier import embedding_log_vendi
from src.labels import VocabMatcher
from src.pipeline import ESTABLISHED, OFF_COUNTRY, PLAUSIBLE, UNRESOLVED, classify_tags
from src.selectors import random_selector, select_topk
from src.store import _main_prompts, load_vocab_file, read_jsonl

READERS = ("qwen", "pixtral", "siglip")
SCORERS = ("pickscore", "imagereward")
N_BOOT = 2000


class _A:
    root = os.path.join(M7, "realimg")


def dish_boot(vals_by_dish, rng, n_boot=N_BOOT):
    ks = sorted(vals_by_dish)
    if len(ks) < 2:
        return [float("nan")] * 2
    out = []
    for _ in range(n_boot):
        pick = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
        out.append(np.mean([x for k in pick for x in vals_by_dish[k]]))
    return pct(out)


def part_a(full):
    rows = realimg._manifest(_A)
    matcher = VocabMatcher(full)
    rng = np.random.default_rng(20260925)
    res = {}
    for reader in READERS:
        lab = realimg.reader_labels(_A, reader, rows, matcher)
        names = {r["image_id"]: {realimg.normalize(r["dish"])} |
                 {realimg.normalize(x) for x in r["aliases"]} for r in rows}

        def dish_ok(r):
            l = lab[r["image_id"]]
            return l["label"] is not None and (
                (r["gt_in_cspace"] and l["label"] == r["gt_label"])
                or l["raw"] in names[r["image_id"]] or l["label"] in names[r["image_id"]])
        ver = {v["image_id"]: v for v in read_jsonl(os.path.join(_A.root, f"verify__{reader}.jsonl"))}
        chance = 0.25 if reader == "siglip" else 0.2
        per = {}
        for c in sorted({r["country"] for r in rows}):
            sel = [r for r in rows if r["country"] == c and r["image_id"] in lab]
            d_ok, c_ok, fc = {}, {}, {}
            for r in sel:
                d_ok.setdefault(r["dish_id"], []).append(float(dish_ok(r)))
                c_ok.setdefault(r["dish_id"], []).append(float(lab[r["image_id"]]["country"] == c))
                if r["image_id"] in ver:
                    fc.setdefault(r["dish_id"], []).append(float(ver[r["image_id"]]["kept"]))
            n = len(sel)
            kd = int(sum(map(sum, d_ok.values())))
            kc = int(sum(map(sum, c_ok.values())))
            nf = sum(map(len, fc.values()))
            kf = int(sum(map(sum, fc.values())))
            per[c] = {"n_images": n, "n_dishes": len(d_ok),
                      "open_dish_acc": kd / n, "open_dish_wilson": wilson(kd, n),
                      "open_dish_dishboot": dish_boot(d_ok, rng),
                      "country_acc": kc / n, "country_wilson": wilson(kc, n),
                      "forced_choice_n": nf,
                      "forced_choice_acc": kf / nf if nf else float("nan"),
                      "forced_choice_wilson": wilson(kf, nf),
                      "forced_choice_dishboot": dish_boot(fc, rng),
                      "forced_choice_chance": chance}
        res[reader] = per
    return res


def pools(cell, cfg):
    for i, p in enumerate(_main_prompts(cfg)):
        for j in range(cfg["n_pools"]):
            yield p, cfg["seed"] + 1000 * i + j


def part_bc(root, cfg, full):
    rng = np.random.default_rng(7)
    out = {}
    for gen in GENS:
        emb_cells = {s: {} for s in SCORERS}      # reader-free
        est = {}
        for reader in READERS:
            cell = Cell(root, cfg, gen, full, reader=reader, scorers=SCORERS)
            est[reader] = {s: {} for s in SCORERS}
            base = {}
            for p, seed in pools(cell, cfg):
                ims, _ = cell.pool(p, seed)
                tags = cell.tagger.tag(ims)
                st, _ = classify_tags(tags, p.country, full[p.country])
                st = np.asarray(st, dtype=object)
                base.setdefault(p.country, {}).setdefault(p.template, []).append(
                    np.mean(st == ESTABLISHED))
                for s in SCORERS:
                    sc = cell.scorers[s].score(ims, p.text)
                    top = np.asarray(select_topk(sc, cfg["k"]))
                    d = {x: float(np.mean(st[top] == x) - np.mean(st == x))
                         for x in (ESTABLISHED, PLAUSIBLE, OFF_COUNTRY, UNRESOLVED)}
                    est[reader][s].setdefault(p.country, {}).setdefault(p.template, []).append(d)
                    if reader == READERS[0]:
                        E = np.stack([cell.emb[im.image_id] for im in ims])
                        mr = np.mean([embedding_log_vendi(E[random_selector(len(ims), cfg["k"], rng)])
                                      for _ in range(200)])
                        emb_cells[s].setdefault(p.country, {}).setdefault(p.template, []).append(
                            embedding_log_vendi(E[top]) - mr)
            est[reader]["_baseline_established"] = {
                c: float(np.mean([np.mean(v) for v in ts.values()])) for c, ts in base.items()}
        g = out[gen] = {"dinov2_log_vendi": {}, "established_shift": {}, "label_vendi_a1": {}}
        for s in SCORERS:
            cells = {c: {t: np.asarray(v) for t, v in ts.items()} for c, ts in emb_cells[s].items()}
            m, ci = strat_boot_mean(cells, N_BOOT, 1)
            g["dinov2_log_vendi"][s] = {
                "mean": m, "ci": ci,
                "by_country": {c: {"mean": float(np.mean([v.mean() for v in ts.values()])),
                                   "ci": strat_boot_mean({c: ts}, N_BOOT, 2)[1],
                                   "n_pools": int(sum(v.size for v in ts.values()))}
                               for c, ts in cells.items()}}
        for reader in READERS:
            g["established_shift"][reader] = {"baseline_by_country": est[reader]["_baseline_established"]}
            bl = est[reader]["_baseline_established"]
            g["established_shift"][reader]["baseline_mean"] = float(np.mean(list(bl.values())))
            for s in SCORERS:
                r = {}
                for x in (ESTABLISHED, PLAUSIBLE, OFF_COUNTRY, UNRESOLVED):
                    cells = {c: {t: np.asarray([d[x] for d in v]) for t, v in ts.items()}
                             for c, ts in est[reader][s].items()}
                    m, ci = strat_boot_mean(cells, N_BOOT, 3)
                    r[x] = {"mean_pp": 100 * m, "ci_pp": [100 * v for v in ci]}
                    if x == ESTABLISHED:
                        r[x]["by_country_pp"] = {
                            c: {"mean": 100 * float(np.mean([v.mean() for v in ts.values()])),
                                "ci": [100 * v for v in strat_boot_mean({c: ts}, N_BOOT, 4)[1]]}
                            for c, ts in cells.items()}
                g["established_shift"][reader][s] = r
            p = os.path.join(RES, "a7", reader, f"{gen}.json")
            if os.path.exists(p):
                with open(p) as fh:
                    a7 = json.load(fh)
                g["label_vendi_a1"][reader] = {
                    s: {"country_means": a7[f"a1__{s}"]["raw"]["marginal"]["country_means"],
                        "simultaneous_intervals":
                            a7[f"a1__{s}"]["raw"]["country_intervals"].get("intervals")}
                    for s in SCORERS if f"a1__{s}" in a7}
        print(f"[{gen}] done", flush=True)
    return out


def pc(x):
    return f"{100 * (np.exp(x) - 1):+.1f}%"


def render(res):
    L = ["# Per-country reliability and selection effects (post hoc, descriptive)", "",
         "## (a) Real photos: per-country reader accuracy", "",
         "Open dish acc [dish-bootstrap 95% CI]; forced-choice acc [dish-bootstrap CI] "
         "(chance 20%; SigLIP 25% since it never picks 'none').", ""]
    for reader, per in res["a"].items():
        L += [f"### {reader}", "", "| country | imgs | dishes | open dish acc | country acc | forced choice |",
              "|---|---|---|---|---|---|"]
        for c, v in per.items():
            L.append(f"| {c} | {v['n_images']} | {v['n_dishes']} | "
                     f"{100*v['open_dish_acc']:.1f} [{100*v['open_dish_dishboot'][0]:.1f}, {100*v['open_dish_dishboot'][1]:.1f}] | "
                     f"{100*v['country_acc']:.1f} | "
                     f"{100*v['forced_choice_acc']:.1f} [{100*v['forced_choice_dishboot'][0]:.1f}, {100*v['forced_choice_dishboot'][1]:.1f}] |")
        L.append("")
    L += ["## (b) DINOv2 log-Vendi, top-16 vs random-16, per country (as 100(exp-1)%)", ""]
    for gen, g in res["bc"].items():
        for s, v in g["dinov2_log_vendi"].items():
            L.append(f"- **{gen}/{s}**: all {pc(v['mean'])} [{pc(v['ci'][0])}, {pc(v['ci'][1])}]; " +
                     ", ".join(f"{c} {pc(x['mean'])}" for c, x in v["by_country"].items()))
    L += ["", "## (c) Established-share shift, top-16 minus pool (pp, stratified pool bootstrap)", "",
          "| gen | reader | baseline % | pickscore | imagereward |", "|---|---|---|---|---|"]
    for gen, g in res["bc"].items():
        for reader, r in g["established_shift"].items():
            cells = [f"{r[s]['established']['mean_pp']:+.1f} [{r[s]['established']['ci_pp'][0]:+.1f}, "
                     f"{r[s]['established']['ci_pp'][1]:+.1f}]" for s in SCORERS]
            L.append(f"| {gen} | {reader} | {100*r['baseline_mean']:.1f} | " + " | ".join(cells) + " |")
    L += ["", "Per-country values, the plausible/off-country/unresolved shifts and the stored "
          "per-country label-Vendi means are in `country_reliability.json`.", ""]
    return "\n".join(L)


def main():
    cfg = load_config(CONFIG)
    full, _ = load_vocab_file(VOCAB)
    root = merged_root()
    res = {"status": "post hoc, descriptive", "config": os.path.relpath(CONFIG, RES),
           "a": part_a(full), "bc": part_bc(root, cfg, full)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "country_reliability.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(OUT, "country_reliability.md"), "w") as fh:
        fh.write(render(res))
    print("wrote", os.path.join(OUT, "country_reliability.{json,md}"))


if __name__ == "__main__":
    main()
