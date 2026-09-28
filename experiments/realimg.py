#!/usr/bin/env python3
"""E8 (amendment 7): the readers and scorers on REAL photographs with dataset labels.

WorldCuisines `food-kb` (Winata et al., NAACL 2025) gives Wikimedia photographs of
dishes with the dish name and its countries.  We keep dishes whose ONLY listed
country is one of the eight study countries, so the dish label and the country
label are dataset ground truth, not a model reading.

    python realimg.py prepare  --kb <food-kb snapshot dir>
    python realimg.py tag      --reader qwen|pixtral|siglip
    python realimg.py verify   --reader qwen|pixtral|siglip
    python realimg.py score    --scorers pickscore,imagereward,laion_aes,hpsv21
    python realimg.py analyze  --out results/a7/realimg          # CPU

The reader prompts, the forced-choice construction, the CSpace matcher and the
exact-match rarefied Vendi are the ones used on generated images; only the images
and the label source change.  The tagger prompt still says the image "was produced
by a text-to-image model": the instrument is validated as it is used.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.data import build_underspecified_prompts
from src.labels import VocabMatcher, normalize, normalize_country, parse_tagger_json
from src.store import append_jsonl, load_vocab_file, read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SIDE = 1024


def _log(*x):
    print(time.strftime("[%H:%M:%S]"), *x, flush=True)


def _countries(a):
    return list(load_config(a.config)["countries"])


def _manifest(a):
    return read_jsonl(os.path.join(a.root, "manifest.jsonl"))


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return [v]
    return [str(x) for x in v]


def _aliases(v):
    """food-kb stores aliases as a python-literal list of {name: language}."""
    import ast
    if not v:
        return []
    try:
        d = ast.literal_eval(v) if isinstance(v, str) else v
    except (ValueError, SyntaxError):
        return []
    out = []
    for x in d if isinstance(d, list) else [d]:
        if isinstance(x, dict):
            out += [str(k) for k in x]
        elif isinstance(x, str):
            out.append(x)
    return out


# ------------------------------------------------------------------ prepare

def cmd_prepare(a):
    import pyarrow.parquet as pq
    from PIL import Image, ImageOps
    files = sorted(glob.glob(os.path.join(a.kb, "data", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {a.kb}/data")
    study = set(_countries(a))
    full, _ = load_vocab_file(a.vocab)
    matcher = VocabMatcher(full)
    os.makedirs(os.path.join(a.root, "img"), exist_ok=True)
    rows, n_dish, di = [], 0, 0
    for f in files:
        t = pq.read_table(f)
        cols = t.column_names
        for rec in t.to_pylist():
            di += 1
            cs = _as_list(rec.get("countries"))
            if len(cs) != 1 or cs[0] not in study:
                continue
            name = rec["name"]
            alias = _aliases(rec.get("alias"))
            gt, inv = matcher.match(name)
            if not inv:                       # an alias may be the CSpace name
                for al in alias:
                    g2, i2 = matcher.match(al)
                    if i2:
                        gt, inv = g2, True
                        break
            n_dish += 1
            for j in range(1, 9):
                im = rec.get(f"image{j}") if f"image{j}" in cols else None
                if not im or not im.get("bytes"):
                    continue
                iid = f"wc{di:05d}_{j}"
                path = os.path.join("img", f"{iid}.jpg")
                try:
                    x = ImageOps.exif_transpose(Image.open(io.BytesIO(im["bytes"])))
                    x = x.convert("RGB")
                    x.thumbnail((MAX_SIDE, MAX_SIDE))
                    x.save(os.path.join(a.root, path), quality=92)
                except Exception as e:           # a corrupt file is skipped, and counted
                    _log(f"skip {iid}: {e}")
                    continue
                rows.append({"image_id": iid, "path": path, "dish_id": f"wc{di:05d}",
                             "dish": name, "aliases": alias, "country": cs[0],
                             "gt_label": gt, "gt_in_cspace": bool(inv),
                             "license": rec.get(f"image{j}_license"),
                             "url": rec.get(f"image{j}_url")})
    with open(os.path.join(a.root, "manifest.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    _log(f"prepare: {n_dish} dishes, {len(rows)} images")
    return 0


# ------------------------------------------------------------------ GPU stages

def cmd_tag(a):
    rows = _manifest(a)
    out = os.path.join(a.root, f"tags__{a.reader}.jsonl")
    have = {r["image_id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["image_id"] not in have]
    _log(f"tag[{a.reader}]: {len(todo)} images")
    if not todo:
        return 0
    from src.gpu_backends import load_reader
    full, _ = load_vocab_file(a.vocab)
    vlm = load_reader(a.reader, a.models_dir, vocab_by_country=full)
    for s in range(0, len(todo), 128):
        chunk = todo[s:s + 128]
        reps = vlm.tag([os.path.join(a.root, r["path"]) for r in chunk], batch=a.vlm_batch)
        out_rows = []
        for r, rep in zip(chunk, reps):
            d, c = parse_tagger_json(rep)
            out_rows.append({"image_id": r["image_id"], "reply": rep, "dish": d, "country": c})
        append_jsonl(out, out_rows)
        _log(f"tag[{a.reader}]: {s + len(chunk)}/{len(todo)}")
    return 0


def options_for(row, by_country):
    """Prompted dish, three decoys from the SAME country's food-kb dishes at a
    position fixed by the image ID, then "none of these" -- the construction of
    `run_gpu.verifier_options`, with food-kb names in place of CSpace names."""
    rng = np.random.default_rng(zlib.crc32(row["image_id"].encode()))
    cand = sorted(d for d in by_country[row["country"]] if d != row["dish"])
    decoys = list(rng.choice(cand, 3, replace=False))
    pos = int(rng.integers(4))
    return decoys[:pos] + [row["dish"]] + decoys[pos:] + ["none of these"], pos


def cmd_verify(a):
    rows = _manifest(a)
    out = os.path.join(a.root, f"verify__{a.reader}.jsonl")
    have = {r["image_id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["image_id"] not in have]
    _log(f"verify[{a.reader}]: {len(todo)} images")
    if not todo:
        return 0
    by_country = {}
    for r in rows:
        by_country.setdefault(r["country"], set()).add(r["dish"])
    from src.gpu_backends import load_reader
    vlm = load_reader(a.reader, a.models_dir)
    for s in range(0, len(todo), 128):
        chunk = todo[s:s + 128]
        paths = [os.path.join(a.root, r["path"]) for r in chunk]
        mc = [options_for(r, by_country) for r in chunk]
        probs = vlm.choose(paths, [m[0] for m in mc], batch=a.vlm_batch)
        append_jsonl(out, [{"image_id": r["image_id"], "options": m[0], "true_pos": m[1],
                            "p_choice": [float(v) for v in p],
                            "chosen": int(np.argmax(p)),
                            "kept": bool(int(np.argmax(p)) == m[1])}
                           for r, m, p in zip(chunk, mc, probs)])
        _log(f"verify[{a.reader}]: {s + len(chunk)}/{len(todo)}")
    return 0


def select_text(country):
    """Template A of the main study, the country-level selection text."""
    return next(p.text for p in build_underspecified_prompts([country], ["cuisine"])
                if p.template == "A")


def cmd_score(a):
    rows = _manifest(a)
    from src.gpu_backends import load_real_scorer
    for name in a.scorers.split(","):
        out = os.path.join(a.root, f"scores__{name}.jsonl")
        have = {r["image_id"] for r in read_jsonl(out)}
        todo = [r for r in rows if r["image_id"] not in have]
        _log(f"score[{name}]: {len(todo)} images")
        if not todo:
            continue
        sc = load_real_scorer(name, models_dir=a.models_dir)
        by_c = {}
        for r in todo:
            by_c.setdefault(r["country"], []).append(r)
        for c, rs in by_c.items():
            t = select_text(c)
            for s in range(0, len(rs), 128):
                ch = rs[s:s + 128]
                v = sc.score([os.path.join(a.root, r["path"]) for r in ch], t)
                append_jsonl(out, [{"image_id": r["image_id"], "score": float(x), "text": t}
                                   for r, x in zip(ch, v)])
        del sc
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    return 0


# ------------------------------------------------------------------ analysis (CPU)

def _vendi_norm(labels):
    """exp(H(label shares)) / n: the exact-match Vendi of one set, normalised as
    in `analysis._exact_match_rarefied`."""
    _, c = np.unique(np.asarray([str(x) for x in labels], dtype=object), return_counts=True)
    p = c / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()) / len(labels))


def _spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def _boot(fn, groups, n_boot, rng, strata):
    """Percentile interval of `fn(sample)` over a bootstrap of dishes (`groups`)
    resampled within each country (`strata`: dish -> country)."""
    by_s = {}
    for k in sorted(groups):
        by_s.setdefault(strata[k], []).append(k)
    vals = []
    for _ in range(n_boot):
        pick = [ks[i] for ks in by_s.values() for i in rng.integers(len(ks), size=len(ks))]
        vals.append(fn([x for k in pick for x in groups[k]]))
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _strata(rows):
    return {r["dish_id"]: r["country"] for r in rows}


def reader_labels(a, reader, rows, matcher):
    tags = {r["image_id"]: r for r in read_jsonl(os.path.join(a.root, f"tags__{reader}.jsonl"))}
    out = {}
    for r in rows:
        t = tags.get(r["image_id"])
        if t is None:
            continue
        d, c = parse_tagger_json(t["reply"]) if t.get("reply") else (t["dish"], t["country"])
        lab, inv = matcher.match(d)
        out[r["image_id"]] = {"label": lab, "in_vocab": inv,
                              "country": normalize_country(c) if c else None,
                              "raw": normalize(d) if d else None}
    return out


def e8a(rows, lab):
    """Open-tag accuracy against dataset labels."""
    names = {r["image_id"]: {normalize(r["dish"])} | {normalize(x) for x in r["aliases"]}
             for r in rows}
    res = {}
    for scope, sel in (("all", rows), ("gt_in_cspace", [r for r in rows if r["gt_in_cspace"]])):
        sel = [r for r in sel if r["image_id"] in lab]
        n = len(sel)
        if not n:
            continue
        resolved = [r for r in sel if lab[r["image_id"]]["label"] is not None]
        dish_ok = [r for r in sel if lab[r["image_id"]]["label"] is not None and (
            (r["gt_in_cspace"] and lab[r["image_id"]]["label"] == r["gt_label"])
            or lab[r["image_id"]]["raw"] in names[r["image_id"]]
            or lab[r["image_id"]]["label"] in names[r["image_id"]])]
        ctry_ok = [r for r in sel if lab[r["image_id"]]["country"] == r["country"]]
        res[scope] = {"n": n, "resolved": len(resolved) / n,
                      "dish_acc": len(dish_ok) / n, "dish_acc_wilson": _wilson(len(dish_ok), n),
                      "country_acc": len(ctry_ok) / n,
                      "country_acc_wilson": _wilson(len(ctry_ok), n),
                      "by_country": {c: {
                          "n": sum(r["country"] == c for r in sel),
                          "dish_acc": float(np.mean([r in dish_ok for r in sel if r["country"] == c])),
                          "country_acc": float(np.mean([r in ctry_ok for r in sel if r["country"] == c])),
                          "country_acc_wilson": _wilson(sum(r in ctry_ok for r in sel if r["country"] == c),
                                                        sum(r["country"] == c for r in sel))}
                          for c in sorted({r["country"] for r in sel})}}
    return res


def _wilson(k, n, z=1.96):
    if not n:
        return [float("nan")] * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def e8b(a, reader, rows, n_boot, rng):
    ver = {r["image_id"]: r for r in read_jsonl(os.path.join(a.root, f"verify__{reader}.jsonl"))}
    sel = [r for r in rows if r["image_id"] in ver]
    if not sel:
        return None
    by_dish = {}
    for r in sel:
        by_dish.setdefault(r["dish_id"], []).append(float(ver[r["image_id"]]["kept"]))
    acc = float(np.mean([x for v in by_dish.values() for x in v]))
    none = float(np.mean([ver[r["image_id"]]["chosen"] == 4 for r in sel]))
    return {"n": len(sel), "acc": acc, "ci": _boot(np.mean, by_dish, n_boot, rng, _strata(rows)),
            "none_rate": none, "chance": 0.2,
            "by_country": {c: float(np.mean([ver[r["image_id"]]["kept"] for r in sel
                                             if r["country"] == c]))
                           for c in sorted({r["country"] for r in sel})}}


def _draw_sets(rows, m, n_sets, rng):
    """`n_sets` sets PER COUNTRY of m CSpace-mapped images whose TRUE dish count
    is spread over 1..m: draw d, then d dishes, then m images from their union
    with every dish represented (uniform random sets would almost always hold
    m distinct dishes, leaving no variation in the truth to track)."""
    by_c = {}
    for r in rows:
        if r["gt_in_cspace"]:
            by_c.setdefault(r["country"], {}).setdefault(r["dish_id"], []).append(r)
    sets = []
    for c in sorted(by_c):
        sets += _draw_country(by_c[c], m, n_sets, rng)
    return sets


def _draw_country(by_dish, m, n_sets, rng):
    sets, tries = [], 0
    while len(sets) < n_sets and tries < 200 * n_sets:
        tries += 1
        dishes = sorted(by_dish)
        d = int(rng.integers(1, m + 1))
        if d > len(dishes):
            continue
        pick = [by_dish[dishes[i]] for i in rng.choice(len(dishes), d, replace=False)]
        if sum(len(p) for p in pick) < m:
            continue
        s = [p[int(rng.integers(len(p)))] for p in pick]      # one per dish
        rest = [x for p in pick for x in p if x not in s]
        s += [rest[i] for i in rng.choice(len(rest), m - d, replace=False)]
        sets.append(s)
    return sets


def e8c(sets, lab, m, n_boot, rng):
    """Does reader-measured exact-match Vendi track the dataset's?"""
    t, rd, frac = [], [], []
    for s in sets:
        got = [lab[r["image_id"]]["label"] for r in s if r["image_id"] in lab]
        got = [g for g in got if g is not None]
        frac.append(len(got) / m)
        if len(got) < 2:
            continue
        t.append(_vendi_norm([r["dish_id"] for r in s]))
        rd.append(_vendi_norm(got))
    t, rd = np.asarray(t), np.asarray(rd)
    # Contrast validity: random pairs of sets, the quantity the paper reports.
    i = rng.integers(len(t), size=4000)
    j = rng.integers(len(t), size=4000)
    keep = i != j
    dt = np.log(t[i[keep]]) - np.log(t[j[keep]])
    dr = np.log(rd[i[keep]]) - np.log(rd[j[keep]])
    nz = np.abs(dt) > 1e-9
    big = np.abs(dt) > -np.log(0.9)
    slope = float(np.polyfit(dt, dr, 1)[0])

    def boot(fn):
        v = []
        for _ in range(n_boot):
            b = rng.integers(len(t), size=len(t))
            v.append(fn(b))
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    # Sets share dishes, so this set-level interval is narrower than a dish-level
    # one would be; it is reported as descriptive.

    return {"n_sets": int(len(t)), "m": m, "mean_resolved": float(np.mean(frac)),
            "spearman": _spearman(t, rd),
            "spearman_ci": boot(lambda b: _spearman(t[b], rd[b])),
            "mean_log_ratio": float(np.mean(np.log(rd / t))),
            "contrast_slope": slope,
            "contrast_sign_agree": float(np.mean(np.sign(dt[nz]) == np.sign(dr[nz]))),
            # post hoc: pairs whose TRUE contrast exceeds the paper's 10% margin
            "contrast_sign_agree_gt10": float(np.mean(np.sign(dt[big]) == np.sign(dr[big]))),
            "n_pairs_gt10": int(big.sum()),
            "true_mean": float(t.mean()), "reader_mean": float(rd.mean())}


def e8d(a, rows, lab, reader, scorers, n_boot, rng):
    """Keep rate by within-country score quartile on real photos: the control
    for the recognition association on generated images."""
    ver = {r["image_id"]: r for r in read_jsonl(os.path.join(a.root, f"verify__{reader}.jsonl"))}
    out = {}
    for sname in scorers:
        sc = {r["image_id"]: r["score"]
              for r in read_jsonl(os.path.join(a.root, f"scores__{sname}.jsonl"))}
        sel = [r for r in rows if r["image_id"] in sc and r["image_id"] in ver]
        if not sel:
            continue
        q = {}
        for c in {r["country"] for r in sel}:
            rs = [r for r in sel if r["country"] == c]
            v = np.asarray([sc[r["image_id"]] for r in rs])
            ranks = v.argsort().argsort() / max(len(v) - 1, 1)
            for r, x in zip(rs, ranks):
                q[r["image_id"]] = min(int(x * 4), 3)
        kept = {r["image_id"]: float(ver[r["image_id"]]["kept"]) for r in sel}
        tag_ok = {r["image_id"]: float(r["image_id"] in lab and lab[r["image_id"]]["label"] is not None
                                       and r["gt_in_cspace"] and lab[r["image_id"]]["label"] == r["gt_label"])
                  for r in sel}
        rates = [float(np.mean([kept[r["image_id"]] for r in sel if q[r["image_id"]] == k]))
                 for k in range(4)]
        by_dish = {}
        for r in sel:
            by_dish.setdefault(r["dish_id"], []).append(r["image_id"])

        def top_minus_bottom(ids):
            top = [kept[i] for i in ids if q[i] == 3]
            bot = [kept[i] for i in ids if q[i] == 0]
            return (np.mean(top) - np.mean(bot)) if top and bot else np.nan

        # Within-dish: the score's association with recognition among photos of
        # the SAME dish, so the dish mix of the quartiles cannot produce it.
        def within(ids):
            d = {}
            for i in ids:
                d.setdefault(i.split("_")[0], []).append(i)
            num, den = 0.0, 0
            for g in d.values():
                if len(g) < 2:
                    continue
                v = np.asarray([sc[i] for i in g])
                k = np.asarray([kept[i] for i in g])
                hi, lo = k[v >= np.median(v)], k[v < np.median(v)]
                if len(hi) and len(lo):
                    num += hi.mean() - lo.mean()
                    den += 1
            return num / den if den else np.nan

        out[sname] = {"n": len(sel), "keep_by_quartile": rates,
                      "top_minus_bottom": float(top_minus_bottom(list(kept))),
                      "top_minus_bottom_ci": _boot(top_minus_bottom, by_dish, n_boot, rng, _strata(rows)),
                      "within_dish_hi_minus_lo": float(within(list(kept))),
                      "within_dish_ci": _boot(within, by_dish, n_boot, rng, _strata(rows)),
                      "open_tag_by_quartile": [
                          float(np.mean([tag_ok[r["image_id"]] for r in sel
                                         if q[r["image_id"]] == k and r["gt_in_cspace"]] or [np.nan]))
                          for k in range(4)]}
    return out


def cmd_analyze(a):
    rows = _manifest(a)
    full, _ = load_vocab_file(a.vocab)
    matcher = VocabMatcher(full)
    rng = np.random.default_rng(20260925)
    sets = _draw_sets(rows, a.m, a.n_sets, rng)
    res = {"n_images": len(rows), "n_dishes": len({r["dish_id"] for r in rows}),
           "gt_in_cspace_images": sum(r["gt_in_cspace"] for r in rows),
           "by_country": {c: {"images": sum(r["country"] == c for r in rows),
                              "dishes": len({r["dish_id"] for r in rows if r["country"] == c})}
                          for c in sorted({r["country"] for r in rows})},
           "readers": {}}
    scorers = [s for s in a.scorers.split(",")
               if os.path.exists(os.path.join(a.root, f"scores__{s}.jsonl"))]
    for reader in ("qwen", "pixtral", "siglip"):
        if not os.path.exists(os.path.join(a.root, f"tags__{reader}.jsonl")):
            continue
        lab = reader_labels(a, reader, rows, matcher)
        r = {"e8a": e8a(rows, lab), "e8c": e8c(sets, lab, a.m, a.n_boot, rng)}
        if os.path.exists(os.path.join(a.root, f"verify__{reader}.jsonl")):
            r["e8b"] = e8b(a, reader, rows, a.n_boot, rng)
            r["e8d"] = e8d(a, rows, lab, reader, scorers, a.n_boot, rng)
        res["readers"][reader] = r
        _log(f"{reader}: dish_acc={r['e8a']['all']['dish_acc']:.3f} "
             f"spearman={r['e8c']['spearman']:.3f}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "e8.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    with open(os.path.join(a.out, "e8.md"), "w") as fh:
        fh.write(render(res))
    return 0


def _ci(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


def render(res):
    L = [f"# E8: readers and scorers on WorldCuisines real photographs\n",
         f"{res['n_images']} images of {res['n_dishes']} single-country dishes "
         f"({res['gt_in_cspace_images']} images whose dish maps into CSpace).\n",
         "| reader | resolved | dish acc (all) | dish acc (in CSpace) | country acc | "
         "forced choice acc [95% CI] | none-of-these | Vendi Spearman [CI] | "
         "mean log(reader/true) | contrast slope | sign agree | sign agree, true >10% (post hoc) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["readers"].items():
        a_, c_ = r["e8a"], r["e8c"]
        b_ = r.get("e8b") or {}
        L.append(f"| {n} | {a_['all']['resolved']:.3f} | {a_['all']['dish_acc']:.3f} | "
                 f"{a_.get('gt_in_cspace', {}).get('dish_acc', float('nan')):.3f} | "
                 f"{a_['all']['country_acc']:.3f} | "
                 + (f"{b_['acc']:.3f} {_ci(b_['ci'])} | {b_['none_rate']:.3f} | " if b_ else "– | – | ")
                 + f"{c_['spearman']:.3f} {_ci(c_['spearman_ci'])} | {c_['mean_log_ratio']:+.3f} | "
                 f"{c_['contrast_slope']:.3f} | {c_['contrast_sign_agree']:.3f} | "
                 f"{c_['contrast_sign_agree_gt10']:.3f} (n={c_['n_pairs_gt10']}) |")
    L.append("\n## E8d: forced-choice keep rate by within-country score quartile (real photos)\n")
    L.append("| reader | scorer | Q1 | Q2 | Q3 | Q4 | Q4−Q1 [dish-cluster CI] | within-dish hi−lo [CI] |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n, r in res["readers"].items():
        for s, d in (r.get("e8d") or {}).items():
            q = d["keep_by_quartile"]
            L.append(f"| {n} | {s} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | {q[3]:.3f} | "
                     f"{d['top_minus_bottom']:+.3f} {_ci(d['top_minus_bottom_ci'])} | "
                     f"{d['within_dish_hi_minus_lo']:+.3f} {_ci(d['within_dish_ci'])} |")
    return "\n".join(L) + "\n"


def cmd_embed(a):
    """DINOv2 and SigLIP embeddings of the photographs (for the E8 selection check)."""
    import numpy as np
    from src.gpu_backends import load_embedder
    rows = _manifest(a)
    for name in a.embedders.split(","):
        path = os.path.join(a.root, f"embed__{name}.npz")
        if os.path.exists(path):
            _log(f"embed {name}: exists")
            continue
        E = load_embedder(name, models_dir=a.models_dir).embed(
            [os.path.join(a.root, r["path"]) for r in rows])
        np.savez(path, ids=np.array([r["image_id"] for r in rows]), E=E.astype(np.float32))
        _log(f"embed {name}: {E.shape}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "tag", "verify", "score", "embed", "analyze"])
    ap.add_argument("--kb", default="")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "pilot.yaml"))
    ap.add_argument("--vocab", default=os.path.join(HERE, "data", "cspace_cuisine_vocab.json"))
    ap.add_argument("--root", default=os.path.join(os.environ.get("ARIS_STORE", "/root/autodl-tmp/store"),
                                                   "realimg"))
    ap.add_argument("--models-dir", default=os.environ.get("ARIS_MODELS", "/root/autodl-tmp/models"))
    ap.add_argument("--reader", default="qwen", choices=["qwen", "pixtral", "siglip"])
    ap.add_argument("--vlm-batch", type=int, default=16)
    ap.add_argument("--scorers", default="pickscore,imagereward,laion_aes,hpsv21")
    ap.add_argument("--embedders", default="dinov2,siglip")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "a7", "realimg"))
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n-sets", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    os.makedirs(a.root, exist_ok=True)
    return {"prepare": cmd_prepare, "tag": cmd_tag, "verify": cmd_verify,
            "score": cmd_score, "embed": cmd_embed, "analyze": cmd_analyze}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
