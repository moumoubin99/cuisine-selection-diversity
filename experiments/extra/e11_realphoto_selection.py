"""E11: top-k selection on real, labelled photographs (CPU).

On generated images the dish identity is only known through a reader.  Here it
is known: every photograph carries the dataset dish id.  For each country we
draw pools of photographs, keep the top quarter under a scorer (template-A
text, as in the main study) and compare the kept set with random subsets of
the same size, on

  * dataset labels     (exact-match Vendi of the dish id: the ground truth)
  * reader labels      (qwen / pixtral / siglip; vocabulary label, else the
                        open answer; unanswered photos dropped)
  * embeddings         (DINOv2 / SigLIP Vendi, if embed__<name>.npz exists)

Two questions: does top-k keep fewer distinct dishes on real photos, and do the
reader and embedding contrasts move with the dataset-label contrast?

Per pool: delta = log V(top-k) - mean_r log V(random-k).  Pools are averaged
within country, countries weighted equally.  Intervals: half-sampling of dishes
within country (without replacement; the half-sample variance matches the
full-sample variance through the finite-population correction), centred on the
point estimate: CI = est + quantiles of (half - mean(half)).
"""
from __future__ import annotations

import argparse
import json
import zlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import realimg  # noqa: E402
from src.labels import VocabMatcher  # noqa: E402
from src.store import load_vocab_file, read_jsonl  # noqa: E402

READERS = ("qwen", "pixtral", "siglip")
EMBEDDERS = ("dinov2", "siglip")


def _log_vendi_labels(codes):
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())  # log of exp(H); /k cancels in the contrast


def _log_vendi_emb(X):
    lam = np.linalg.eigvalsh(X @ X.T / len(X))
    lam = lam[lam > 1e-12]
    return float(-(lam * np.log(lam)).sum())


class Data:
    def __init__(self, root, vocab):
        rows = read_jsonl(os.path.join(root, "manifest.jsonl"))
        self.rows = rows
        self.ids = [r["image_id"] for r in rows]
        self.idx = {i: n for n, i in enumerate(self.ids)}
        self.country = np.array([r["country"] for r in rows])
        self.dish = np.array([r["dish_id"] for r in rows])
        self.countries = sorted(set(self.country))
        # label codes per measure; -1 = missing
        self.labels = {"dataset": self._codes([r["dish_id"] for r in rows])}
        matcher = VocabMatcher(load_vocab_file(vocab)[0])
        a = argparse.Namespace(root=root)
        self.unresolved = {}
        for rd in READERS:
            if not os.path.exists(os.path.join(root, f"tags__{rd}.jsonl")):
                continue
            lab = realimg.reader_labels(a, rd, rows, matcher)
            vals = []
            for i in self.ids:
                x = lab.get(i)
                v = None if x is None else (x["label"] or x["raw"])
                vals.append(v)
            self.labels[rd] = self._codes(vals)
            self.unresolved[rd] = float(np.mean([v is None for v in vals]))
        self.emb = {}
        for name in EMBEDDERS:
            p = os.path.join(root, f"embed__{name}.npz")
            if os.path.exists(p):
                z = np.load(p)
                E = np.zeros((len(rows), z["E"].shape[1]), dtype=np.float64)
                for i, e in zip(z["ids"], z["E"]):
                    E[self.idx[str(i)]] = e
                self.emb[name] = E / np.linalg.norm(E, axis=1, keepdims=True)
        self.scores = {}
        for s in ("pickscore", "imagereward", "laion_aes", "hpsv21"):
            p = os.path.join(root, f"scores__{s}.jsonl")
            if os.path.exists(p):
                v = np.full(len(rows), np.nan)
                for r in read_jsonl(p):
                    v[self.idx[r["image_id"]]] = r["score"]
                self.scores[s] = v

    @staticmethod
    def _codes(vals):
        keys = {}
        out = np.full(len(vals), -1)
        for n, v in enumerate(vals):
            if v is not None:
                out[n] = keys.setdefault(str(v), len(keys))
        return out

    def measures(self):
        return list(self.labels) + [f"emb_{e}" for e in self.emb]


def _set_value(D, measure, members):
    if measure.startswith("emb_"):
        return _log_vendi_emb(D.emb[measure[4:]][members])
    c = D.labels[measure][members]
    c = c[c >= 0]
    return _log_vendi_labels(c) if len(c) >= 2 else np.nan


def country_deltas(D, scorer, photos, rng, n_pools, n_rand, pool_max, frac):
    """photos: array of row indices of one (possibly resampled) country.
    Returns {measure: mean over pools of log V(top) - mean log V(random)}."""
    P = min(pool_max, len(photos))
    k = max(4, int(round(P * frac)))
    sc = D.scores[scorer]
    ms = D.measures()
    acc = {m: [] for m in ms}
    for _ in range(n_pools):
        pool = photos[rng.permutation(len(photos))[:P]]
        # tie-break the (duplicated under bootstrap) photos at random
        order = np.lexsort((rng.random(P), -sc[pool]))
        top = pool[order[:k]]
        rand = [pool[rng.permutation(P)[:k]] for _ in range(n_rand)]
        for m in ms:
            t = _set_value(D, m, top)
            r = np.nanmean([_set_value(D, m, x) for x in rand])
            acc[m].append(t - r)
    return {m: float(np.nanmean(v)) for m, v in acc.items()}


def estimate(D, scorer, groups, rng, a):
    """groups: {country: [(dish rows array), ...]} -> per-country and pooled deltas."""
    per = {}
    for c, dishes in groups.items():
        photos = np.concatenate(dishes)
        per[c] = country_deltas(D, scorer, photos, rng, a.n_pools, a.n_rand, a.pool_max, a.frac)
    ms = D.measures()
    pooled = {m: float(np.nanmean([per[c][m] for c in per])) for m in ms}
    return per, pooled


def _groups(D):
    g = {}
    for c in D.countries:
        sel = np.where(D.country == c)[0]
        by = {}
        for i in sel:
            by.setdefault(D.dish[i], []).append(i)
        g[c] = [np.array(v) for _, v in sorted(by.items())]
    return g


def _boot_chunk(args):
    root, vocab, scorer, seed, n, a = args
    D = Data(root, vocab)
    rng = np.random.default_rng(seed)
    base = _groups(D)
    out = []
    for _ in range(n):
        # half-sample of dishes, without replacement: a with-replacement
        # bootstrap duplicates photos, and duplicates share a score, so top-k
        # keeps both copies and the kept set looks spuriously less diverse.
        g = {c: [ds[i] for i in rng.permutation(len(ds))[:(len(ds) + 1) // 2]]
             for c, ds in base.items()}
        per, pooled = estimate(D, scorer, g, rng, a)
        out.append({"pooled": pooled, "per": per})
    return out


def _ci(est, half):
    d = half - half.mean()
    return [float(est + np.percentile(d, 2.5)), float(est + np.percentile(d, 97.5))]


def eta2_null(D, scorers, n_perm=1000, seed=20260925):
    """Between-dish share of within-country score variance, with a null from
    permuting dish labels within country."""
    rng = np.random.default_rng(seed)
    sel = {c: np.where(D.country == c)[0] for c in D.countries}

    def share(v, dish):
        num = den = 0.0
        for c, ix in sel.items():
            x, d = v[ix], dish[ix]
            mu = x.mean()
            den += ((x - mu) ** 2).sum()
            for u in np.unique(d):
                xd = x[d == u]
                num += len(xd) * (xd.mean() - mu) ** 2
        return num / den

    out = {}
    for s in scorers:
        v = D.scores[s]
        obs = share(v, D.dish)
        null = []
        for _ in range(n_perm):
            dish = D.dish.copy()
            for ix in sel.values():
                dish[ix] = dish[ix][rng.permutation(len(ix))]
            null.append(share(v, dish))
        null = np.array(null)
        out[s] = {"eta2": float(obs), "null_mean": float(null.mean()),
                  "null_p95": float(np.percentile(null, 95)),
                  "p": float((1 + (null >= obs).sum()) / (1 + n_perm)), "n_perm": n_perm}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(EXP, "results", "store_mirror_a7", "realimg"))
    ap.add_argument("--vocab", default=os.path.join(EXP, "data", "cspace_cuisine_vocab.json"))
    ap.add_argument("--out", default=os.path.join(EXP, "results", "extra", "e11"))
    ap.add_argument("--scorers", default="pickscore,imagereward,laion_aes,hpsv21")
    ap.add_argument("--pool-max", type=int, default=64)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--n-pools", type=int, default=40)
    ap.add_argument("--n-rand", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--null-only", action="store_true",
                    help="only (re)compute the eta^2 permutation null into an existing e11.json")
    a = ap.parse_args(argv)
    D = Data(a.root, a.vocab)
    if a.null_only:
        path = os.path.join(a.out, "e11.json")
        with open(path) as fh:
            res = json.load(fh)
        res["between_dish_score_variance_share_null"] = eta2_null(
            D, [s for s in a.scorers.split(",") if s in D.scores])
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        print(res["between_dish_score_variance_share_null"])
        return 0
    ms = D.measures()
    print("measures:", ms, "unresolved:", D.unresolved, flush=True)
    res = {"design": {k: getattr(a, k) for k in ("pool_max", "frac", "n_pools", "n_rand", "n_boot")},
           "measures": ms, "unresolved_reader_labels": D.unresolved,
           "countries": {c: {"photos": int((D.country == c).sum()),
                             "dishes": int(len(set(D.dish[D.country == c])))} for c in D.countries},
           "scorers": {}}
    # score-dish association: share of score variance between dishes, per country
    scorers = [s for s in a.scorers.split(",") if s in D.scores]
    per_worker = max(1, a.n_boot // max(1, a.workers // len(scorers)))
    jobs = []
    for s in scorers:
        nb, seed = 0, 0
        while nb < a.n_boot:
            n = min(per_worker, a.n_boot - nb)
            jobs.append((s, (a.root, a.vocab, s, 1000 * (zlib.crc32(s.encode()) % 997) + seed, n, a)))
            nb += n
            seed += 1
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [(s, ex.submit(_boot_chunk, j)) for s, j in jobs]
        rng = np.random.default_rng(20260925)
        point = {s: estimate(D, s, _groups(D), rng, a) for s in scorers}
        boots = {s: [] for s in scorers}
        for s, f in futs:
            boots[s].extend(f.result())
    for s in scorers:
        per, pooled = point[s]
        B = boots[s]
        cell = {"pooled": {}, "per_country": per}
        for m in ms:
            v = np.array([b["pooled"][m] for b in B])
            v = v[np.isfinite(v)]
            cell["pooled"][m] = {"log_delta": pooled[m],
                                 "pct": 100 * (np.exp(pooled[m]) - 1),
                                 "ci": _ci(pooled[m], v)}
        # agreement with the dataset-label contrast across bootstrap draws and countries
        agree = {}
        for m in ms:
            if m == "dataset":
                continue
            x = np.array([per[c]["dataset"] for c in per])
            y = np.array([per[c][m] for c in per])
            ok = np.isfinite(x) & np.isfinite(y)
            diff = np.array([b["pooled"][m] - b["pooled"]["dataset"] for b in B])
            diff = diff[np.isfinite(diff)]
            agree[m] = {"country_pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else None,
                        "minus_dataset": pooled[m] - pooled["dataset"],
                        "minus_dataset_ci": _ci(pooled[m] - pooled["dataset"], diff)}
        cell["agreement_with_dataset"] = agree
        res["scorers"][s] = cell
        print(s, {m: (round(cell["pooled"][m]["pct"], 1),
                      [round(100 * (np.exp(x) - 1), 1) for x in cell["pooled"][m]["ci"]]) for m in ms},
              flush=True)
    # between-dish share of within-country score variance (does the scorer rank dishes?)
    eta = {}
    for s in scorers:
        v = D.scores[s]
        num = den = 0.0
        for c in D.countries:
            sel = np.where(D.country == c)[0]
            x = v[sel]
            mu = x.mean()
            den += ((x - mu) ** 2).sum()
            for d in set(D.dish[sel]):
                xd = v[sel[D.dish[sel] == d]]
                num += len(xd) * (xd.mean() - mu) ** 2
        eta[s] = num / den
    res["between_dish_score_variance_share"] = eta
    res["between_dish_score_variance_share_null"] = eta2_null(D, scorers)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "e11.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("eta2:", eta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
