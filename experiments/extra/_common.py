"""Shared helpers for the post-hoc descriptive analyses in `extra/`.

`merged_root()` builds a read-only view of the two local store mirrors
(SDXL / FLUX from `store_mirror`, PixArt-Sigma and the amendment-7 scores from
`store_mirror_a7`) as symlinks in a temporary directory, so `analyze.Cell`
replays them unchanged.  Nothing here writes into the mirrors.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

RES = os.path.join(EXP, "results")
M1 = os.path.join(RES, "store_mirror")
M7 = os.path.join(RES, "store_mirror_a7")
OUT = os.path.join(RES, "extra")
CONFIG = os.path.join(EXP, "configs", "amend7.yaml")
VOCAB = os.path.join(EXP, "data", "cspace_cuisine_vocab.json")
GENS = ("sdxl", "flux-schnell", "pixart-sigma")


def merged_root():
    root = tempfile.mkdtemp(prefix="aris_extra_store_")
    for sub in ("manifest", "tags", "verify", "embed", "scores"):
        os.makedirs(os.path.join(root, sub))
        # a7 first: it holds all four scorers for every generator.
        for base in (M7, M1):
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                dst = os.path.join(root, sub, f)
                if not os.path.exists(dst):
                    os.symlink(os.path.join(d, f), dst)
    return root


def wilson(k, n, z=1.96):
    if not n:
        return [float("nan")] * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def pct(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def strat_boot_mean(cells, n_boot=2000, seed=0):
    """Equal-weight mean over the top-level keys of `cells` ({country:
    {template: array}}), each country the mean of its template means; the
    bootstrap resamples pools within each (country, template) cell, as
    `analyze.frontier.fixed_design` does."""
    rng = np.random.default_rng(seed)

    def est(pick):
        return float(np.mean([np.mean([pick(v).mean() for v in ts.values()])
                              for ts in cells.values()]))
    point = est(lambda v: v)
    boots = [est(lambda v: v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)]
    return point, pct(boots)
