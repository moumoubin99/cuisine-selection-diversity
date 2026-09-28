"""Fig 1. Left: the 64 candidates of one pool fixed by rule, ordered by PickScore,
with subset membership markers (hero_pool.json,
hero_thumbs/).  Right: (a) DINOv2 Vendi, preference-score top-16 vs random-16, per
cell; (b) DINOv2 Vendi, STK-16 vs top-16, PickScore-selected cell.
Values: a3__<scorer>.chosen_on_tuning.{topk.evaluation, stk.vs_topk}.emb_delta
(Qwen results dir; the embedding contrasts are reader-free).  95% t CIs over
countries, as in tables.md."""
import json
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

from common import FIG, GEN_NAME, GENS, SC_NAME, SCORERS, SEL_COL, load, pct, pct_ci, style

style()
SC_ABBR = {"pickscore": "PS", "imagereward": "IR"}
hero = json.load(open(os.path.join(FIG, "data", "hero_pool.json")))
thumbs = os.path.join(FIG, "data", "hero_thumbs")

fig = plt.figure(figsize=(5.5, 2.75))
outer = gridspec.GridSpec(1, 2, width_ratios=[0.74, 1.0], wspace=0.34)
# All 64 candidates, ordered by PickScore (rank 1 top-left).  Markers show which
# subsets keep each image; subsets come from hero_pool.json, never from the plot.
order = sorted(range(len(hero["pickscore"])), key=lambda i: -hero["pickscore"][i])
grid = gridspec.GridSpecFromSubplotSpec(8, 8, subplot_spec=outer[0], wspace=0.05, hspace=0.05)
MARK = [("random", "o", "white", "black"), ("topk", "s", "#D55E00", "white"),
        ("stk", "D", SEL_COL["stk"], "white")]
for pos, idx in enumerate(order):
    ax = fig.add_subplot(grid[pos // 8, pos % 8])
    ax.imshow(Image.open(os.path.join(thumbs, "%04d.jpg" % idx)))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for m, (key, mk, fc, ec) in enumerate(MARK):
        if idx in hero[key]:
            ax.plot(0.16 + 0.34 * m, 0.84, mk, transform=ax.transAxes, ms=3.4, mfc=fc, mec=ec,
                    mew=0.6, clip_on=False)
    if pos == 0:
        ax.set_title("one pool, 64 candidates (illustration)", loc="left", fontsize=7.5)
from matplotlib.lines import Line2D
handles = [Line2D([], [], ls="", marker=mk, mfc=fc, mec=ec if ec != "white" else fc, ms=4.5,
                  label=lab) for (key, mk, fc, ec), lab in
           zip(MARK, ["random-16", "top-16", "STK-16"])]
_lp = outer[0].get_position(fig)
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=((_lp.x0 + _lp.x1) / 2, _lp.y0 - 0.005),
           ncol=3, frameon=False,
           fontsize=6.5, handletextpad=0.2, columnspacing=0.8)

right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], height_ratios=[4, 2.3],
                                         hspace=0.75)
# (a) top-16 vs random-16, four cells
axa = fig.add_subplot(right[0])
labels, y = [], 0
for gen in GENS:
    for sc in SCORERS:
        e = load("real", gen)["a3__" + sc]["chosen_on_tuning"]["topk"]["evaluation"]["emb_delta"]
        m, (lo, hi) = pct(e["mean"]), pct_ci(e["ci"])
        axa.plot([lo, hi], [y, y], color=SEL_COL["topk"], lw=1.2, solid_capstyle="round")
        axa.plot(m, y, "o", color=SEL_COL["topk"], ms=4.5, mec="white", mew=0.6)
        labels.append(f"{GEN_NAME[gen].replace(".1-s", "")} / {SC_ABBR[sc]}")
        y += 1
axa.axvline(0, color="#999999", lw=0.7)
axa.set_yticks(range(len(labels))); axa.set_yticklabels(labels, fontsize=6.5)
axa.set_xlim(-18, 2); axa.set_ylim(len(labels) - 0.5, -0.5)
axa.set_xlabel("DINOv2 Vendi change, %")
axa.set_title("(a) preference-score top-16 vs random-16", loc="left")
# (b) STK vs top-16, PickScore-selected (primary A3) cell
axb = fig.add_subplot(right[1])
labels = []
for y, gen in enumerate(GENS):
    e = load("real", gen)["a3__pickscore"]["chosen_on_tuning"]["stk"]["vs_topk"]["emb_delta"]
    m, (lo, hi) = pct(e["mean"]), pct_ci(e["ci"])
    axb.plot([lo, hi], [y, y], color=SEL_COL["stk"], lw=1.2, solid_capstyle="round")
    axb.plot(m, y, "D", color=SEL_COL["stk"], ms=4, mec="white", mew=0.6)
    labels.append(f"{GEN_NAME[gen].replace(".1-s", "")} / PS")
axb.axvline(0, color="#999999", lw=0.7)
axb.set_yticks(range(len(labels))); axb.set_yticklabels(labels, fontsize=6.5)
axb.set_xlim(-2, 12); axb.set_ylim(len(labels) - 0.5, -0.5)
axb.set_xlabel("DINOv2 Vendi change, %")
axb.set_title("(b) STK-16 vs preference-score top-16", loc="left")
for ax in (axa, axb):
    ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(FIG, "fig1_hero.pdf"))
fig.savefig(os.path.join(FIG, "fig1_hero.png"), dpi=200)
print("fig1 ok")
