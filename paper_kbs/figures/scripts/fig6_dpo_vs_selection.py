"""Fig 6. Generation-time preference optimization vs output selection (E15, post hoc).
x: DINOv2 Vendi of a 16-image set relative to a random 16 of the stock-SDXL pool, %.
y: gain of the set's mean score over the stock-SDXL pool mean, in native units, under
two judges that neither the selectors nor Diffusion-DPO's training data use
(LAION-Aes, HPSv2.1).  Same prompts, seeds and pools; bars are 95% fixed-country t
intervals.  Values from results/extra/e15/qwen/e15.json (label-free columns do not
depend on the reader)."""
import json
import os
import sys

import matplotlib.pyplot as plt

from common import RES, pct, style

FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "extra", "e15", "qwen", "e15.json")
style()
d = json.load(open(src))
E, Q = d["emb"], d["score_native"]
COL = {"pickscore": "#0072B2", "imagereward": "#E69F00"}
NAME = {"pickscore": "PickScore", "imagereward": "ImageReward"}


def pt(key, j):
    if key is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    e, q = E[key], Q[j][key]
    return (pct(e["mean"]), pct(e["ci"][0]), pct(e["ci"][1]),
            q["mean"], q["ci"][0], q["ci"][1])


def draw(ax, a, b, col, ls="-"):
    ax.annotate("", xy=(b[0], b[3]), xytext=(a[0], a[3]),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, ls=ls,
                                shrinkA=2, shrinkB=2, mutation_scale=7))


def dot(ax, p, col, mk):
    ax.errorbar(p[0], p[3], xerr=[[p[0] - p[1]], [p[2] - p[0]]],
                yerr=[[p[3] - p[4]], [p[5] - p[3]]], fmt=mk, color=col, ms=3.2,
                elinewidth=0.5, capsize=0)


fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
for ax, (j, jname) in zip(axes, (("hpsv21", "HPSv2.1"), ("laion_aes", "LAION-Aes"))):
    o = pt(None, j)
    g = pt("generation_dpo_minus_sdxl", j)
    draw(ax, o, g, "#333333")
    dot(ax, o, "#333333", "o")
    dot(ax, g, "#333333", "s")
    for s in ("pickscore", "imagereward"):
        a = pt(f"selection_sdxl:{s}", j)
        b = pt(f"both_dpo_top_minus_sdxl_rand:{s}", j)
        draw(ax, o, a, COL[s])
        draw(ax, g, b, COL[s], ls="--")
        dot(ax, a, COL[s], "o")
        dot(ax, b, COL[s], "s")
    ax.axvline(0, color="#bbbbbb", lw=0.6)
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax.set_xlabel("DINOv2 diversity change (%)")
    ax.set_ylabel(f"{jname} gain\nover SDXL pool mean")
h = [plt.Line2D([], [], color="#333333", marker="s", ls="-", ms=3, lw=0.9),
     plt.Line2D([], [], color=COL["pickscore"], marker="o", ls="-", ms=3, lw=0.9),
     plt.Line2D([], [], color=COL["imagereward"], marker="o", ls="-", ms=3, lw=0.9),
     plt.Line2D([], [], color="#777777", marker="s", ls="--", ms=3, lw=0.9)]
fig.legend(h, ["Diffusion-DPO, random-16", "SDXL, PickScore top-16",
               "SDXL, ImageReward top-16", "Diffusion-DPO, top-16 (colour = scorer)"],
           frameon=False, fontsize=6, loc="lower center", ncol=2, handlelength=1.8,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(w_pad=3.0, rect=(0, 0.14, 1, 1))
fig.savefig(os.path.join(FIG, "fig6_dpo_vs_selection.pdf"))
fig.savefig(os.path.join(FIG, "fig6_dpo_vs_selection.png"), dpi=200)
print("wrote fig6_dpo_vs_selection.pdf")
