"""Fig 5. Best-of-N selection intensity (E14, post hoc, descriptive).  x: number of
candidates N a kept set of k=16 is chosen from (N=16 is random selection, N=64 the
paper's top-16).  y: DINOv2 Vendi of the kept set vs random-16 of the same pool, %.
Bands: 95% fixed-country t intervals.  Values from results/extra/e14/e14.json."""
import json
import os

import matplotlib.pyplot as plt

from common import FIG, RES, pct, style

style()
with open(os.path.join(RES, "extra", "e14", "e14.json")) as fh:
    d = json.load(fh)
NS = d["NS"]
GENS = (("sdxl", "SDXL"), ("flux-schnell", "FLUX.1-s"), ("pixart-sigma", "PixArt-Σ"),
        ("sd35m", "SD3.5-M"))
SC = (("pickscore", "PickScore", "#0072B2"), ("imagereward", "ImageReward", "#E69F00"),
      ("laion_aes", "LAION-Aes", "#009E73"), ("hpsv21", "HPSv2.1", "#CC79A7"))
fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.75), sharey=True)
for ax, (g, gname) in zip(axes, GENS):
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    for s, sname, col in SC:
        r = d["results"][g][s]["by_N"]
        y = [pct(r[str(n)]["emb_delta"]["mean"]) for n in NS]
        lo = [pct(r[str(n)]["emb_delta"]["ci"][0]) for n in NS]
        hi = [pct(r[str(n)]["emb_delta"]["ci"][1]) for n in NS]
        ax.fill_between(NS, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(NS, y, "-o", color=col, lw=1.0, ms=2.2, label=sname)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64])
    ax.set_xticklabels(["16", "32", "64"])
    ax.set_title(gname, fontsize=7.5)
    ax.set_xlabel("candidates N")
axes[0].set_ylabel("DINOv2 diversity\nvs random-16, %")
axes[0].legend(frameon=False, loc="lower left", fontsize=6, handlelength=1.2)
fig.savefig(os.path.join(FIG, "fig5_best_of_n.pdf"))
print("wrote fig5_best_of_n.pdf")
