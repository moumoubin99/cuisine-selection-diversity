"""Fig 3. Forced-choice verifier keep rate by aesthetic-score quartile on the
named-dish (known-label) stress-test pools, three readers x four cells, and the
post hoc dish-mix split of Q4 - Q1 into composition and within-dish terms.
Values: r1__<scorer>.verifier_keep_by_score_quartile (results/<reader>/<gen>.json)
and results/posthoc/dish_mix_decomposition.json (pool-cluster bootstrap)."""
import json
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec

from common import (FIG, GEN_NAME, GENS, RD_COL, RD_NAME, READERS, RES, SC_NAME, SCORERS,
                    load, style)

style()
SHORT = {"sdxl": "SDXL", "flux-schnell": "FLUX"}
RD_SHORT = {"qwen": "Qwen", "pixtral": "Pixtral", "siglip": "SigLIP"}
dm = json.load(open(os.path.join(RES, "posthoc", "dish_mix_decomposition.json")))["cells"]
cells = [(g, s) for g in GENS for s in SCORERS]
fig = plt.figure(figsize=(5.5, 3.5))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1.05], hspace=0.62, wspace=0.12)
for j, (gen, sc) in enumerate(cells):
    ax = fig.add_subplot(gs[0, j])
    for rd, rdir in READERS:
        v = load(rdir, gen)["r1__" + sc]["verifier_keep_by_score_quartile"]
        ax.plot(range(1, 5), [100 * a for a in v["accuracy_by_bin"]], "-o", color=RD_COL[rd],
                lw=1.2, ms=3, label=RD_NAME[rd])
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_ylim(20, 90)
    ax.set_title(f"{SHORT[gen]} / {SC_NAME[sc]}", fontsize=7)
    if j == 0:
        ax.set_ylabel("verifier keep rate, %")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("score quartile")
    if j == 0:
        ax.text(-0.32, 1.13, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold")
fig.axes[0].legend(loc="lower left", frameon=False, handlelength=1.2, fontsize=6, borderaxespad=0.1,
                   labelspacing=0.25)

ax = fig.add_subplot(gs[1, :])
terms = [("total", "Q4 − Q1 total", "#555555", "o"),
         ("composition", "composition (dish mix)", "#D55E00", "s"),
         ("within", "within-dish remainder", "#56B4E9", "^")]
x, ticks = 0, []
for gen, sc in cells:
    for rd, _ in READERS:
        c = dm[f"{gen}/{sc}/{rd}"]
        for t, (key, lab, col, mk) in enumerate(terms):
            e = c[key]
            xx = x + (t - 1) * 0.24
            ax.plot([xx, xx], [100 * e["ci"][0], 100 * e["ci"][1]], color=col, lw=1.0)
            ax.plot(xx, 100 * e["est"], mk, color=col, ms=3.6, mec="white", mew=0.4,
                    label=lab if x == 0 else None)
        ticks.append((x, RD_SHORT[rd]))
        x += 1
    x += 0.5
ax.axhline(0, color="#999999", lw=0.7)
ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks], fontsize=6)
ax.set_xlim(-0.6, x - 0.9)
for i, (gen, sc) in enumerate(cells):
    ax.text(i * 3.5 + 1, 1.02, f"{SHORT[gen]} / {SC_NAME[sc]}", ha="center", va="bottom",
            fontsize=6.5, transform=ax.get_xaxis_transform())
ax.set_ylabel("keep-rate change, points")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncol=3, fontsize=6.5,
          markerscale=1.4)
ax.grid(axis="x", visible=False)
ax.text(-0.075, 1.12, "(b) post hoc", transform=ax.transAxes, fontsize=8, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig3_recognition.pdf"))
fig.savefig(os.path.join(FIG, "fig3_recognition.png"), dpi=200)
print("fig3 ok")
