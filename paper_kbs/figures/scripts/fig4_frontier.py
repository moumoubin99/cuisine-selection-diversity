"""Fig 4. A3 diversity-quality frontier in the primary (PickScore -> ImageReward)
cell.  x: held-out judge-gain retention = curves[cfg].gain_judge /
curves['topk|None'].gain_judge (point estimates, SD-standardised gains -- the
registered A3 estimand).  y: diversity change vs aesthetic top-16,
curves_vs_topk[cfg].{delta, emb_delta}.mean.  Configurations chosen on the tuning
pools (chosen_on_tuning.<family>.param) are outlined.  Top-k sits at (100%, 0) by
definition.  The dashed line is the registered A3 retention threshold (common.py)."""
import os

import matplotlib.pyplot as plt

from common import (A3_RETENTION_THRESHOLD, FIG, GEN_NAME, GENS, SEL_COL, SEL_NAME, load, pct,
                    style)

style()
MK = {"stk": "D", "mmr": "o", "dpp": "s"}
fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.6), sharex=True)
for j, gen in enumerate(GENS):
    a = load("real", gen)["a3__pickscore"]
    top = a["curves"]["topk|None"]["gain_judge"]["mean"]
    for i, (key, ylab) in enumerate((("delta", "dish-label diversity\nvs top-16, %"),
                                     ("emb_delta", "DINOv2 diversity\nvs top-16, %"))):
        ax = axes[i, j]
        ax.axvline(100 * A3_RETENTION_THRESHOLD, color="#999999", lw=0.8, ls="--")
        ax.axhline(0, color="#bbbbbb", lw=0.6)
        ax.plot(100, 0, "*", color=SEL_COL["topk"], ms=8, zorder=5,
                label="top-16" if (i, j) == (0, 0) else None)
        for fam in ("stk", "mmr", "dpp"):
            cfgs = [c for c in a["curves"] if c.startswith(fam + "|")]
            xs = [100 * a["curves"][c]["gain_judge"]["mean"] / top for c in cfgs]
            ys = [pct(a["curves_vs_topk"][c][key]["mean"]) for c in cfgs]
            ax.plot(xs, ys, "-" + MK[fam], color=SEL_COL[fam], lw=0.9, ms=3, alpha=0.9,
                    label=SEL_NAME[fam] if (i, j) == (0, 0) else None)
            chosen = f"{fam}|{a['chosen_on_tuning'][fam]['param']}"
            k = cfgs.index(chosen)
            ax.plot(xs[k], ys[k], MK[fam], color=SEL_COL[fam], ms=6.5, mec="black", mew=0.9,
                    zorder=6)
        if i == 0:
            ax.set_title(f"{GEN_NAME[gen]}, PickScore → ImageReward", fontsize=7.5)
        if i == 1:
            ax.set_xlabel("held-out judge-gain retention, % of top-16")
        if j == 0:
            ax.set_ylabel(ylab)
from matplotlib.lines import Line2D
handles, labels = axes[0, 0].get_legend_handles_labels()
handles += [Line2D([], [], ls="", marker="o", mfc="none", mec="black", mew=0.9, ms=6.5),
            Line2D([], [], color="#999999", lw=0.8, ls="--")]
labels += ["chosen on tuning pools", "90% retention (A3 rule)"]
fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=6.5,
           handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
fig.align_ylabels(axes[:, 0])
fig.tight_layout(h_pad=0.6, w_pad=0.8, rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(FIG, "fig4_frontier.pdf"))
fig.savefig(os.path.join(FIG, "fig4_frontier.png"), dpi=200)
print("fig4 ok")
