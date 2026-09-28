"""Fig 2 (appendix C). Diversity change of the aesthetic top-16 vs random-16 of the
same pool: DINOv2 Vendi (reader-free) and dish-label Vendi under three readers,
per generator x scorer cell.  Values: a3__<scorer>.chosen_on_tuning.topk.
evaluation.emb_delta and a1__<scorer>.raw.{mean_delta, marginal.ci}; the
three-reader flag is sensitivity.json diversity_robust."""
import json
import os

import matplotlib.pyplot as plt

from common import FIG, GEN_NAME, GENS, RD_COL, RD_NAME, READERS, RES, SC_NAME, SCORERS, load, pct, pct_ci, style

style()
robust = json.load(open(os.path.join(RES, "sensitivity.json")))["diversity_robust"]
series = [("dinov2", "DINOv2 (reader-free)", "#555555", "D")] + \
         [(rd, RD_NAME[rd], RD_COL[rd], "o") for rd, _ in READERS]
fig, ax = plt.subplots(figsize=(5.5, 2.2))
x, ticks = 0, []
for g in GENS:
    for s in SCORERS:
        for t, (key, lab, col, mk) in enumerate(series):
            if key == "dinov2":
                e = load("real", g)["a3__" + s]["chosen_on_tuning"]["topk"]["evaluation"]["emb_delta"]
                m, ci = e["mean"], e["ci"]
            else:
                rdir = dict(READERS)[key]
                e = load(rdir, g)["a1__" + s]["raw"]
                m, ci = e["mean_delta"], e["marginal"]["ci"]
            xx = x + (t - 1.5) * 0.18
            lo, hi = pct_ci(ci)
            ax.plot([xx, xx], [lo, hi], color=col, lw=1.1)
            ax.plot(xx, pct(m), mk, color=col, ms=3.8, mec="white", mew=0.5,
                    label=lab if x == 0 else None)
        flag = "all 3 readers CI<0" if robust[f"{g}|{s}"] else "not all 3 CI<0"
        ticks.append((x, f"{GEN_NAME[g].replace('.1-s', '')} / {SC_NAME[s]}\n({flag})"))
        x += 1
ax.axhline(0, color="#999999", lw=0.7)
ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks], fontsize=6.5)
ax.set_ylabel("change vs random-16, %")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower right", frameon=False, ncol=4, fontsize=6.5, bbox_to_anchor=(1.0, 0.0))
ax.set_ylim(-22, 4)
fig.savefig(os.path.join(FIG, "fig2_diversity_grid.pdf"))
fig.savefig(os.path.join(FIG, "fig2_diversity_grid.png"), dpi=200)
print("fig2 ok")
