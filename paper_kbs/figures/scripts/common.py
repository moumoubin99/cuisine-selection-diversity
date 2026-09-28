"""Shared loaders and style for the paper figures and tables.

Every empirical value is read from the results files below; nothing is typed in.
"""
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RES = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "paper", "figures")
GENS = ("sdxl", "flux-schnell")
SCORERS = ("pickscore", "imagereward")
READERS = (("qwen", "real"), ("pixtral", "reader_pixtral"), ("siglip", "reader_siglip"))
GEN_NAME = {"sdxl": "SDXL", "flux-schnell": "FLUX.1-s"}
SC_NAME = {"pickscore": "PickScore", "imagereward": "ImageReward"}
RD_NAME = {"qwen": "Qwen2.5-VL", "pixtral": "Pixtral", "siglip": "SigLIP"}
# Okabe-Ito, fixed order: identity follows the entity, never its rank.
RD_COL = {"qwen": "#0072B2", "pixtral": "#E69F00", "siglip": "#009E73"}
SEL_COL = {"topk": "#555555", "stk": "#0072B2", "mmr": "#E69F00", "dpp": "#009E73"}
SEL_NAME = {"topk": "top-k", "stk": "STK", "mmr": "MMR", "dpp": "DPP"}
# Registered A3 rule (refine-logs/FINAL_PROPOSAL.md, PREREG_AMENDMENT.md): the
# re-selector must keep >= 90% of top-k's held-out judge gain.  Not empirical.
A3_RETENTION_THRESHOLD = 0.90

_cache = {}


def load(reader_dir, gen):
    key = (reader_dir, gen)
    if key not in _cache:
        with open(os.path.join(RES, reader_dir, gen + ".json")) as f:
            _cache[key] = json.load(f)
    return _cache[key]


def pct(logv):
    return 100.0 * (math.exp(logv) - 1.0)


def pct_ci(ci):
    return [pct(ci[0]), pct(ci[1])]


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                                               "DejaVu Serif"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
        "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
