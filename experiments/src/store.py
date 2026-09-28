"""The on-disk record of a real run, and backends that replay it.

GPU stages (`run_gpu.py`) write images and per-image model outputs here; the
analysis (`analyze.py`) reads them back through `CachedGenerator`,
`CachedScorer` and `CachedTagger`, which implement the same three interfaces as
the mocks.  So the real analysis is `pipeline.run_audit` -- the code the smoke
controls validate -- and not a second implementation of it.

Layout under `root`:

    images/<gen>/<pool_seed>/<index>.jpg
    manifest/<gen>.jsonl            one row per image (see `plan_pools`)
    scores/<gen>__<scorer>.jsonl    {"image_id", "score", "text"}
    tags/<gen>[__<reader>].jsonl    {"image_id", "reply", "dish", "country"}
    verify/<gen>[__<reader>].jsonl  {"image_id", "dish", "kept", "p_choice", "p_yes",
                                     "p_yes_decoy", ...}   known-label only
    embed/<gen>__dinov2.npz         ids, E

Every row is keyed by the image ID of `src/annotations.py`, which names the
generator's weights AND configuration, so outputs of two generators can never
be joined to each other's images.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .data import Prompt, build_underspecified_prompts
from .known_label import known_label_dishes, known_label_prompt
from .labels import VocabMatcher, normalize_country, parse_tagger_json
from .pipeline import TAGGER_ERROR

__all__ = ["generator_config", "load_vocab_file", "PoolSpec", "plan_pools", "read_jsonl", "append_jsonl",
           "CachedImage", "CachedGenerator", "CachedScorer", "CachedTagger",
           "load_embeddings", "gen_tag", "reader_file"]


def gen_tag(model_id):
    """Short directory name for a generator (a generator entry may override it
    with an explicit `tag`, e.g. SDXL with a swapped UNet)."""
    if isinstance(model_id, dict):
        if model_id.get("tag"):
            return model_id["tag"]
        model_id = model_id["id"]
    return {"stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
            "black-forest-labs/FLUX.1-schnell": "flux-schnell",
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "pixart-sigma",
            "stabilityai/stable-diffusion-3.5-medium": "sd35m"}.get(
        model_id, model_id.split("/")[-1].lower())


def generator_config(g, cfg):
    """The settings that determine a generator's images, as fingerprinted.

    One definition for preflight, generation and replay, so the id stamped on an
    image and the id the analysis expects cannot be computed two ways.
    """
    c = {k: v for k, v in g.items() if k not in ("id", "batch", "tag")}
    c.setdefault("dtype", cfg.get("dtype", "float16"))
    return c


def reader_file(root, sub, tag, reader="qwen"):
    """tags/ and verify/ files: the pre-registered Qwen reader keeps the plain
    name; a sensitivity reader (amendment 6) gets `<gen>__<reader>.jsonl`."""
    name = tag if reader in (None, "qwen") else f"{tag}__{reader}"
    return os.path.join(root, sub, f"{name}.jsonl")


def load_vocab_file(path, min_kb_prominence=None):
    """(full vocab, known-label vocab) per country, from `data/cspace_cuisine_vocab.json`."""
    with open(path) as fh:
        d = json.load(fh)
    full = d["vocab"]
    if min_kb_prominence is None:
        return full, full
    known = {c: sorted(a for a, v in d["kb_prominence"][c].items()
                       if v >= min_kb_prominence) for c in full}
    return full, known


@dataclass(frozen=True)
class PoolSpec:
    kind: str                   # "main" | "known" | "tune"
    prompt: Prompt              # prompt.text is the SELECTION text
    pool_seed: int
    gen_prompts: tuple          # per-image generation prompt
    dishes: tuple = ()          # known-label pools only


def _main_prompts(cfg):
    return [p for p in build_underspecified_prompts(
        countries=cfg["countries"], domains=[cfg["domain"]])
        if p.template in cfg["templates"] and p.country in cfg["countries"]]


def plan_pools(cfg, vocab_by_country):
    """Every pool the config asks for, with the seeds `run_audit` will use.

    Main pools use `seed + 1000*i + j` over the prompt list, exactly as
    `pipeline.run_audit` numbers them; the known-label and tuning pools use the
    same rule from their own seed bases, so a replay through `run_audit` with
    `seed=<base>` finds them.
    """
    n = cfg["n"]
    out = []
    prompts = _main_prompts(cfg)
    for i, p in enumerate(prompts):
        for j in range(cfg["n_pools"]):
            out.append(PoolSpec("main", p, cfg["seed"] + 1000 * i + j, (p.text,) * n))
    by_a = {p.country: p for p in prompts if p.template == "A"}
    kl = cfg.get("known_label") or {}
    for i, p in enumerate(known_label_prompt_list(cfg)):
        for j in range(kl.get("pools_per_country", 0)):
            seed = kl["seed_base"] + 1000 * i + j
            dishes = known_label_dishes(vocab_by_country[p.country], n, seed,
                                        n_dishes=kl.get("n_dishes", 32),
                                        zipf_s=kl.get("zipf_s", 1.1))
            out.append(PoolSpec("known", p, seed,
                                tuple(known_label_prompt(d, p.country) for d in dishes),
                                tuple(dishes)))
    tu = cfg.get("tune") or {}
    for i, c in enumerate(cfg["countries"]):
        p = by_a[c]
        for j in range(tu.get("pools_per_country", 0)):
            out.append(PoolSpec("tune", p, tu["seed_base"] + 1000 * i + j,
                                (p.text,) * n))
    seeds = [s.pool_seed for s in out]
    if len(set(seeds)) != len(seeds):
        raise ValueError("pool seeds collide across pool kinds; widen the seed bases")
    return out


def known_label_prompt_list(cfg):
    """One `K` prompt per country.  Its text is the country's template-A prompt:
    the text the SELECTOR scores against, not the one the image was drawn from."""
    a = {p.country: p for p in _main_prompts(cfg) if p.template == "A"}
    return [Prompt(f"{cfg['domain']}|{c}|K", a[c].text, c, cfg["domain"], "K")
            for c in cfg["countries"]]


def tune_prompt_list(cfg):
    return [p for p in _main_prompts(cfg) if p.template == "A"]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        lines = [line for line in fh if line.strip()]
    out = [json.loads(line) for line in lines[:-1]]
    if lines:
        try:  # a concurrent writer may leave the last line incomplete
            out.append(json.loads(lines[-1]))
        except json.JSONDecodeError:
            if lines[-1].endswith("\n"):
                raise
    return out


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class CachedImage:
    __slots__ = ("image_id", "country", "artifact", "label_source", "path")

    def __init__(self, image_id, country, path, artifact=None, label_source=None):
        self.image_id = image_id
        self.country = country
        self.path = path
        self.label_source = label_source
        # Left UNSET on ordinary pools, so `hasattr(im, "artifact")` is False
        # and `run_pool(audit=True)` refuses them instead of auditing against
        # a column of Nones.
        if label_source is not None:
            self.artifact = artifact


class CachedGenerator:
    """Replays generated pools.  `generate` looks a pool up by its seed.

    For known-label pools each image carries `artifact` = its prompted dish, or
    None when `known_keep="verified"` and the closed-question verifier said the
    image does not show it.  `run_pool(..., audit=True)` reads that attribute
    and reports it as `known_by_construction`.
    """

    def __init__(self, root, tag, known_keep="verified", p_yes_threshold=0.5, reader="qwen"):
        rows = read_jsonl(os.path.join(root, "manifest", f"{tag}.jsonl"))
        if not rows:
            raise FileNotFoundError(f"no manifest for {tag} under {root}")
        self.root = root
        self.tag = tag
        ids = {r["generator_id"] for r in rows}
        if len(ids) != 1:
            raise ValueError(f"{tag}: manifest mixes generator ids {sorted(ids)}")
        self.generator_id = ids.pop()
        self._pools = {}
        for r in rows:
            self._pools.setdefault(int(r["pool_seed"]), []).append(r)
        for v in self._pools.values():
            v.sort(key=lambda r: r["index"])
        ver = {r["image_id"]: r for r in read_jsonl(
            reader_file(root, "verify", tag, reader))}
        self.known_keep = known_keep
        self.p_yes_threshold = p_yes_threshold
        self._ver = ver

    def pool_rows(self, pool_seed):
        return self._pools[int(pool_seed)]

    def generate(self, prompt_text, n, seed, country=None):
        rows = self._pools.get(int(seed))
        if rows is None:
            raise KeyError(f"{self.tag}: pool {seed} was never generated")
        if len(rows) != n:
            raise ValueError(f"{self.tag}: pool {seed} has {len(rows)} images, "
                             f"analysis asked for {n}")
        out = []
        for r in rows:
            if r["select_text"] != prompt_text or r["country"] != country:
                raise ValueError(f"{self.tag}: pool {seed} was generated for "
                                 f"{r['country']!r}/{r['select_text']!r}, replayed "
                                 f"as {country!r}/{prompt_text!r}")
            art, src = None, None
            if r.get("dish") is not None:
                src = "known_by_construction"
                if self.known_keep == "all":
                    art = r["dish"]
                else:
                    v = self._ver.get(r["image_id"])
                    if v is None:
                        raise KeyError(f"{r['image_id']}: known-label image not verified")
                    # Forced-choice keep (see gpu_backends.VERIFIER_MC_PROMPT);
                    # the yes/no probability only where no choice was recorded.
                    ok = v["kept"] if "kept" in v else v["p_yes"] >= self.p_yes_threshold
                    art = r["dish"] if ok else None
            out.append(CachedImage(r["image_id"], r["country"],
                                   os.path.join(self.root, r["path"]), art, src))
        return out


class CachedScorer:
    def __init__(self, root, tag, scorer):
        rows = read_jsonl(os.path.join(root, "scores", f"{tag}__{scorer}.jsonl"))
        if not rows:
            raise FileNotFoundError(f"no {scorer} scores for {tag}")
        self._s = {r["image_id"]: (float(r["score"]), r["text"]) for r in rows}
        self.name = scorer

    def score(self, images, prompt_text=None):
        out = []
        for im in images:
            s, text = self._s[im.image_id]
            if prompt_text is not None and text != prompt_text:
                raise ValueError(f"{im.image_id}: scored against {text!r}, "
                                 f"replayed against {prompt_text!r}")
            out.append(s)
        return np.asarray(out)


class CachedTagger:
    """Replays the VLM's OPEN answers, mapped to CSpace by `VocabMatcher`.

    No `scores` argument, as for the real tagger: the answers were produced
    before any score existed on disk, and the mapping here is text-only.
    """

    def __init__(self, root, tag, all_vocab, reader="qwen"):
        rows = read_jsonl(reader_file(root, "tags", tag, reader))
        if not rows:
            raise FileNotFoundError(f"no tags for {tag}")
        self._r = {r["image_id"]: r for r in rows}
        self.matcher = VocabMatcher(all_vocab)
        self.in_vocab = {}

    def tag_one(self, image_id):
        r = self._r[image_id]
        # The raw reply is the record; the parsed fields written at tagging
        # time are a convenience.  Re-parsing here means a fix to the parser
        # or the normaliser (round 3: native-script names were parsed to
        # None) reaches every stored reply without re-running the VLM.
        if r.get("reply"):
            dish, country = parse_tagger_json(r["reply"])
        else:
            dish, country = r.get("dish"), r.get("country")
        label, inv = self.matcher.match(dish)
        self.in_vocab[image_id] = inv
        if label is None:
            return ("UNKNOWN", "UNKNOWN", TAGGER_ERROR)
        return ("CONTINENT", normalize_country(country), label)

    def tag(self, images, prompt_text=None, examples=None):
        return [self.tag_one(im.image_id) for im in images]


def load_embeddings(root, tag, name="dinov2"):
    z = np.load(os.path.join(root, "embed", f"{tag}__{name}.npz"), allow_pickle=False)
    return {i: e for i, e in zip(z["ids"].tolist(), z["E"])}
