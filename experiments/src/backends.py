"""Model backends, behind narrow interfaces so the analysis pipeline can be
exercised end-to-end on CPU with a mock before any GPU time is spent.

Three interfaces:

  Generator   .generate(prompt_text, n, seed) -> list[PIL.Image]
  Scorer      .score(images, prompt_text) -> np.ndarray, and optionally
              .image_embeddings / .text_embeddings / .weight_direction
  GeoTagger   .tag(images, prompt_text, examples) -> list[(continent, country, artifact)]

`MockGenerator` / `MockScorer` / `MockGeoTagger` implement a *planted* ground
truth: each mock image carries a latent artifact label drawn from a per-country
Zipf distribution, and the mock scorer prefers prototypical artifacts by a
tunable amount.  That makes the whole analysis path testable, and gives a
positive control -- if the pipeline cannot recover a planted effect of known
size, it cannot be trusted to report a real one.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

__all__ = ["MockGenerator", "MockScorer", "MockGeoTagger", "MockImage",
           "load_generator", "load_scorer", "load_geotagger",
           "generator_fingerprint"]


def generator_fingerprint(name, config):
    """A short, stable id for *these weights under this configuration*.

    Round-2 code review, P1: the human-audit join key did not name the
    generator, so two models' pools collided and one model's adjudicated labels
    were accepted as the other's.  The key now carries this string, and it must
    change whenever the images would change -- which includes the sampler, step
    count, guidance scale, resolution and dtype, not only the checkpoint.  The
    digest is over a canonical JSON rendering of the configuration so that key
    order cannot produce two ids for one setting.

    `#` is excluded from the result because it separates ID components.
    """
    blob = json.dumps({"name": str(name), "config": config}, sort_keys=True,
                      default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return "%s@%s" % (str(name).replace("#", "_"), digest)


def image_seeds_for_pool(pool_seed, n):
    """`n` reproducible, pool-disjoint image seeds.

    Two different `pool_seed` values yield disjoint streams with probability
    1 - O(n^2 / 2^64), which is the property "independent pools" requires and
    which `pool_seed + i` does not have.
    """
    ss = np.random.SeedSequence(entropy=int(pool_seed) & 0xFFFFFFFF,
                                spawn_key=(int(pool_seed) >> 32,))
    return ss.generate_state(n, dtype=np.uint32).astype(np.int64)


class MockImage:
    """Stand-in for a PIL image that carries its own ground-truth label."""

    __slots__ = ("artifact", "country", "embedding", "fidelity", "nbytes")

    def __init__(self, artifact, country, embedding, fidelity, nbytes):
        self.artifact = artifact
        self.country = country
        self.embedding = embedding
        self.fidelity = fidelity
        self.nbytes = nbytes


class MockGenerator:
    """Samples artifacts from a per-country Zipf distribution over a vocabulary.

    `rare_artifact_fidelity` < 1 plants the *fidelity confound*: rarer
    artifacts are rendered worse.  Set it to 1.0 to plant a pure-taste world,
    and below 1.0 to check that the fidelity controls actually separate the two.
    """

    def __init__(self, vocab_by_country, dim=64, zipf_s=1.1, seed=0,
                 rare_artifact_fidelity=1.0, name="mock-generator"):
        self.vocab = {c: list(v) for c, v in vocab_by_country.items()}
        # Part of the human-audit join key: two mock worlds that differ in seed
        # or vocabulary are different generators and must not share image IDs.
        self.generator_id = generator_fingerprint(name, {
            "vocab": {c: list(v) for c, v in sorted(self.vocab.items())},
            "dim": dim, "zipf_s": zipf_s, "seed": seed,
            "rare_artifact_fidelity": rare_artifact_fidelity})
        self.dim = dim
        self.zipf_s = zipf_s
        self.rare_artifact_fidelity = rare_artifact_fidelity
        rng = np.random.default_rng(seed)
        # Keyed by artifact NAME, not (country, name): an artifact that appears
        # in two countries' vocabularies is the same thing and must look the
        # same, otherwise cross-country convergence (A5) cannot be tested.
        self._emb = {}
        for a in sorted({a for v in self.vocab.values() for a in v}):
            self._emb[a] = rng.normal(size=dim)
        self._p = {}
        for c, v in self.vocab.items():
            w = 1.0 / np.power(np.arange(1, len(v) + 1), zipf_s)
            self._p[c] = w / w.sum()

    def artifact_embedding(self, country, artifact):
        """`country` is accepted for call-site symmetry; embeddings are global."""
        return self._emb[artifact]

    def generate(self, prompt_text, n, seed, country=None):
        rng = np.random.default_rng(seed)
        v = self.vocab[country]
        idx = rng.choice(len(v), size=n, p=self._p[country])
        out = []
        for i in idx:
            a = v[i]
            rank_frac = i / max(1, len(v) - 1)
            fid = 1.0 - (1.0 - self.rare_artifact_fidelity) * rank_frac
            e = self._emb[a] + 0.1 * rng.normal(size=self.dim)
            out.append(MockImage(a, country, e, fid, int(50_000 + 10_000 * rng.random())))
        return out


class MockScorer:
    """Linear scorer with a planted taste direction plus a fidelity term.

    score(x) = taste_weight * <w, e(x)> + fidelity_weight * fidelity(x) + noise

    `w` is set to the population prototypicality direction, so the planted
    "taste" is a genuine preference for prototypical artifacts.  Turning
    `taste_weight` to 0 gives a scorer whose only preference is render quality
    -- the null world in which Claim A should NOT fire.
    """

    def __init__(self, w, taste_weight=1.0, fidelity_weight=0.0, noise=0.05, seed=0):
        self.w = np.asarray(w, dtype=np.float64)
        self.taste_weight = taste_weight
        self.fidelity_weight = fidelity_weight
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    @property
    def weight_direction(self):
        return self.w

    def image_embeddings(self, images):
        return np.stack([im.embedding for im in images])

    def score(self, images, prompt_text=None):
        E = self.image_embeddings(images)
        fid = np.array([im.fidelity for im in images])
        return (self.taste_weight * (E @ self.w)
                + self.fidelity_weight * fid
                + self.noise * self.rng.normal(size=len(images)))


class MockGeoTagger:
    """Reads the planted label, with tunable per-country error AND label merging.

    Two confounds are plantable here.

    `error_rate_by_country` plants differential measurement error (confound
    iii): if the tagger is worse for Nigeria than for Italy, a heterogeneity
    result can appear with no real effect.

    `merge_top_frac` / `merge_strength` plant what the Phase-4 reviewer called
    the single most likely false finding (R1): **score-dependent label
    merging**.  A vocabulary-primed VLM tends to give polished, canonical-
    looking images a familiar label and ambiguous images a guessed one, so
    several genuinely different dishes collapse onto one name exactly in the
    high-scoring set.  An exact-match Vendi score converts that annotation
    behaviour directly into "homogenisation" even when selection removed
    nothing.  Turning this on with `taste_weight=0` produces a world where the
    machine endpoint fires and the audit endpoint does not -- which is the
    control the reviewer demanded, made executable.
    """

    def __init__(self, error_rate_by_country=None, seed=0,
                 merge_top_frac=0.0, merge_strength=1.0, canonical_by_country=None):
        self.err = error_rate_by_country or {}
        self.rng = np.random.default_rng(seed)
        self.merge_top_frac = merge_top_frac
        self.merge_strength = merge_strength
        self.canonical = canonical_by_country or {}

    def tag(self, images, prompt_text=None, examples=None, scores=None):
        """Returns (continent, country, artifact) per image.

        `scores` is optional and is used ONLY to plant the merging confound.
        A real tagger must never see the selector's scores; `load_geotagger`
        does not accept them.
        """
        merge_idx = set()
        if scores is not None and self.merge_top_frac > 0:
            sc = np.asarray(scores, dtype=np.float64)
            n_top = int(round(self.merge_top_frac * sc.size))
            if n_top:
                merge_idx = set(np.argsort(-sc, kind="stable")[:n_top].tolist())
        out = []
        for i, im in enumerate(images):
            if self.rng.random() < self.err.get(im.country, 0.0):
                out.append(("UNKNOWN", "UNKNOWN", "__tagger_error__"))
                continue
            artifact = im.artifact
            if i in merge_idx and self.rng.random() < self.merge_strength:
                artifact = self.canonical.get(im.country, f"canonical_{im.country}")
            out.append(("CONTINENT", im.country, artifact))
        return out


# ------------------------------------------------------------------ real backends

def load_generator(name, device="cuda", dtype="float16", gen_config=None):
    """Real T2I generator.  Requires diffusers + GPU; not importable on CPU-only.

    `gen_config` is the generator block from the config file (steps, guidance
    scale, sampler, resolution, negative prompt).  It is not merely recorded:
    it goes into `generator_id`, which is part of the human-audit join key, and
    the per-call keyword arguments are checked against it so that a run cannot
    quietly generate at different settings than the ones its image IDs claim.
    """
    import torch
    from diffusers import AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained(
        name, torch_dtype=getattr(torch, dtype), use_safetensors=True)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    cfg = dict(gen_config or {})
    cfg.pop("id", None)
    cfg["dtype"] = dtype

    class _Gen:
        model_name = name
        generator_id = generator_fingerprint(name, cfg)
        config = cfg

        def generate(self, prompt_text, n, seed, country=None, **kw):
            for key, want in self.config.items():
                if key in kw and kw[key] != want:
                    raise ValueError(
                        f"{name}: generate() was given {key}={kw[key]!r} but "
                        f"generator_id was computed with {key}={want!r}; the "
                        "audit join key would name settings these images were "
                        "not produced at")
            # Per-image seeds must be DISJOINT across pools.  `seed + i` is not:
            # with pool seeds one apart and n=64, adjacent pools share 63 of
            # their 64 image seeds, so "four independent pools" would be four
            # heavily overlapping ones and every pool-level interval would be
            # too narrow.  A SeedSequence hierarchy gives statistically
            # independent, reproducible, non-overlapping streams.
            image_seeds = image_seeds_for_pool(seed, n)
            gens = [torch.Generator(device=device).manual_seed(int(s))
                    for s in image_seeds]
            out = pipe(prompt=[prompt_text] * n, generator=gens, **kw).images
            self.last_image_seeds = [int(s) for s in image_seeds]
            return out

    return _Gen()


def load_scorer(name, device="cuda"):
    """Real preference scorer.  See `neutralize.py` for which are linear."""
    raise NotImplementedError(
        f"no scorer backend for {name!r}: PickScore/ImageReward need GPU "
        "weights that were not reachable from the session that wrote this "
        "repository.  README lists the exact checkpoints and the per-scorer "
        "affine-form check that must pass before `neutralize.py` is used.")


def load_geotagger(name, device="cuda"):
    """Open VLM geo-tagger (Qwen2.5-VL / InternVL).

    Note the signature: there is NO `scores` argument, and there must never be
    one.  Only `MockGeoTagger.tag` accepts scores, and only so that the
    score-dependent label merging the Phase-4 reviewer named as the likeliest
    false finding can be planted and then detected.
    """
    raise NotImplementedError(
        f"no geo-tagger backend for {name!r}: an open VLM and its weights were "
        "not reachable from the session that wrote this repository.  See "
        "README for the checkpoint and the prompting protocol.")
