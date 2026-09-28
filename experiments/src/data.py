"""CUBE data loading, and construction of the UNDER-SPECIFIED prompt set the
diversity protocol actually requires.

## The design correction that this module exists to enforce

CUBE ships two artifacts and they are not interchangeable:

  * **CUBE-1K** -- 1002 prompts, each of which NAMES A SPECIFIC ARTIFACT:
        "A high resolution image of carne de panela from Brazilian cuisine, realistic"
    The README says these "enable evaluation of cultural AWARENESS".  They are
    a faithfulness benchmark.  Generating N samples from one of these prompts
    yields N renderings of *the same dish*, so artifact-level diversity is ~1
    by construction and there is nothing for a selector to homogenise.
    **Using CUBE-1K as the diversity prompt set would guarantee a null result
    for reasons that have nothing to do with the hypothesis.**

  * **CUBE-CSpace** -- ~295k (country, domain, artifact) records, described in
    the README as "grounding for diversity evaluation".  It is the ground-truth
    vocabulary, and the source of the in-context artifact examples the geo-tagger
    is given.

The released `cultural_diversity.ipynb` is explicit about the prompt form:

    "Use 'sample_prompt' for input prompt. The prompt needs to be under-specified."
    sample_prompt = "Image of traditional clothing"
    num_images = 16

So the diversity prompt set must be *constructed*: country-conditioned but
artifact-underspecified.  That is what `build_underspecified_prompts` does.
CUBE-1K is retained only as the faithfulness control (does neutralisation hurt
the model's ability to render a *named* artifact?).

## Domain coverage

CSpace covers three domains -- cuisine, landmark, art.  CUBE-1K additionally
contains 222 "landscapes" prompts with NO CSpace grounding, so landscapes
cannot be scored against a ground-truth vocabulary and are excluded from the
diversity analysis.

## Why cuisine is the primary domain

Unique CSpace artifacts per country (counted from the released CSpace):

    country          cuisine     landmark        art
    India              1,413       15,112        186
    Italy                876       49,039         92
    Japan                622        5,515        199
    Turkey               384        2,334         69
    France               880       78,228        218
    United States        634       33,010        119
    Brazil               217        6,649        184
    Nigeria              196        1,193         91

The landmark vocabulary spans 1,193 (Nigeria) to 78,228 (France) -- a 66x
range that tracks Wikidata/Wikipedia coverage density far more than it tracks
cultural richness.  Running the headline analysis on landmarks would confound
"aesthetic selection homogenises cultures" with "Wikidata is Eurocentric".
Cuisine spans 196 to 1,413, a 7x range, and supplies the largest CUBE-1K
prompt set (517 of 1002).  **Cuisine is the primary domain; art is the
secondary replication; landmarks are reported only as the confounded case, with
the coverage disparity stated.**
"""

from __future__ import annotations

import collections
import json
import os
import tarfile
from dataclasses import dataclass

__all__ = [
    "COUNTRIES", "DIVERSITY_DOMAINS", "load_cube_1k", "load_cspace",
    "cspace_vocabulary", "artifact_frequencies", "build_underspecified_prompts",
    "Prompt",
]

COUNTRIES = ["Brazil", "France", "India", "Italy", "Japan", "Nigeria",
             "Turkey", "United States"]

# CSpace domain name -> how the domain is spoken about in a prompt.
DIVERSITY_DOMAINS = {
    "cuisine": "a traditional dish from {country}",
    "art": "a traditional work of art from {country}",
    "landmark": "a famous landmark in {country}",
}

# Adjectival forms, used for the second prompt template so results are not an
# artefact of one phrasing.
DEMONYM = {
    "Brazil": "Brazilian", "France": "French", "India": "Indian",
    "Italy": "Italian", "Japan": "Japanese", "Nigeria": "Nigerian",
    "Turkey": "Turkish", "United States": "American",
}


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str
    country: str
    domain: str
    template: str


def load_cube_1k(path):
    with open(path) as fh:
        return json.load(fh)


def load_cspace(path):
    """Load CUBE-CSpace from either the extracted JSON or the shipped tar.gz."""
    if os.path.isdir(path):
        path = os.path.join(path, "cube_CSpace.json.tar.gz")
    if path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as tf:
            member = next(m for m in tf.getmembers()
                          if m.name.endswith("CUBE_CSpace.json")
                          and not os.path.basename(m.name).startswith("._"))
            with tf.extractfile(member) as fh:
                return json.load(fh)
    with open(path) as fh:
        return json.load(fh)


def cspace_vocabulary(cspace, country=None, domain=None):
    """Sorted unique artifact names, optionally filtered by country/domain."""
    out = set()
    for r in cspace:
        if country is not None and r.get("country") != country:
            continue
        if domain is not None and r.get("domain") != domain:
            continue
        name = r.get("name")
        if name and name != "None":
            out.add(name)
    return sorted(out)


def artifact_frequencies(cspace, country, domain):
    """Artifact -> number of CSpace records.

    **This is NOT a usable prominence proxy and must not be used as one.**
    Measured on the released CSpace, the distribution is degenerate: for
    Nigeria/Italy/India cuisine the counts are {1: 193, 2: 3}, {1: 874, 2: 2}
    and {1: 1411, 2: 2} respectively -- essentially every artifact appears
    exactly once, because CSpace is a deduplicated KB extraction, not a corpus
    frequency table.  The function is kept for completeness and for the
    degeneracy check in `kb_prominence`.
    """
    c = collections.Counter(
        r["name"] for r in cspace
        if r.get("country") == country and r.get("domain") == domain
        and r.get("name") and r["name"] != "None"
    )
    return dict(c)


def build_underspecified_prompts(countries=None, domains=None):
    """Country-conditioned, artifact-UNDER-specified prompts.

    Two templates per (country, domain) so that no finding rests on a single
    phrasing, plus the global (country-free) template that reproduces CUBE's
    own example and supports the cross-culture convergence analysis (A5) under
    the continent-level kernel.
    """
    countries = list(countries or COUNTRIES)
    domains = list(domains or DIVERSITY_DOMAINS)
    prompts = []
    for domain in domains:
        noun = DIVERSITY_DOMAINS[domain]
        for country in countries:
            a = f"An image of {noun.format(country=country)}"
            prompts.append(Prompt(f"{domain}|{country}|A", a, country, domain, "A"))
            adj = DEMONYM[country]
            b_noun = {"cuisine": f"a traditional {adj} dish",
                      "art": f"a traditional {adj} work of art",
                      "landmark": f"a famous {adj} landmark"}[domain]
            prompts.append(Prompt(f"{domain}|{country}|B", f"An image of {b_noun}",
                                  country, domain, "B"))
        # CUBE's own global form: no country named at all.
        g = {"cuisine": "An image of a traditional dish",
             "art": "An image of a traditional work of art",
             "landmark": "An image of a famous landmark"}[domain]
        prompts.append(Prompt(f"{domain}|GLOBAL|G", g, "GLOBAL", domain, "G"))
    return prompts


def kb_prominence(cspace, country, domain):
    """Artifact -> a weak but real Wikidata prominence score in [0, 1].

    Two signals actually carry information in the released CSpace:

      * property richness -- how many of P31/P279/P495/P17/P361 are non-empty
        (observed distribution for Nigeria cuisine: 0:13, 2:38, 3:110, 4:29, 5:9);
      * whether the entity has a Wikipedia `title` (observed ~50% for Nigeria
        cuisine, ~40% for Italy/France cuisine).

    Score = (richness / 5 + has_title) / 2.  This is a knowledge-base
    prominence proxy, not a real-world or corpus frequency, and an artifact can
    be culturally central while KB-sparse.  It is reported as a secondary
    prototypicality proxy only; the primary one is `TEXT_PROTOTYPICALITY`
    below, which needs no frequency signal at all.
    """
    props = ("P31", "P279", "P495", "P17", "P361")
    out = {}
    for r in cspace:
        if r.get("country") != country or r.get("domain") != domain:
            continue
        name = r.get("name")
        if not name or name == "None":
            continue
        rich = sum(1 for k in props if r.get(k) not in ("[]", "nan", None, ""))
        titled = 1.0 if r.get("title") not in ("None", None, "") else 0.0
        out[name] = max(out.get(name, 0.0), (rich / len(props) + titled) / 2.0)
    return out


TEXT_PROTOTYPICALITY = """The primary prototypicality proxy, computed in
`pipeline.py` because it needs the text encoder:

    proto(a) = cos( E_text(a) , E_text(underspecified prompt) )

i.e. how strongly the artifact name "moqueca" already reads as "a traditional
dish from Brazil" to the *scorer's own text encoder*.  This needs no frequency
data, no images, no labels and no training; it is defined for every artifact in
CSpace; and it is exactly the quantity a CLIP-family scorer is sensitive to,
which is what makes the mechanism prediction in `predict_country_loss`
falsifiable rather than merely plausible.  `kb_prominence` above is the
robustness check: the differential subspace estimated from the two proxies
should largely agree, and if it does not, that disagreement is reportable."""
