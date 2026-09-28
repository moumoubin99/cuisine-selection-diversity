"""Turning a VLM's free-text answer into a label the diversity metric can count.

The tagger is asked an OPEN question ("what dish is this, and from which
country?") with no vocabulary list in the prompt.  The pre-registered design
primed it with 32 CSpace names from the prompted country; that priming is the
mechanism behind the reviewer's R1 (a vocabulary-primed tagger gives polished,
canonical-looking images a familiar name), and with the human audit removed
there is no second annotator to catch it.  Open answers are mapped onto CSpace
afterwards, here, by rules that never look at the image or its score -- the
same order as the human protocol it replaces ("open-ended identity first,
vocabulary mapping second").

What the mapping does NOT do is resolve synonyms that share no string ("cheese
bread" vs "pao de queijo").  Those count as two labels.  That inflates measured
diversity equally in a top-k set and a random one unless the tagger's naming
depends on the score, and whether it does is exactly what the known-label pools
measure (`src/known_label.py`).
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata

__all__ = ["normalize", "normalize_country", "VocabMatcher", "parse_tagger_json",
           "strip_presentation", "UNKNOWN"]

UNKNOWN = "unknown"

_ARTICLES = re.compile(r"^(the|a|an)\s+")
_PAREN = re.compile(r"\([^)]*\)")
# Any letter or digit in any script survives; only punctuation and symbols go.
# Round-3 code review, P1: the ASCII-only class turned every native-script
# name (57 Japanese and 6 Indian CSpace entries, and any reply Qwen writes in
# kana, kanji or Devanagari) into the empty string, i.e. an UNRESOLVED tag.
# (Unicode categories L*, N* and M*: the vowel signs of Indic scripts are
# spacing marks, which `\w` does not match.)
def _strip_nonword(s):
    return "".join(ch if ch == " " or unicodedata.category(ch)[0] in "LNM" else " "
                   for ch in s)
_SPACE = re.compile(r"\s+")

_COUNTRY_ALIASES = {
    "usa": "United States", "us": "United States", "u s": "United States",
    "united states": "United States", "united states of america": "United States",
    "america": "United States", "american": "United States",
    "turkiye": "Turkey", "turkey": "Turkey", "turkish": "Turkey",
    "brazil": "Brazil", "brasil": "Brazil", "brazilian": "Brazil",
    "france": "France", "french": "France",
    "india": "India", "indian": "India",
    "italy": "Italy", "italian": "Italy",
    "japan": "Japan", "japanese": "Japan",
    "nigeria": "Nigeria", "nigerian": "Nigeria",
    "uk": "United Kingdom", "england": "United Kingdom",
    "britain": "United Kingdom", "great britain": "United Kingdom",
}


def normalize(name):
    """Lower-case, strip accents, parentheticals, punctuation and a leading article."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    # Accents are dropped from Latin letters only ("feijão" == "feijao").  In
    # kana the combining mark is a different sound (そば is not そは) and in
    # Devanagari the vowel signs are combining marks, so there they stay.
    kept = []
    for ch in s:
        if unicodedata.combining(ch) and kept and ord(kept[-1]) < 0x250:
            continue
        kept.append(ch)
    s = unicodedata.normalize("NFC", "".join(kept)).lower()
    s = _PAREN.sub(" ", s)
    s = s.replace("&", " and ").replace("-", " ").replace("_", " ")
    s = _strip_nonword(s)
    s = _SPACE.sub(" ", s).strip()
    s = _ARTICLES.sub("", s)
    return s


def normalize_country(name):
    """Canonical country name, or the input title-cased if it is not one we know."""
    n = normalize(name)
    if not n or n in ("unknown", "none", "n a", "unclear"):
        return UNKNOWN
    if n in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[n]
    # "Southern India", "Northern Italy", "Japanese cuisine" ...
    for tok in n.split():
        if tok in _COUNTRY_ALIASES and len(tok) > 3:
            return _COUNTRY_ALIASES[tok]
    return " ".join(w.capitalize() for w in n.split())


def _one_edit(x, y):
    """True if `y` is `x` with one insertion, deletion, substitution or
    adjacent transposition (Damerau-Levenshtein distance 1)."""
    if x == y or abs(len(x) - len(y)) > 1:
        return False
    if len(x) == len(y):
        diff = [i for i in range(len(x)) if x[i] != y[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and x[diff[0]] == y[diff[1]]
                                  and x[diff[1]] == y[diff[0]])
    if len(x) > len(y):
        x, y = y, x
    i = 0
    while i < len(x) and x[i] == y[i]:
        i += 1
    return x[i:] == y[i + 1:]


#: A word shorter than this is never spelling-corrected: at four letters one
#: edit is usually another word ("beet"/"beef", "rice"/"ride").
MIN_EDIT_WORD_LEN = 5


def _spelling_variant(a, b, vocab_words):
    """True if answer `a` is CSpace name `b` misspelt: the same number of
    words, and every differing word is at least `MIN_EDIT_WORD_LEN` letters,
    one edit from its counterpart, and not itself a word of the vocabulary
    (a vocabulary word is a different ingredient, not a typo).

    Round-5 code review, P1: the earlier rule (difflib ratio >= 0.75 per word)
    mapped "beet and broccoli" onto "beef and broccoli"."""
    wa, wb = a.split(), b.split()
    if len(wa) != len(wb) or wa == wb:
        return False
    return all(x == y or (min(len(x), len(y)) >= MIN_EDIT_WORD_LEN
                          and x not in vocab_words and _one_edit(x, y))
               for x, y in zip(wa, wb))


# Presentation phrases that describe the serving, not the dish.  Round-5 code
# review, P1: "a bowl of beef and broccoli" stayed a separate label from "beef
# and broccoli", so a tagger that describes polished images more verbosely
# would have inflated their measured diversity.  Applied only after an exact
# match on the full answer has failed, so a CSpace name that begins with one
# of these words ("plate lunch") is untouched.
_CONTAINERS = ("bowl|plate|dish|serving|portion|platter|cup|glass|slice|piece|"
               "pot|skillet|tray|basket|box|stack|pile|jar|mug|order|helping|"
               "spoonful|scoop|loaf|sandwich of|side")
_PRESENTATION = re.compile(
    r"^(?:(?:a|an|the|some|two|three|several)\s+)?"
    r"(?:(?:small|large|big|hot|steaming|full)\s+)?"
    r"(?:%s)s?\s+of\s+" % _CONTAINERS)
_QUALIFIER = re.compile(
    r"^(?:traditional|classic|homemade|home made|authentic|fresh|freshly made|"
    r"delicious|typical|simple|plated|served|homestyle|home style)\s+")


_DEMONYM_COUNTRY = {"american": "United States", "brazilian": "Brazil",
                    "french": "France", "indian": "India", "italian": "Italy",
                    "japanese": "Japan", "nigerian": "Nigeria", "turkish": "Turkey"}


def _strip_steps(n):
    """The normalised name and each successively stripped form of it, in
    order; the last is `strip_presentation(n)`."""
    out = [n]
    prev = None
    while n != prev:
        prev = n
        for rx in (_PRESENTATION, _QUALIFIER, _ARTICLES):
            m = rx.match(n)
            if m and n[m.end():].strip():
                n = n[m.end():].strip()
                out.append(n)
    return out


def strip_presentation(n):
    """Removes leading serving phrases and qualifiers from a NORMALISED name;
    returns the input unchanged if nothing would be left."""
    return _strip_steps(n)[-1]


class VocabMatcher:
    """Maps a free-text dish name onto a CSpace label, or returns it normalised.

    **A label is a normalised name** (`normalize`), not a CSpace display
    string.  Round-6 code review, P1: keyed by the first CSpace string seen,
    "soba" returned Brazil's "Sobá" and was then classified as unlisted in a
    Japan pool.  CSpace names that normalise alike (21 keys: mostly case
    variants such as "Praline"/"praline", three names shared by two countries
    such as soba, cioppino and sarapatel, and one parenthetical
    disambiguation, "pâté" / "Pâté (pâtisserie)") are one label.  Whether a
    label belongs to the prompted country is decided later against that
    country's normalised vocabulary (`pipeline.classify_tags`).  Prompted
    labels in the known-label pools are normalised the same way.

    Rules, in order, none of which sees the image or its score:

    1. an exact match of the normalised answer;
    2. the same after each step of stripping leading presentation phrases
       ("a bowl of", "traditional"; `strip_presentation`) -- checked after
       every step, so "a bowl of fresh tomme" finds "fresh tomme";
    3. dropping a leading demonym, only if the rest is an exact name in THAT
       country's vocabulary ("japanese ramen" -> ramen; "japanese curry" is
       not India's curry).  Needs `vocab` as a {country: names} dict;
    4. a unique CSpace name whose normalised form differs only by a plural
       's' or by word order;
    5. a unique CSpace name of which the answer is a word-by-word spelling
       variant (`_spelling_variant`: one edit per differing word, words of at
       least five letters, never onto a word the vocabulary itself uses), and
       only if that name is not in a crowded neighbourhood.  Round-4 code
       review: CSpace holds families of distinct dishes that differ by one
       word (e.g. "alu diye bhola machher jhol" and "alu diye ban machher
       jhol", which name different fish).

    Anything else is returned as its normalised, presentation-stripped free
    text and is later classified `plausible_unlisted` -- right country, not in
    CSpace -- which the reviewer's R3 insists is not the same as inauthentic.
    """

    def __init__(self, vocab, cutoff=0.9):
        self.cutoff = cutoff
        by_country = vocab if isinstance(vocab, dict) else {None: list(vocab)}
        self.names = {}
        self._country_keys = {}
        for c, names in by_country.items():
            for v in names:
                k = normalize(v)
                if k:
                    self.names.setdefault(k, set()).add(v)
                    self._country_keys.setdefault(c, set()).add(k)
        self._keys = sorted(self.names)
        self._words = {w for k in self._keys for w in k.split()}
        self._by_len = {}
        for k in self._keys:
            self._by_len.setdefault(len(k.split()), []).append(k)
        self._bag = {}
        for k in self._keys:
            self._bag.setdefault(self._bagkey(k), []).append(k)
        self._crowded = {k for k in self._keys
                         if len(difflib.get_close_matches(k, self._keys, n=2,
                                                          cutoff=cutoff)) > 1}
        self._demonym_country = ({d: c for d, c in _DEMONYM_COUNTRY.items()
                                  if c in self._country_keys}
                                 if isinstance(vocab, dict) else {})
        self._cache = {}

    @staticmethod
    def _bagkey(s):
        return " ".join(sorted(w[:-1] if len(w) > 3 and w.endswith("s") else w
                               for w in s.split()))

    def match(self, name):
        """Returns (label, in_vocab)."""
        n = normalize(name)
        if not n or n == UNKNOWN:
            return None, False
        if n not in self._cache:
            self._cache[n] = self._match(n)
        return self._cache[n]

    def _match(self, n):
        for step in _strip_steps(n):
            if step in self.names:
                return step, True
        n = strip_presentation(n)
        head, _, rest = n.partition(" ")
        c = self._demonym_country.get(head)
        if rest and c is not None:
            for step in _strip_steps(rest):
                if step in self._country_keys[c]:
                    return step, True
        hits = self._bag.get(self._bagkey(n), [])
        if len(hits) == 1:
            return hits[0], True
        close = [k for k in self._by_len.get(len(n.split()), [])
                 if _spelling_variant(n, k, self._words)]
        if len(close) == 1 and close[0] not in self._crowded:
            return close[0], True
        return n, False


_JSON = re.compile(r"\{.*?\}", re.S)


def parse_tagger_json(text):
    """(dish, country) from the tagger's reply; (None, None) if unparseable.

    Tolerates code fences and trailing prose, which 7B VLMs produce even when
    told not to.  An unparseable reply is an UNRESOLVED tag, counted as such.
    """
    if not text:
        return None, None
    m = _JSON.search(text)
    if not m:
        return None, None
    try:
        d = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None, None
    dish = d.get("dish")
    country = d.get("country")
    if not isinstance(dish, str) or normalize(dish) in ("", UNKNOWN, "none"):
        dish = None
    if not isinstance(country, str):
        country = None
    return dish, country
