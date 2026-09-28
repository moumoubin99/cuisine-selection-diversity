"""End-to-end audit: generate -> score -> tag -> analyse.

Structure follows the Phase-4 critical review (idea-stage/reviews/
critical_review.md), which changed three things about how this has to be run:

1. **The unit of replication is a candidate POOL**, not an image and not a
   rarefaction draw.  `run_pool` produces exactly one independent observation;
   everything above it counts pools.
2. **The contrast is on the log scale**, `delta = log D_top - mean log D_rand`,
   so that countries whose vocabularies differ 7x are comparable and the
   pre-registered "10% loss" threshold means one thing everywhere.
3. **Two endpoints are reported jointly, never one.**  The machine endpoint
   uses the VLM tagger's labels; the audit endpoint uses adjudicated labels for
   the subset that was human-annotated.  If they disagree, the machine endpoint
   is measuring the annotator, not the model.

The same code path runs with mock backends on CPU and with real backends on
GPU, so the analysis is validated against a planted ground truth before any GPU
hour is spent.
"""

from __future__ import annotations

import numpy as np

from .analysis import (MAX_NULL_BIAS_LOG, MAX_NULL_INSUFFICIENT,
                       country_marginal, dose_response, equivalence_test,
                       equivalence_test_country_marginal, heterogeneity_test,
                       heterogeneity_verdict, log_contrast, null_bias_bounds,
                       pool_contrasts_by_country, rarefy_to_common_count,
                       simultaneous_country_intervals, stereotype_shift)
from .annotations import audit_labels_for_pool, image_ids_for_pool
from .labels import normalize
from .selectors import permutation_null, random_selector, select_topk

__all__ = ["TAGGER_ERROR", "classify_tags", "run_pool", "run_audit",
           "MAX_NULL_INSUFFICIENT", "generator_id_of"]


def generator_id_of(generator):
    """The generator's contribution to the human-audit join key.

    Required, not optional.  Round-2 code review, P1: without it two models'
    pools produce identical image IDs and one model's adjudicated labels are
    accepted as the other's, reported as `human_adjudicated` at 100% coverage.
    A backend that cannot say which weights and settings produced an image
    cannot participate in an audit, so this raises rather than inventing a
    default.
    """
    gid = getattr(generator, "generator_id", None)
    if not gid:
        raise TypeError(
            f"{type(generator).__name__} has no `generator_id`.  It is part of "
            "the image-ID join key (see src/annotations.py) and must identify "
            "the weights AND the generation configuration; see "
            "`src.backends.generator_fingerprint`.")
    return str(gid)


def _reason_counts(recs):
    """Why pools dropped out -- never just how many."""
    out = {}
    for r in recs:
        if r.get("non_estimable"):
            key = str(r.get("reason", "unspecified")).split("(")[0].strip()
            out[key] = out.get(key, 0) + 1
    return out

TAGGER_ERROR = "__tagger_error__"

#: Four-way label status.  The reviewer's R3: membership in CSpace cannot
#: establish authenticity and non-membership cannot establish its absence, so
#: "in vocabulary" and "authentic" are deliberately different words here.
ESTABLISHED = "established"        # attributed to the prompted country, in CSpace
PLAUSIBLE = "plausible_unlisted"   # right country, not in CSpace -- may be a
                                   # real dish Wikidata does not record
OFF_COUNTRY = "off_country"        # attributed elsewhere
UNRESOLVED = "unresolved"          # tagger declined or failed


def _normalised_keys(d):
    """Prototypicality keyed like the labels `classify_tags` returns
    (round-7 code review, P2); names that normalise alike get their mean."""
    if not d:
        return d
    acc = {}
    for name, v in d.items():
        acc.setdefault(normalize(name), []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items() if k}


def classify_tags(tags, prompted_country, cspace_vocab):
    """Four-way status per sample, with no category silently discarded.

    Returns (statuses, labels).  `labels[i]` is the artifact name or None.
    """
    # Compared as normalised names (round-6 code review, P1): the matcher's
    # labels are normalised, and a name CSpace lists for two countries
    # ("soba", Japan and Brazil's "Sobá") is established in either.
    # Round-7 code review, P2: the returned label is normalised too, so a
    # caller passing raw names ("Praline", "praline") gets one label.
    vocab = {normalize(v) for v in cspace_vocab}
    statuses, labels = [], []
    for _, country, artifact in tags:
        if artifact != TAGGER_ERROR and isinstance(artifact, str):
            artifact = normalize(artifact)
        if artifact == TAGGER_ERROR:
            statuses.append(UNRESOLVED)
            labels.append(None)
        elif country != prompted_country:
            statuses.append(OFF_COUNTRY)
            labels.append(artifact)
        elif vocab and normalize(artifact) not in vocab:
            statuses.append(PLAUSIBLE)
            labels.append(artifact)
        else:
            statuses.append(ESTABLISHED)
            labels.append(artifact)
    return statuses, labels


def _endpoint(labels, scores, k, m_auth, n_null, rng, keep, keep_name):
    """One (endpoint, k) contrast.

    **Selection happens over the whole pool of N, then the label filter is
    applied to whatever was selected.**  The code review caught the reverse
    order here: filtering to `keep` first and selecting the top-k *within the
    survivors* measures a different intervention -- "the best of the images we
    judged usable" -- when the deployed system ranks all N and ships the top k
    whatever they turn out to depict.  The reviewer's reproduction: four
    high-scoring off-country images followed by four low-scoring established
    ones, k=4, returned delta=0 while the actual top-4 contained no established
    image at all.

    The consequence of the correct order is that each condition now has its own
    surviving count, and either side can fall below the common rarefaction
    count `m`.  Those outcomes are reported, never dropped: `non_estimable` on
    the selected side, and `null_insufficient_frac` on the null side, whose
    surviving draws are conditioned on having enough authentic images and so
    are NOT a valid reference once that fraction gets large.
    """
    n = len(labels)
    keep = list(keep)
    base = {"k": int(k), "m": int(m_auth), "filter": keep_name,
            "n_pool": int(n), "n_kept_pool": int(sum(keep))}
    if n < k:
        return {**base, "non_estimable": True, "reason": "pool smaller than k"}

    sel_idx = select_topk(scores, k)                       # over ALL N
    sel = [labels[i] for i in sel_idx if keep[i]]
    base["n_kept_selected"] = len(sel)
    base["selected_deficit"] = 1.0 - len(sel) / float(k)

    draws, insufficient, kept_null = [], 0, []
    for _ in range(n_null):
        idx = random_selector(n, k, rng)
        nl = [labels[i] for i in idx if keep[i]]
        kept_null.append(len(nl))
        v = rarefy_to_common_count(nl, m_auth, rng=rng)
        if v["non_estimable"]:
            insufficient += 1
        else:
            draws.append(v["value"])
    frac = insufficient / float(n_null) if n_null else 1.0
    base["null_insufficient_frac"] = frac
    base["mean_n_kept_null"] = float(np.mean(kept_null)) if kept_null else float("nan")
    # The selection-induced change in how many images survive the filter is
    # itself a result -- on the conditional endpoint it IS the authenticity
    # cost -- so it is recorded even when the diversity contrast is not
    # estimable.
    base["kept_shift"] = (len(sel) - base["mean_n_kept_null"]) / float(k)

    d_sel = rarefy_to_common_count(sel, m_auth, rng=rng)
    if d_sel["non_estimable"]:
        return {**base, "non_estimable": True,
                "reason": f"selected set has {len(sel)} < m={m_auth} kept images"}
    if frac > MAX_NULL_INSUFFICIENT or not draws:
        return {**base, "non_estimable": True,
                "reason": f"{frac:.1%} of null draws fell below the common count "
                          f"(> {MAX_NULL_INSUFFICIENT:.0%}); the survivors are a "
                          "biased reference"}
    out = log_contrast(d_sel["value"], draws)
    # What the DISCARDED null draws could have done to this contrast, bounded
    # without assuming anything about them.  A bracket wider than the margin
    # means the pool cannot be placed on either side of the pre-registered
    # threshold, whatever the point estimate says.
    bounds = null_bias_bounds(np.log(d_sel["value"]),
                              float(np.log(np.asarray(draws)).mean()), frac, m_auth)
    if bounds[1] - bounds[0] > MAX_NULL_BIAS_LOG:
        return {**base, "non_estimable": True, "delta_bounds": bounds,
                "reason": f"unusable null draws leave the contrast bracketed "
                          f"over {bounds[1] - bounds[0]:.4f} log units, wider "
                          f"than the {MAX_NULL_BIAS_LOG:.4f} margin"}
    out.update(base)
    out["delta_bounds"] = bounds
    out["diversity_selected"] = d_sel["value"]
    out["selected_labels"] = sel
    return out


def run_pool(generator, scorer, tagger, prompt, n, k, m_auth, cspace_vocab,
             pool_seed, prototypicality=None, n_null=200, secondary_ks=(),
             audit=False, tagger_wants_scores=False, annotations=None):
    """ONE independent candidate pool = one observation.

    The audit endpoint has two sources, and only one of them is real:

    * `annotations` -- adjudicated labels from two blinded human annotators,
      joined to this pool by image ID (see `src/annotations.py`).  This is what
      a real run uses, and a pool that is not fully annotated is skipped with
      its coverage recorded rather than audited on the part that happens to be
      covered.
    * `audit=True` with no annotations -- the label each image carries.  For
      the mock backends that is the planted ground truth, the stand-in that
      makes the label-merging control executable.  For real images it exists
      only in KNOWN-LABEL pools (src/known_label.py), where it is the dish the
      image was prompted with, filtered by a closed-question verifier, and is
      reported as `known_by_construction`.  Ordinary real pools have no such
      attribute and are refused.
    """
    # Checked FIRST, before a single image is generated: a backend that cannot
    # name itself cannot be audited, and discovering that after the GPU hours
    # is discovering it too late.
    gid = generator_id_of(generator)
    rng = np.random.default_rng(pool_seed)
    images = generator.generate(prompt.text, n, seed=pool_seed, country=prompt.country)
    scores = np.asarray(scorer.score(images, prompt.text), dtype=np.float64)
    kw = {"scores": scores} if tagger_wants_scores else {}
    tags = tagger.tag(images, prompt.text, examples=list(cspace_vocab)[:32], **kw)

    statuses, labels = classify_tags(tags, prompt.country, cspace_vocab)
    counts = {s: int(sum(1 for x in statuses if x == s))
              for s in (ESTABLISHED, PLAUSIBLE, OFF_COUNTRY, UNRESOLVED)}

    row = {
        "prompt_id": prompt.prompt_id, "country": prompt.country,
        "domain": prompt.domain, "template": prompt.template,
        "pool_seed": int(pool_seed), "N": int(n), "k": int(k), "m": int(m_auth),
        "status_counts": counts,
        "unresolved_rate": counts[UNRESOLVED] / n,
        "authenticity_deficit": 1.0 - counts[ESTABLISHED] / n,
    }

    resolved = [s != UNRESOLVED for s in statuses]
    established = [s == ESTABLISHED for s in statuses]
    # RAW: every resolved sample, whatever it was tagged as.  This is the
    # endpoint that answers "did the output set get less varied", without any
    # authenticity judgement layered on top.
    row["raw"] = _endpoint(labels, scores, k, m_auth, n_null, rng, resolved,
                           "resolved")
    # CONDITIONAL: established artifacts only.  Reported WITH the deficit above,
    # never on its own -- a condition that removes inauthentic images improves
    # things, and this endpoint alone cannot tell those cases apart.
    row["conditional"] = _endpoint(labels, scores, k, m_auth, n_null, rng,
                                   established, "established")

    row["generator_id"] = gid
    row["image_ids"] = image_ids_for_pool(gid, prompt.prompt_id, pool_seed, n)
    audit_labels, audit_source = None, None
    if annotations is not None:
        audit_labels, coverage = audit_labels_for_pool(
            gid, prompt.prompt_id, pool_seed, n, annotations)
        row["audit_coverage"] = coverage
        audit_source = "human_adjudicated"
        if audit_labels is None:
            row["audit"] = {"non_estimable": True, "filter": "audit_human",
                            "reason": "pool not fully annotated "
                                      f"({coverage['n_annotated']}/{coverage['n_images']})"}
    elif audit:
        if not all(hasattr(im, "artifact") for im in images):
            raise TypeError(
                "audit=True without `annotations` reads the generator's planted "
                "label; real images do not have one.  Supply an adjudicated "
                "annotation file instead (see src/annotations.py).")
        # Normalised like the tagger's labels (round-6 code review, P1): a
        # known-label pool may prompt both "Praline" and "praline", which are
        # one dish and must be one label on the prompted side too.  Mock
        # images keep their planted label unchanged.
        audit_labels = [(normalize(im.artifact) if im.artifact is not None else None)
                        if getattr(im, "label_source", None) else im.artifact
                        for im in images]
        # Mock images carry a planted label; replayed known-label pools
        # (src/store.py) carry the dish they were PROMPTED with, and say so.
        sources = {getattr(im, "label_source", None) or "planted_ground_truth"
                   for im in images}
        if len(sources) != 1:
            raise ValueError(f"one pool, several label sources: {sorted(sources)}")
        audit_source = sources.pop()

    if audit_labels is not None:
        row["audit_source"] = audit_source
        keep_audit = [x is not None for x in audit_labels]
        row["audit"] = _endpoint(audit_labels, scores, k, m_auth, n_null, rng,
                                 keep_audit, "audit_resolved")
        # The gap is taken on ONE image set.  Round-3 code review, P0: the
        # machine raw endpoint counted verifier-rejected images that the
        # prompted-label endpoint dropped, so if rejected renderings vary with
        # score their difference can hide or imitate label merging.  Both
        # sides of the gap therefore keep exactly the images that the tagger
        # resolved AND that carry a prompted label; the endpoint-level
        # difference above is kept as `machine_minus_audit_unmatched`.
        joint = [a and b for a, b in zip(resolved, keep_audit)]
        # Common random numbers: both sides draw the SAME random k-subsets and
        # rarefaction subsets (same mask, so same counts, so the same stream).
        # Round-4 code review: with separate draws, identical labels on
        # identical images still gave a nonzero gap -- Monte Carlo noise
        # added to a quantity tested against a 5% margin.
        crn = int(rng.integers(2 ** 63))
        row["raw_joint"] = _endpoint(labels, scores, k, m_auth, n_null,
                                     np.random.default_rng(crn),
                                     joint, "resolved_and_audit_resolved")
        row["audit_joint"] = _endpoint(audit_labels, scores, k, m_auth, n_null,
                                       np.random.default_rng(crn),
                                       joint, "resolved_and_audit_resolved")
        mr, ar = row["raw_joint"], row["audit_joint"]
        if not mr.get("non_estimable") and not ar.get("non_estimable"):
            row["machine_minus_audit"] = float(mr["delta"] - ar["delta"])
        if (not row["raw"].get("non_estimable")
                and not row["audit"].get("non_estimable")):
            row["machine_minus_audit_unmatched"] = float(
                row["raw"]["delta"] - row["audit"]["delta"])

    if secondary_ks and n >= max(secondary_ks):
        # Selection ranks the whole pool here too (see `_endpoint`); the dose
        # curve is computed on the RAW endpoint, where every resolved sample
        # counts, so that no k in the sweep is silently evaluated on a
        # different filtered subpopulation than its neighbours.
        # `keep=resolved`, NOT a pre-filtered pool: selection ranks all N here
        # exactly as it does in `_endpoint`, and the filter is applied to what
        # selection returned.  Handing this function `labels[raw_idx]` was the
        # filter-first intervention again, restored at every point of the dose
        # curve (round-2 code review, P2).
        row["dose_response"] = dose_response(
            list(labels), scores, list(secondary_ks),
            lambda kk, r: random_selector(n, kk, r),
            n_null=min(n_null, 60), rng=rng, eval_size=m_auth, keep=resolved)
        # Experiment audit run 02: the primary k and the secondary ks on one
        # estimand and one null count, so the k = 32 vs k = 16 comparison is
        # like for like.  Its own rng stream, so no other endpoint's draws move.
        row["dose_response_matched"] = dose_response(
            list(labels), scores, [k] + [x for x in secondary_ks if x != k],
            lambda kk, r: random_selector(n, kk, r),
            n_null=min(n_null, 200), rng=np.random.default_rng([int(pool_seed), 32]),
            eval_size=m_auth, keep=resolved)

    if prototypicality and not row["conditional"].get("non_estimable"):
        pool_lab = [labels[i] for i in range(n) if established[i]]
        row["stereotype_shift"] = stereotype_shift(
            pool_lab, row["conditional"]["selected_labels"], prototypicality)
        perm = [pool_lab[i] for i in permutation_null(
            np.asarray(scores)[established], k, rng)]
        row["stereotype_shift_permutation_null"] = stereotype_shift(
            pool_lab, perm, prototypicality)

    for key in ("raw", "conditional", "audit", "raw_joint", "audit_joint"):
        if key in row and isinstance(row[key], dict):
            row[key].pop("selected_labels", None)
    return row


def run_audit(generator, scorer, tagger, prompts, vocab_by_country, n, k,
              m_auth=8, n_pools=4, prototypicality_by_country=None, seed=0,
              n_null=200, secondary_ks=(), audit=False,
              tagger_wants_scores=False, margin_proportional=0.10, rng=None,
              annotations=None):
    """Replicate every prompt over `n_pools` independent pools and aggregate.

    The aggregation deliberately reports an *inconclusive* verdict rather than
    a non-significant p-value: with 8 purposively chosen countries and a handful
    of pools each, "no effect" and "no power" are different statements and the
    equivalence test is what separates them.
    """
    rng = rng or np.random.default_rng(seed)
    rows = []
    for i, p in enumerate(prompts):
        for j in range(n_pools):
            rows.append(run_pool(
                generator, scorer, tagger, p, n, k, m_auth,
                vocab_by_country[p.country], pool_seed=seed + 1000 * i + j,
                prototypicality=_normalised_keys(
                    (prototypicality_by_country or {}).get(p.country)),
                n_null=n_null, secondary_ks=secondary_ks, audit=audit,
                tagger_wants_scores=tagger_wants_scores,
                annotations=annotations))

    out = {"rows": rows, "n_pools_total": len(rows), "k": k, "N": n, "m": m_auth,
           "generator_id": generator_id_of(generator)}
    all_countries = sorted({p.country for p in prompts})
    # The design cell, not just the country.  Round-2 code review, P1: dropping
    # every pool of one template left `complete_design` True although the
    # marginal had become a single-template marginal -- a different estimand
    # reported under the same verdict vocabulary.
    all_cells = sorted({(p.country, p.template) for p in prompts})
    for endpoint in ("raw", "conditional", "audit"):
        recs = [{"country": r["country"], "template": r["template"], **r[endpoint]}
                for r in rows if endpoint in r and isinstance(r[endpoint], dict)]
        if not recs:
            continue
        by = pool_contrasts_by_country(recs)
        # The same pools split by template, so the country value is the mean of
        # its template means (round-4 code review: a country that lost more
        # B-template pools than A-template pools was re-weighted toward A).
        by_cells = {}
        for r in recs:
            if not r.get("non_estimable") and np.isfinite(r.get("delta", np.nan)):
                by_cells.setdefault(r["country"], {}).setdefault(
                    r["template"], []).append(float(r["delta"]))
        obs_cells = sorted({(r["country"], r["template"]) for r in recs
                            if not r.get("non_estimable")
                            and np.isfinite(r.get("delta", np.nan))})
        # The estimand is the EQUAL-WEIGHT country marginal, and the eight
        # countries are fixed by design rather than sampled.  Pooling every
        # surviving pool into one vector (as an earlier version did) weights
        # countries by how many pools happened to survive -- and pools go
        # missing precisely where authentic artifacts are scarce, so that
        # weighting is informative, not incidental.  `country_marginal`
        # resamples pools within each country with the country set held fixed.
        marg = country_marginal(by_cells, n_boot=2000, rng=rng,
                                expected_countries=all_countries,
                                expected_cells=all_cells, observed_cells=obs_cells)
        # The equivalence verdict is taken on the equal-weight country
        # marginal -- the unit the pre-registered 10% margin was written about
        # -- and with THAT estimator's sampling variance, which comes from
        # pools within countries.  Feeding the K country means to
        # `equivalence_test` instead (as this did) treats the eight purposively
        # chosen countries as eight exchangeable draws and measured 10.1% false
        # equivalence at a nominal 5%: see `_fixed_country_moments`.
        eq = equivalence_test_country_marginal(
            by_cells, margin_proportional=margin_proportional,
            expected_countries=all_countries)
        # Welch and the simultaneous intervals use the same template-balanced
        # country values (round-5 code review, P1: on the pooled vectors a
        # country that lost B pools was re-weighted toward A, so A2's p-value
        # and its spread bound described different country values from A1's).
        het = heterogeneity_test(by_cells, n_perm=2000, rng=rng)
        iv = simultaneous_country_intervals(by_cells, n_boot=2000, rng=rng)
        if not marg.get("complete_design", False) and eq["verdict"] != "non_estimable":
            # An incomplete design does not get a complete-design verdict.  The
            # number it would have produced is kept, clearly labelled, because
            # suppressing it entirely invites someone to recompute it by hand.
            eq = {**eq, "verdict": "non_estimable",
                  "equivalence_on_observed": {kk: eq[kk] for kk in
                                              ("verdict", "mean_delta", "ci")
                                              if kk in eq},
                  "reason": "incomplete design: missing countries %s, missing "
                            "(country, template) cells %s"
                            % (marg.get("missing_countries"),
                               marg.get("missing_cells"))}
        out[endpoint] = {
            "n_estimable": int(sum(len(v) for v in by.values())),
            "n_non_estimable": int(sum(1 for r in recs if r.get("non_estimable"))),
            "non_estimable_reasons": _reason_counts(recs),
            "marginal": marg,
            "mean_delta": marg.get("mean_delta", float("nan")),
            "mean_proportional_change": marg.get("proportional_change", float("nan")),
            "equivalence": eq,
            "heterogeneity": het,
            "country_intervals": iv,
            # The A4 decision rule, EXECUTED.  Round-2 code review, P2: this
            # function existed and was unit-tested but the runner never called
            # it, so replacing every heterogeneity p-value with 1.0 still
            # passed all nine smoke controls -- the disposition's claim that
            # "the decision rule compensates" did not describe running code.
            "heterogeneity_decision": heterogeneity_verdict(het, iv),
            "design": {kk: marg.get(kk) for kk in
                       ("complete_design", "missing_countries", "missing_cells")},
            "mean_selected_deficit": float(np.mean(
                [r["selected_deficit"] for r in recs if "selected_deficit" in r]))
                if any("selected_deficit" in r for r in recs) else float("nan"),
            "mean_kept_shift": float(np.mean(
                [r["kept_shift"] for r in recs if "kept_shift" in r]))
                if any("kept_shift" in r for r in recs) else float("nan"),
        }
    out["mean_authenticity_deficit"] = float(np.mean([r["authenticity_deficit"] for r in rows]))
    out["mean_unresolved_rate"] = float(np.mean([r["unresolved_rate"] for r in rows]))
    if any("audit" in r for r in rows):
        out["machine_minus_audit"] = _gap_test(rows, "machine_minus_audit",
                                               all_countries)
        out["machine_minus_audit_unmatched"] = _gap_test(
            rows, "machine_minus_audit_unmatched", all_countries)
    return out


GAP_MARGIN_PROPORTIONAL = 0.05


def _gap_test(rows, key, all_countries):
    """TOST of the tagger-minus-prompted-label gap at the 5% margin, on the
    equal-weight country marginal with fixed-country moments.

    Round-3 code review, P1: the pooled-gap test treated every estimable pool
    as an exchangeable replicate and had no completeness check, so four clean
    pools from two countries could declare the tagger unbiased while the other
    six countries were non-estimable.  A country with fewer than two estimable
    pools now makes the gap `non_estimable`.
    """
    by = {}
    for r in rows:
        if key in r and np.isfinite(r[key]):
            by.setdefault(r["country"], []).append(float(r[key]))
    eq = equivalence_test_country_marginal(
        by, margin_proportional=GAP_MARGIN_PROPORTIONAL,
        expected_countries=all_countries)
    gaps = [g for v in by.values() for g in v]
    # Exact agreement is DESCRIPTIVE (round-6 code review, P1, withdrawing
    # the round-5 rule that called it `equivalent`).  Sixteen zero pool gaps
    # do not bound the population gap: if a pool gap were 0 with probability
    # .9 and 1 otherwise, the mean gap would be .1 and all sixteen would still
    # be 0 with probability .185.  The verdict stays `non_estimable`, so a
    # perfect-looking tagger does not clear R1; the flag says why.
    # Round-7 code review, P2: only on the complete design (every country
    # with >= 2 pools), so a missing country is not reported as agreement.
    complete = all(len(by.get(c, [])) >= 2 for c in all_countries)
    exact = bool(eq["verdict"] == "non_estimable" and gaps and complete
                 and all(g == 0.0 for g in gaps))
    return {"mean": eq.get("mean_delta", float(np.mean(gaps)) if gaps else float("nan")),
            "exact_agreement": exact,
            "n": len(gaps), "n_pools_total": len(rows),
            "ci": eq.get("ci"), "verdict": eq["verdict"],
            "reason": eq.get("reason"),
            "country_means": eq.get("country_means"),
            "country_n_pools": eq.get("country_n_pools")}
