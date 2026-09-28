#!/usr/bin/env python3
"""Single entrypoint for the selection-diversity audit.

    python run.py --smoke                        # CPU, mocks, ~2 min, no downloads
    python run.py --config c.yaml --preflight     # validate a config, no GPU
    python run.py --config configs/pilot.yaml   # real GPU run

`--smoke` is not a toy.  It runs the entire analysis path against backends with
a planted ground truth, in six worlds:

  effect        a large taste direction (-82%)  -> verdict must be `harmful`
  small_effect  a taste direction just past the -> must NOT be called
                pre-registered margin (-12%)       `equivalent`
  sub_margin    a taste direction inside it     -> verdict must be `equivalent`
                (-1%)
  null          a pure-noise scorer             -> verdict must be `equivalent`
  merging       no taste, but the tagger merges -> the MACHINE endpoint fires
                labels in the high-scoring set     and the AUDIT does not
  heterogeneous half the countries at the cliff  -> the A4 decision rule must
                and half inside the margin          return `heterogeneous`

The last world is the one that matters most.  The Phase-4 reviewer's single most
likely false finding is score-dependent label merging: a vocabulary-primed VLM
gives polished images a familiar name, several different dishes collapse onto
one, and exact-match Vendi reports "homogenisation" that selection never
caused.  If this run cannot separate that from a real effect, no GPU hour
should be spent, because the real run could not either.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.annotations import audit_labels_for_pool, load_annotations
from src.backends import (MockGenerator, MockGeoTagger, MockScorer,
                          generator_fingerprint, load_generator,
                          load_geotagger, load_scorer)
from src.analysis import heterogeneity_verdict
from src.config import ConfigError, load_config
from src.data import Prompt, build_underspecified_prompts
from src.pipeline import run_audit

# CSpace cuisine vocabulary sizes, scaled down for CPU but keeping the real 7x
# spread so the per-country contrast is exercised, not just the pooled one.
CUISINE_VOCAB_SIZES = {"Brazil": 217, "France": 880, "India": 1413, "Italy": 876,
                       "Japan": 622, "Nigeria": 196, "Turkey": 384,
                       "United States": 634}


class _PerCountryNoiseScorer:
    """A scorer whose effect size genuinely DIFFERS by country.

    Without a world like this the A4 decision rule has nothing to act on: at
    four pools per country the simultaneous band is (correctly) non-estimable,
    so `heterogeneity_verdict` returned `non_estimable` in all five worlds and
    a control asserting only that it was called was vacuous -- which is how the
    second code review's mutation (every heterogeneity p-value set to 1.0)
    could leave every control passing.

    The country is read off the images, never off the scores, so the tagger
    blinding is untouched.
    """

    def __init__(self, w, noise_by_country, seed=0):
        self._by = {c: MockScorer(w, taste_weight=1.0, fidelity_weight=0.2,
                                  noise=v, seed=seed + 1 + i)
                    for i, (c, v) in enumerate(sorted(noise_by_country.items()))}

    def score(self, images, text=None):
        countries = {getattr(im, "country", None) for im in images}
        if len(countries) != 1:
            raise ValueError("a pool must come from exactly one country")
        return self._by[countries.pop()].score(images, text)


def _world(taste, merge_top_frac=0.0, seed=0, scale=8, noise=0.1,
           noise_by_country=None):
    sizes = {c: max(24, v // scale) for c, v in CUISINE_VOCAB_SIZES.items()}
    shared = [f"shared_{i}" for i in range(4)]
    vocab = {c: shared + [f"{c[:2].lower()}_{i}" for i in range(n)]
             for c, n in sizes.items()}
    gen = MockGenerator(vocab, seed=seed)
    w = gen.artifact_embedding(None, "shared_0")
    w = w / np.linalg.norm(w)
    # `noise` is the effect-size dial.  `taste_weight` is not: the planted
    # taste direction points at an artifact every country's vocabulary
    # contains, so above a threshold the top-k collapses onto it completely
    # and every taste_weight from 0.05 upward gives the same -82% cliff.
    # Diluting the ranking with noise is what produces a continuum, and a
    # continuum is what the near-margin controls need.
    if noise_by_country:
        scorer = _PerCountryNoiseScorer(w, {c: noise_by_country[c] for c in sizes},
                                        seed=seed)
    else:
        scorer = MockScorer(w, taste_weight=taste, fidelity_weight=0.2, noise=noise,
                            seed=seed + 1)
    tagger = MockGeoTagger({}, seed=seed + 2, merge_top_frac=merge_top_frac,
                           canonical_by_country={c: "shared_0" for c in sizes})
    prompts = [Prompt(f"cuisine|{c}|A",
                      f"An image of a traditional dish from {c}", c, "cuisine", "A")
               for c in sizes]
    proto = {c: {a: float(np.exp(-i / 12.0)) for i, a in enumerate(vocab[c])}
             for c in sizes}
    return gen, scorer, tagger, prompts, vocab, proto


#: Five planted worlds.  The first code review's finding about the previous
#: three: a -82.4% effect and a 0% null "prove nothing about sensitivity near
#: the pre-registered 10% margin", and every control could still pass with a
#: completely broken audit endpoint, because the audit endpoint was only ever
#: checked where it was supposed to DISagree with the machine.
#:
#: `noise` values are calibrated, not guessed -- measured conditional losses at
#: 8 countries x 4 pools: noise=0.1 -> -82%, 20 -> -12%, 26 -> -8%, 80 -> -0.5%.
WORLDS = {
    # Far from the margin: the path can recover a large planted effect at all.
    "effect_world":      dict(taste=1.0, noise=0.1,  merge=0.0,  audit=True),
    # Just past the margin (~-12%).  This is the sensitivity control.
    "small_effect_world": dict(taste=1.0, noise=20.0, merge=0.0,  audit=True),
    # Inside the margin (~-1%).  This is the specificity control: the verdict
    # must be `equivalent`, not merely "not harmful".
    "sub_margin_world":  dict(taste=1.0, noise=80.0, merge=0.0,  audit=True),
    # No preference at all.
    "null_world":        dict(taste=0.0, noise=0.1,  merge=0.0,  audit=True),
    # R1: no taste, only score-dependent label merging by the tagger.
    "merging_world":     dict(taste=0.0, noise=0.1,  merge=0.25, audit=True),
    # A4, made executable.  Four countries at the -82% cliff and four inside
    # the margin, run at the 8 pools per country the simultaneous band needs.
    # This is the only world in which the heterogeneity DECISION RULE has
    # anything to decide; without it the rule can be deleted and every control
    # still passes, which is what the second code review demonstrated.
    "heterogeneous_world": dict(
        taste=1.0, noise=0.1, merge=0.0, audit=True, pools=8,
        noise_by_country={c: (0.1 if i < 4 else 80.0) for i, c in
                          enumerate(sorted(CUISINE_VOCAB_SIZES))}),
}


def smoke(n=96, k=16, m_auth=8, n_pools=4, out_dir="results/smoke", seed=0,
          n_null=60):
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for name, cfg in WORLDS.items():
        gen, scorer, tagger, prompts, vocab, proto = _world(
            cfg["taste"], cfg["merge"], seed=seed, noise=cfg["noise"],
            noise_by_country=cfg.get("noise_by_country"))
        t0 = time.time()
        out = run_audit(gen, scorer, tagger, prompts, vocab, n=n, k=k,
                        m_auth=m_auth, n_pools=cfg.get("pools", n_pools),
                        prototypicality_by_country=proto, seed=seed,
                        n_null=n_null, audit=cfg["audit"],
                        tagger_wants_scores=cfg["merge"] > 0)
        out["seconds"] = round(time.time() - t0, 2)
        report[name] = out
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(_jsonable({kk: vv for kk, vv in out.items() if kk != "rows"}),
                      f, indent=2)

    ok = _check_controls(report)
    with open(os.path.join(out_dir, "controls.json"), "w") as f:
        json.dump(_jsonable(ok), f, indent=2)
    _print_controls(ok)
    return 0 if ok["passed"] else 1


def _het_decision_consistent(block):
    """The A4 rule, re-derived from the two inputs the runner stored."""
    dec = block.get("heterogeneity_decision")
    if dec is None:
        return False
    if dec["verdict"] == "non_estimable":
        return bool(block["heterogeneity"].get("non_estimable")
                    or block["country_intervals"].get("non_estimable"))
    rejects = block["heterogeneity"]["p_value"] < 0.05
    if dec["welch_rejects"] != rejects:
        return False
    return dec["verdict"] == ("heterogeneous"
                              if rejects and dec["disjoint_pairs"] else "not_shown")


def _check_controls(rep):
    """Thirteen controls, each of which a previously passing build would have failed.

    Three are new, and all three come from the first code review:

    * `small_effect_world_is_not_called_equivalent` -- the old suite tested a
      -82% effect and a 0% null, so it said nothing about behaviour near the
      pre-registered 10% margin, which is the only place the verdict is
      actually decided.
    * `sub_margin_world_is_equivalent` and `null_world_is_equivalent` -- the old
      control asked only that the null "is not harmful", which an estimator
      that can never conclude anything passes trivially.
    * `audit_endpoint_agrees_where_it_should` -- the old suite only checked the
      audit endpoint where it was supposed to DISagree with the machine, so an
      audit endpoint that always returned zero passed every control.  It now
      also has to reproduce the large effect in the effect world.
    """
    def cond(name):
        return rep[name]["conditional"]

    def verdict(name):
        return cond(name)["equivalence"]["verdict"]

    eff, small, sub = cond("effect_world"), cond("small_effect_world"), cond("sub_margin_world")
    nul = cond("null_world")
    mrg = rep["merging_world"]
    het = cond("heterogeneous_world")
    gap = mrg.get("machine_minus_audit", {}).get("mean", float("nan"))
    eff_gap = rep["effect_world"].get("machine_minus_audit", {}).get("mean", float("nan"))
    checks = {
        "effect_world_is_harmful":
            verdict("effect_world") == "harmful",
        "small_effect_world_is_not_called_equivalent":
            verdict("small_effect_world") != "equivalent",
        "sub_margin_world_is_equivalent":
            verdict("sub_margin_world") == "equivalent",
        "null_world_is_equivalent":
            verdict("null_world") == "equivalent",
        "audit_endpoint_agrees_where_it_should":
            abs(eff_gap) < 0.08 and rep["effect_world"]["audit"]["mean_delta"] < -0.5,
        "label_merging_fools_the_machine_endpoint":
            mrg["raw"]["mean_delta"] < -0.10,
        "label_merging_does_not_fool_the_audit_endpoint":
            abs(mrg["audit"]["mean_delta"]) < 0.06,
        "audit_endpoint_separates_the_two":
            gap < -0.08,
        "effect_world_heterogeneity_is_estimable":
            not eff["heterogeneity"]["non_estimable"],
        # Round-2 P2: `heterogeneity_verdict` existed, was unit-tested, and was
        # never called by this runner -- replacing every heterogeneity p-value
        # with 1.0 passed all nine controls.  The decision rule is now on the
        # executed path and checked against its own inputs here.
        "heterogeneity_decision_is_executed_and_consistent":
            all(_het_decision_consistent(cond(name)) for name in WORLDS),
        "design_is_complete_in_every_world":
            all(cond(name)["design"]["complete_design"] for name in WORLDS),
        # A4 has to be able to FIRE somewhere, or "the runner executes the
        # decision rule" is a claim about a branch that is never taken.
        "heterogeneity_is_detected_where_it_was_planted":
            het["heterogeneity_decision"]["verdict"] == "heterogeneous",
        # ...and it has to depend on the Welch p-value it is given.  This is the
        # second code review's mutation, run in-process against the world that
        # can feel it: with the p-value set to 1.0 the same rule must come back
        # `not_shown`.
        "heterogeneity_decision_depends_on_the_welch_p_value":
            heterogeneity_verdict({**het["heterogeneity"], "p_value": 1.0},
                                  het["country_intervals"])["verdict"]
            == "not_shown",
    }
    summary = {}
    for name in WORLDS:
        c = cond(name)
        summary[name] = {
            "mean_delta": c["mean_delta"],
            "proportional": c["mean_proportional_change"],
            "verdict": c["equivalence"]["verdict"],
            "ci": c["equivalence"].get("ci"),
            "heterogeneity_p": c["heterogeneity"].get("p_value"),
            "heterogeneity_underpowered": c["heterogeneity"].get("underpowered"),
            "heterogeneity_decision": c["heterogeneity_decision"]["verdict"],
            "complete_design": c["design"]["complete_design"],
            "equivalence_df": c["equivalence"].get("df"),
            "machine_minus_audit": rep[name].get("machine_minus_audit", {}).get("mean"),
            "machine_raw_delta": rep[name]["raw"]["mean_delta"],
            "audit_delta": rep[name].get("audit", {}).get("mean_delta"),
        }
    return {"worlds": summary, "checks": checks,
            "passed": all(bool(v) for v in checks.values())}


def _print_controls(ok):
    print("\n=== smoke controls (mock backends, CPU) ===")
    print("  %-19s %9s %9s %14s %9s" %
          ("world", "delta", "change", "verdict", "audit gap"))
    for name, w in ok["worlds"].items():
        gap = w["machine_minus_audit"]
        print("  %-19s %+9.4f %+8.1f%% %14s %9s" %
              (name, w["mean_delta"], 100 * w["proportional"], w["verdict"],
               "n/a" if gap is None else "%+.4f" % gap))
    print()
    for kk, v in ok["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {kk}")
    print("=== " + ("ALL CONTROLS PASSED" if ok["passed"] else "CONTROLS FAILED") + " ===\n")
    if not ok["passed"]:
        print("Do not spend GPU time until these pass: the analysis path cannot\n"
              "recover an effect of known size or reject a planted artefact, so\n"
              "it cannot be trusted on real data.")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def preflight(config_path, annotations_path=None):
    """Everything the real run can check WITHOUT a GPU, checked first.

    The code review's finding: `--config` never opened the file it was handed.
    It raised the GPU blocker immediately, so a typo in the config, an
    impossible (k, m) pair, an annotation file that did not match the pools, or
    a design too small for its own analysis would all have been discovered on
    the rented machine after the weights had downloaded.
    """
    cfg = load_config(config_path)
    prompts = [p for p in build_underspecified_prompts(
        countries=cfg["countries"], domains=[cfg["domain"]])
        if p.template in cfg["templates"] and p.country in cfg["countries"]]
    expected = len(cfg["countries"]) * len(cfg["templates"])
    if len(prompts) != expected:
        raise ConfigError(
            f"{config_path}: built {len(prompts)} prompt strings, expected "
            f"{expected}; check `templates` against src/data.py")

    report = {"config": config_path, "budget": cfg.budget(),
              "warnings": list(cfg["warnings"]),
              "prompts": [p.prompt_id for p in prompts]}

    # The generator id is part of the image-ID join key and is computable
    # without a GPU, so preflight reports it: an annotation sheet cut against a
    # different checkpoint, sampler or step count carries different IDs and
    # will match nothing, and that has to be visible before the run, not after.
    gen_ids = {}
    from src.store import generator_config
    for g in cfg["generators"]:
        gen_ids[g["id"]] = generator_fingerprint(g["id"], generator_config(g, cfg))
    report["generator_ids"] = gen_ids

    if annotations_path:
        ann = load_annotations(annotations_path)
        report["annotations"] = {"path": annotations_path, "n_rows": len(ann)}
        covered = {}
        for name, gid in gen_ids.items():
            c = 0
            for i, pr in enumerate(prompts):
                for j in range(cfg["n_pools"]):
                    _, cov = audit_labels_for_pool(
                        gid, pr.prompt_id, cfg["seed"] + 1000 * i + j,
                        cfg["n"], ann)
                    c += bool(cov["complete"])
            covered[name] = c
        report["annotations"]["fully_annotated_pools"] = covered
        report["annotations"]["fully_annotated_pools_total"] = sum(covered.values())
        if not any(covered.values()):
            report["warnings"].append(
                "no pool is fully annotated for any configured generator; the "
                "audit endpoint will not run for any pool, and the headline "
                "claim is gated on it")
        for name, c in covered.items():
            if c == 0 and any(covered.values()):
                report["warnings"].append(
                    f"no pool of {name} is fully annotated (generator_id "
                    f"{gen_ids[name]}); its audit endpoint will not run")

    missing = []
    for label, fn, arg in (("generator", load_generator, cfg["generators"][0]["id"]),
                           ("scorer", load_scorer, cfg["scorers"]["primary"]),
                           ("geotagger", load_geotagger, cfg["geotagger"]["id"])):
        try:
            fn(arg)
        except Exception as exc:                     # noqa: BLE001 - reporting
            missing.append(f"{label} ({arg}): {type(exc).__name__}: "
                           f"{str(exc).splitlines()[0][:120]}")
    report["unavailable_backends"] = missing
    report["ready_to_run"] = not missing
    return report


def real(config_path, annotations_path=None, out_dir="results/real"):
    report = preflight(config_path, annotations_path)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "preflight.json"), "w") as f:
        json.dump(_jsonable(report), f, indent=2)

    b = report["budget"]
    print("\n=== preflight (CPU only, no weights loaded) ===")
    print("  config            %s" % report["config"])
    print("  prompt strings    %d" % b["prompt_strings"])
    print("  images            %d per model x %d models = %d"
          % (b["images_per_model"], b["models"], b["images_total"]))
    print("  pools per country %d" % b["pools_per_country"])
    print("  known-label       %d images" % b["known_label_images"])
    print("  tuning            %d images" % b["tuning_images"])
    print("  human-annotated   0 images (no human-subject annotation)")
    for w in report["warnings"]:
        print("  WARNING: %s" % w)
    if report["ready_to_run"]:
        raise SystemExit(
            "Preflight passed and all three backends constructed.  Executing "
            "the real run is not implemented in this repository: no GPU was "
            "reachable from the session that wrote it, so the generate/score/"
            "tag loop has never been executed once and shipping it untested "
            "would be worse than shipping the blocker.  The analysis path it "
            "would feed is fully tested -- see `python run.py --smoke`.")
    raise SystemExit(
        "Preflight ran; the real run cannot.  Unavailable backends:\n  - "
        + "\n  - ".join(report["unavailable_backends"])
        + "\n\nSee ../.aris/compute/REACHABILITY.md: no CUDA device and no "
          "network route to the model weights was reachable from the session\n"
          "that prepared this repository, so every number in the proposal is a "
          "plan, not a measurement.\nImplement load_scorer/load_geotagger in "
          "src/backends.py on a machine that has both, then re-run.\n"
          "`python run.py --smoke` must pass first.\n"
          f"Preflight report written to {os.path.join(out_dir, 'preflight.json')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=96, help="candidate pool size")
    ap.add_argument("--k", type=int, default=16, help="primary selection size")
    ap.add_argument("--m", type=int, default=8, help="common rarefaction count")
    ap.add_argument("--pools", type=int, default=4, help="independent pools per prompt")
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/smoke")
    ap.add_argument("--config", default=None)
    ap.add_argument("--annotations", default=None,
                    help="adjudicated human-audit CSV (see src/annotations.py)")
    ap.add_argument("--preflight", action="store_true",
                    help="validate a config and report the budget; no GPU")
    a = ap.parse_args(argv)
    if a.smoke:
        return smoke(n=a.n, k=a.k, m_auth=a.m, n_pools=a.pools, out_dir=a.out,
                     seed=a.seed, n_null=a.n_null)
    if a.preflight and a.config:
        print(json.dumps(_jsonable(preflight(a.config, a.annotations)), indent=2))
        return 0
    if a.config:
        return real(a.config, a.annotations, out_dir=a.out
                    if a.out != "results/smoke" else "results/real")
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
