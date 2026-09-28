"""analyze.py end to end on a synthetic store (no GPU, no model)."""
import json

import numpy as np

import analyze
from src.config import load_config
import fake_store


def test_analyze_replays_a_store_end_to_end(tmp_path):
    cfgp = fake_store.small_config(str(tmp_path))
    cfg = load_config(cfgp)
    root = tmp_path / "store"
    tag = fake_store.build(str(root), cfg)
    out = tmp_path / "out"
    assert analyze.main(["--root", str(root), "--config", cfgp, "--out", str(out),
                         "--frontier-n-null", "100"]) == 0
    res = json.loads((out / f"{tag}.json").read_text())
    a1 = res["a1__pickscore"]
    # The scorer prefers four head dishes, so top-k must lose diversity.
    assert a1["raw"]["mean_delta"] < 0
    r1 = res["r1__pickscore"]
    assert r1["verified"]["audit_sources"] == ["known_by_construction"]
    assert r1["verified"]["gap"]["n"] > 0
    assert r1["verifier_diagnostics"]["auc_true_vs_decoy"] > 0.5
    acc = r1["tagger_accuracy_by_score_quartile"]["unfiltered"]
    assert 0.8 < np.nanmean(acc["accuracy_by_bin"]) <= 1.0
    rep = res["rep__pickscore"]
    assert rep["mean_repetition_excess"] > 0
    a3 = res["a3__pickscore"]
    assert a3["judge"] == "imagereward"
    assert set(a3["chosen_on_tuning"]) >= {"stk", "mmr", "dpp"}
    # Diversity-aware selectors must recover some of the top-k loss.
    stk = a3["chosen_on_tuning"]["stk"]["evaluation"]["delta"]["mean"]
    assert stk > a3["curves"]["topk|None"]["delta"]["mean"]
    vs = a3["chosen_on_tuning"]["stk"]["vs_topk"]
    assert abs(vs["delta"]["mean"] - (stk - a3["curves"]["topk|None"]["delta"]["mean"])) < 0.05
    assert np.isfinite(vs["judge_gain_retained"])
    # post hoc: the saved per-grid-point paired contrast agrees with the
    # frozen configuration's own paired contrast
    cv = a3["curves_vs_topk"][f"stk|{a3['chosen_on_tuning']['stk']['param']}"]
    assert abs(cv["delta"]["mean"] - vs["delta"]["mean"]) < 1e-12
    assert abs(cv["emb_delta"]["mean"] - vs["emb_delta"]["mean"]) < 1e-12
    assert "topk|None" not in a3["curves_vs_topk"]
    # amendment 6 (descriptive): the retention interval brackets the point
    # estimate, the within-dish faithfulness contrast and the cardinality-
    # matched repetition excess are reported
    lo, hi = vs["judge_gain_retained_ci"]
    assert lo <= vs["judge_gain_retained"] <= hi
    wd = res["r1__pickscore"]["verifier_keep_by_score_quartile_within_dish"]
    assert wd is None or wd["n_images"] > 0
    wp = res["r1__pickscore"]["verifier_keep_by_score_quartile_within_dish_subset_pool_rank"]
    assert (wd is None) == (wp is None)
    assert wd is None or wp["n_images"] == wd["n_images"]
    assert np.isfinite(res["rep__pickscore"]["mean_repetition_excess_matched"])
    # experiment audit: raw judge gains beside the standardised ones
    assert vs["topk_raw_judge_gain"]["mean"] > 0
    assert np.isfinite(vs["raw_gain_judge"]["mean"])
    assert res["a1__pickscore"]["dose_response_summary"] == {}  # secondary_k=None here
    assert "reader" not in res["provenance"]
    cl = res["claims__pickscore"]
    assert cl["A1"]["p_holm"] >= cl["A1"]["p_loss_beyond_margin"]
    assert set(cl) >= {"A1", "A2", "A3"}
    # The report reads the same output without recomputing anything.
    import make_report
    rep = tmp_path / "report"
    assert make_report.main(["--in", str(out), "--out", str(rep)]) == 0
    tables = (rep / "tables.md").read_text()
    assert f"{100 * np.expm1(a1['raw']['mean_delta']):+.1f}%" in tables
    # round-6 review, P2: the deciding (verified) gap and both gap intervals
    # are displayed, beside A1 and in the R1 table
    r1 = res["r1__pickscore"]
    ident = lambda x: x
    gv, gu = r1["verified"]["gap"], r1["unfiltered"]["gap"]
    a1_row = [l for l in tables.splitlines() if "| pickscore | raw |" in l][0]
    assert (f"{gv['verdict']} {make_report.fmt(gv.get('mean'), 3)} "
            f"{make_report.fmt_ci(gv.get('ci'), f=ident, nd=3)} (verified)") in a1_row
    r1_row = [l for l in tables.splitlines()
              if l.startswith(f"| {res['generator']} | pickscore | {gv['verdict']}")][0]
    assert make_report.fmt_ci(gv.get("ci"), f=ident, nd=3) in r1_row
    assert (f"{gu['verdict']} ({make_report.fmt(gu.get('mean'), 3)}) | "
            f"{make_report.fmt_ci(gu.get('ci'), f=ident, nd=3)}") in r1_row
    for fig in ("fig_country_delta", "fig_frontier", "fig_r1_quartile"):
        assert (rep / f"{fig}.pdf").stat().st_size > 0


def test_holm_is_step_down_and_monotone():
    assert analyze.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_verifier_options_are_deterministic_and_contain_the_truth_once():
    import run_gpu
    row = {"image_id": "g#p#1#0003", "dish": "Moqueca"}
    pool = {"Moqueca", "Feijoada", "Acarajé", "Coxinha", "Pão de queijo"}
    o1, pos, decoy = run_gpu.verifier_options(row, pool, [])
    o2, pos2, _ = run_gpu.verifier_options(row, pool, [])
    assert o1 == o2 and pos == pos2
    assert o1[pos] == "Moqueca" and o1.count("Moqueca") == 1
    assert o1[-1] == "none of these" and len(o1) == 5 and decoy != "Moqueca"
    # A pool with too few distinct dishes is topped up from the vocabulary.
    o3, _, _ = run_gpu.verifier_options(row, {"Moqueca"}, ["Moqueca", "A", "B", "C"])
    assert sorted(o3[:4]) == ["A", "B", "C", "Moqueca"]


# ---- round-3 code review: decision rules -----------------------------------

def _res(a1_verdict="harmful", mean=-0.3, se=0.02, df=20.0, gap="equivalent",
         welch_p=0.001, a2_claim=True, a3_ci=(0.05, 0.2), a3_p=0.001,
         tk_lo=0.3, retained=0.95):
    eq = {"verdict": a1_verdict, "mean_delta": mean, "se": se, "df": df}
    a2 = {"welch_p": welch_p, "claim": a2_claim,
          "status": "heterogeneous_spread_established" if a2_claim
          else "heterogeneity_not_established"}
    return {"generator": "g",
            "a1__pickscore": {"raw": {"equivalence": eq, "a2": a2}},
            "r1__pickscore": {"verified": {"gap": {"verdict": gap}}},
            "a3__pickscore": {"chosen_on_tuning": {"stk": {
                "param": 4, "vs_topk": {
                    "delta": {"mean": 0.1, "ci": list(a3_ci), "p_le_0": a3_p},
                    "topk_judge_gain": {"mean": 0.5, "ci_lo": tk_lo},
                    "judge_gain_retained": retained}}}}}


def test_a1_claim_requires_an_equivalent_known_label_gap():
    assert analyze.claims(_res())["A1"]["claim"]
    c = analyze.claims(_res(gap="inconclusive"))["A1"]
    assert not c["claim"] and c["status"] == "unvalidated_at_5pct"
    c = analyze.claims(_res(gap="harmful"))["A1"]
    assert not c["claim"] and c["status"] == "stop_rule_4"
    r = _res()
    del r["r1__pickscore"]
    c = analyze.claims(r)["A1"]
    assert not c["claim"] and any("missing" in w for w in c["withheld_because"])
    # amendment 4 (round-5 review, P0): the unfiltered gap is sensitivity
    # only; it never clears A1 when the verified gap is non-estimable
    r = _res(gap="non_estimable")
    assert not analyze.claims(r)["A1"]["claim"]
    r["r1__pickscore"]["unfiltered"] = {"gap": {"verdict": "equivalent"}}
    c = analyze.claims(r)["A1"]
    assert not c["claim"] and c["known_label_gap_basis"] == "verified"
    assert c["known_label_gap_unfiltered_sensitivity"]["verdict"] == "equivalent"
    r["r1__pickscore"]["unfiltered"] = {"gap": {"verdict": "harmful"}}
    assert analyze.claims(r)["A1"]["status"] == "not_claimed"
    assert analyze.claims(_res())["A1"]["known_label_gap_basis"] == "verified"


def test_a3_claim_requires_ci_above_zero_and_a_positive_topk_gain():
    assert analyze.claims(_res())["A3"]["claim"]
    assert not analyze.claims(_res(a3_ci=(-0.01, 0.2)))["A3"]["claim"]
    # A worse selector with a negative top-k gain must not pass the 90% rule.
    assert not analyze.claims(_res(tk_lo=-0.1, retained=1.5))["A3"]["claim"]
    assert not analyze.claims(_res(retained=0.85))["A3"]["claim"]


def test_holm_keeps_the_fixed_family_when_a_p_value_is_missing():
    assert analyze.holm([0.02, float("nan"), 0.02]) == [0.06, 1.0, 0.06]
    c = analyze.claims(_res(welch_p=None, a3_p=float("nan")))
    assert c["A1"]["p_holm"] == 3 * c["A1"]["p_loss_beyond_margin"]
    r = _res()
    r.pop("a3__pickscore")
    c = analyze.claims(r)
    assert c["A3"]["claim"] is False and c["A3"]["p_holm"] == 1.0


def test_a2_needs_the_complete_design_and_reports_a_spread_lower_bound():
    ep = {"equivalence": {"country_means": {"a": 0.0, "b": -0.3}},
          "heterogeneity_decision": {"verdict": "heterogeneous"},
          "heterogeneity": {"p_value": 0.001},
          "design": {"complete_design": True},
          "country_intervals": {"intervals": {"a": (-0.05, 0.05), "b": (-0.35, -0.25)}}}
    r = analyze.a2_rule(ep)
    assert r["claim"] and abs(r["spread_lcb"] - 0.2) < 1e-9
    assert r["status"] == "heterogeneous_spread_established"
    ep["country_intervals"]["intervals"]["b"] = (-0.5, -0.0)
    assert analyze.a2_rule(ep)["status"] == "heterogeneous_spread_size_not_established"
    ep["design"]["complete_design"] = False
    r = analyze.a2_rule(ep)
    assert not r["claim"] and r["status"] == "non_estimable_incomplete_design"


def test_known_label_gap_is_non_estimable_when_a_country_is_thin():
    from src.pipeline import _gap_test
    rows = ([{"country": "A", "machine_minus_audit": x} for x in (0.01, -0.01, 0.0)]
            + [{"country": "B", "machine_minus_audit": 0.0}])
    g = _gap_test(rows, "machine_minus_audit", ["A", "B"])
    assert g["verdict"] == "non_estimable"
    g = _gap_test(rows[:3], "machine_minus_audit", ["A", "B"])
    assert g["verdict"] == "non_estimable"


def test_tuning_ignores_configurations_missing_a_country():
    from src.frontier import choose_on_tuning
    nan = float("nan")
    rows = [{("topk", None): {"delta": -0.3, "gain_judge": 1.0},
             ("stk", 2): {"delta": -0.05, "gain_judge": 0.5},
             ("stk", 4): {"delta": 0.0 if c == "A" else nan, "gain_judge": 0.9},
             "_country": c} for c in ("A", "B")]
    ch = choose_on_tuning(rows)
    assert ch["stk"]["param"] == 2 and ch["stk"]["excluded_incomplete"] == {"4": ["B"]}


def test_quartile_interval_resamples_pools():
    from src.known_label import tagger_accuracy_by_score_quantile
    rng = np.random.default_rng(0)
    # Four pools; accuracy differs by pool, not by score.
    correct, ranks, groups = [], [], []
    for g, acc in enumerate((0.2, 0.9, 0.3, 0.95)):
        for i in range(64):
            correct.append(rng.random() < acc)
            ranks.append(i / 64.0)
            groups.append(g)
    img = tagger_accuracy_by_score_quantile(correct, ranks)
    pool = tagger_accuracy_by_score_quantile(correct, ranks, groups=groups)
    assert pool["interval"] == "pool_cluster_bootstrap" and pool["n_clusters"] == 4
    assert img["top_minus_bottom"] == pool["top_minus_bottom"]


def test_native_script_dish_names_survive_normalisation():
    from src.labels import VocabMatcher, normalize, parse_tagger_json
    assert normalize("そば米汁") == "そば米汁"
    assert normalize("そば") != normalize("そは")
    assert normalize("दाल") == "दाल"
    assert normalize("Feijão") == "feijao"
    assert parse_tagger_json('{"dish": "そば米汁", "country": "Japan"}')[0] == "そば米汁"
    assert VocabMatcher(["そば米汁", "Sushi"]).match("そば米汁") == ("そば米汁", True)


def test_tagger_diagnostics_count_native_script_answers(tmp_path):
    from src.store import CachedTagger
    (tmp_path / "tags").mkdir()
    rows = [{"image_id": "a", "reply": '{"dish": "そば米汁", "country": "Japan"}'},
            {"image_id": "b", "reply": '{"dish": "Sushi", "country": "Japan"}'},
            {"image_id": "c", "reply": '{"dish": "unknown", "country": "unknown"}'},
            {"image_id": "d", "reply": "", "dish": "Feijão", "country": "Brazil"}]
    (tmp_path / "tags" / "t.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    d = analyze.tagger_diagnostics(CachedTagger(str(tmp_path), "t", ["Sushi", "そば米汁"]))
    assert d["n"] == 4
    assert d["native_script_rate"] == 0.25   # Feijão is Latin with a diacritic
    assert d["resolved_rate"] == 0.75        # "unknown" is the only unresolved answer
    assert d["in_vocab_rate"] == 0.5


def test_a_partial_rerun_cannot_mix_provenance(tmp_path):
    import pytest
    cfgp = fake_store.small_config(str(tmp_path))
    cfg = load_config(cfgp)
    root = tmp_path / "store"
    tag = fake_store.build(str(root), cfg)
    out = tmp_path / "out"
    args = ["--root", str(root), "--config", cfgp, "--out", str(out), "--parts", "a1"]
    assert analyze.main(args + ["--n-null", "50"]) == 0
    res = json.loads((out / f"{tag}.json").read_text())
    assert "claims__pickscore" in res
    with pytest.raises(SystemExit):
        analyze.main(args + ["--n-null", "60"])
    assert analyze.main(args + ["--n-null", "60", "--replace"]) == 0
    # round-5 code review, P2: the frontier's Monte Carlo setting is provenance
    with pytest.raises(SystemExit):
        analyze.main(args + ["--n-null", "60", "--frontier-n-null", "500"])
    res = json.loads((out / f"{tag}.json").read_text())
    assert res["provenance"]["frontier_n_null"] == 2000 and res["provenance"]["code_sha"]


def test_frozen_config_matches_the_selector_grid():
    import copy
    cfg = load_config(analyze.os.path.join(analyze.HERE, "configs", "pilot.yaml"))
    analyze.validate_frontier_config(cfg)
    bad = copy.deepcopy(cfg)
    bad["frontier"]["selectors"] = ["random", "topk", "mmr"]
    import pytest
    with pytest.raises(ValueError):
        analyze.validate_frontier_config(bad)


def test_country_value_weights_templates_equally():
    """Round-4 code review: a country that kept more A than B pools was
    re-weighted toward A.  Its value is now the mean of its template means."""
    from src.analysis import _fixed_country_moments, country_marginal
    by = {"X": {"A": np.array([0.0, 0.1, 0.0, 0.1]), "B": np.array([1.0, 1.1])},
          "Y": {"A": np.array([0.0, 0.2]), "B": np.array([0.0, 0.2])}}
    theta, se, df, st = _fixed_country_moments(by)
    assert np.isclose(st["country_means"]["X"], (0.05 + 1.05) / 2)
    assert np.isclose(theta, ((0.05 + 1.05) / 2 + 0.1) / 2)
    assert st["country_n_pools"]["X"] == 6
    # a plain vector is one cell: the per-country formula is unchanged
    flat = {c: np.concatenate(list(v.values())) for c, v in by.items()}
    t2, _, _, s2 = _fixed_country_moments(flat)
    assert np.isclose(s2["country_means"]["X"], np.mean([0, .1, 0, .1, 1, 1.1]))
    m = country_marginal(by, n_boot=50, rng=np.random.default_rng(0))
    assert np.isclose(m["mean_delta"], theta)
    # a single-pool cell carries no variance estimate
    thin = {"X": {"A": np.array([0.0, 0.1]), "B": np.array([1.0])}, "Y": by["Y"]}
    assert _fixed_country_moments(thin)[0] is None


def test_known_label_gap_is_exactly_zero_when_the_tagger_is_right():
    """Round-4 code review: both sides of the gap drew their own random
    reference, so identical labels still gave a nonzero gap."""
    from src.pipeline import _endpoint
    import zlib
    rng = np.random.default_rng(3)
    labels = [f"d{int(x)}" for x in rng.zipf(1.3, 64) % 20]
    scores = rng.normal(size=64)
    keep = list(rng.random(64) < 0.8)
    crn = 12345
    a = _endpoint(labels, scores, 16, 8, 200, np.random.default_rng(crn), keep, "j")
    b = _endpoint(list(labels), scores, 16, 8, 200, np.random.default_rng(crn), keep, "j")
    assert a["delta"] - b["delta"] == 0.0


def test_fuzzy_matching_does_not_merge_distinct_dishes():
    from src.labels import VocabMatcher
    M = VocabMatcher(["Alu diye latha machher jhol", "Alu diye bhola machher jhol",
                      "Alu diye ban machher jhol", "Feijoada", "Jollof rice"])
    assert M.match("alu diye katla machher jhol") == ("alu diye katla machher jhol", False)
    assert M.match("alu diye bhol machher jhol")[1] is False     # crowded family
    assert M.match("feijoda") == ("feijoada", True)                # a typo still maps
    assert M.match("jolof rice") == ("jollof rice", True)


def test_round5_matcher_safe_spelling_and_presentation():
    """Round-5 code review, P1: one-letter ingredient swaps are not typos,
    and serving phrases do not make a new label."""
    import json
    import os
    from src.labels import VocabMatcher, strip_presentation
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab = json.load(open(os.path.join(here, "data", "cspace_cuisine_vocab.json")))["vocab"]
    M = VocabMatcher(vocab["United States"])
    assert M.match("beet and broccoli") == ("beet and broccoli", False)
    assert M.match("a bowl of beef and broccoli") == ("beef and broccoli", True)
    assert M.match("A plate of traditional beef and broccoli")[1] is True
    assert M.match("plate lunch") == ("plate lunch", True)      # CSpace name kept whole
    # an unlisted answer collapses with and without its serving phrase
    assert M.match("a bowl of zzz stew") == M.match("zzz stew") == ("zzz stew", False)
    assert strip_presentation("bowl") == "bowl"
    R = VocabMatcher({"Japan": ["Ramen", "Rice"], "France": ["French onion soup"]})
    assert R.match("japanese ramen") == ("ramen", True)
    assert R.match("french ramen") == ("french ramen", False)   # not France's
    assert R.match("japanese curry") == ("japanese curry", False)
    assert R.match("ride") == ("ride", False)                   # 4 letters: never corrected


def test_round5_holm_treats_unavailable_members_as_p_one():
    """Round-5 code review, P0: a non-estimable A1 that still carries its
    observed-only moments must enter Holm at p = 1, not at its t test."""
    r = _res(a3_p=0.02)
    base = analyze.claims(r)
    assert base["A3"]["p_holm"] < 0.05          # A1 small p shrinks the multiplier
    r["a1__pickscore"]["raw"]["equivalence"]["verdict"] = "non_estimable"
    c = analyze.claims(r)
    assert c["A1"]["p_holm"] == 1.0
    assert abs(c["A3"]["p_holm"] - 0.04) < 1e-12 and c["A3"]["claim"]
    r["a1__pickscore"]["raw"]["a2"]["status"] = "non_estimable_incomplete_design"
    c = analyze.claims(r)
    assert abs(c["A3"]["p_holm"] - 0.06) < 1e-12 and not c["A3"]["claim"]
    r = _res()
    r["a3__pickscore"]["chosen_on_tuning"]["stk"]["vs_topk"]["delta"].update(
        non_estimable=True, reason="x")
    assert analyze.claims(r)["A3"]["p_holm"] == 1.0


def test_round6_exact_known_label_agreement_is_descriptive_only():
    """Round-6 code review, P1: all-zero pool gaps do not bound the
    population gap, so they are flagged but stay non-estimable."""
    from src.pipeline import _gap_test
    rows = [{"country": c, "g": 0.0} for c in ("A", "B") for _ in range(2)]
    g = _gap_test(rows, "g", ["A", "B"])
    assert g["verdict"] == "non_estimable" and g["exact_agreement"]
    rows = [{"country": c, "g": 0.01} for c in ("A", "B") for _ in range(2)]
    g = _gap_test(rows, "g", ["A", "B"])
    assert g["verdict"] == "non_estimable" and not g["exact_agreement"]


def test_round6_unavailable_a2_decision_enters_holm_at_one():
    """Round-6 code review, P1: Welch estimable but the simultaneous
    intervals not -> A2 is unavailable, p = 1 in Holm."""
    ep = {"equivalence": {"country_means": {"A": 0.0, "B": 0.3}},
          "heterogeneity_decision": {"verdict": "non_estimable"},
          "heterogeneity": {"p_value": 1.4e-7},
          "design": {"complete_design": True},
          "country_intervals": {"non_estimable": True}}
    a2 = analyze.a2_rule(ep)
    assert a2["status"] == "non_estimable_heterogeneity_decision" and not a2["claim"]
    r = _res(a3_p=0.02)
    r["a1__pickscore"]["raw"]["a2"] = a2
    c = analyze.claims(r)
    assert c["A2"]["p_holm"] == 1.0
    assert abs(c["A3"]["p_holm"] - 0.04) < 1e-12   # 2 x .02: A1's p is ~0


def test_round5_a2_uses_template_balanced_country_values():
    """Round-5 code review, P1: unequal A and B pool counts must not re-weight
    the templates in Welch or the simultaneous intervals."""
    import numpy as np
    from src.analysis import heterogeneity_test, simultaneous_country_intervals
    rng = np.random.default_rng(1)
    by = {}
    for c in "ABCDEFGH":
        by[c] = {"A": list(-0.2 + 0.01 * rng.normal(size=4)),
                 "B": list(0.2 + 0.01 * rng.normal(size=3 if c == "A" else 6))}
    het = heterogeneity_test(by, n_perm=50, rng=rng)
    iv = simultaneous_country_intervals(by, n_boot=300, rng=rng)
    # round-6 review, P2: assert the balanced value itself, not a tolerance
    # that the pooled mean also meets
    bal = {c: (np.mean(v["A"]) + np.mean(v["B"])) / 2 for c, v in by.items()}
    pooled_a = np.mean(by["A"]["A"] + by["A"]["B"])
    assert abs(bal["A"] - pooled_a) > 0.02
    for c in by:
        assert abs(het["country_means"][c] - bal[c]) < 1e-12
        assert abs(iv["point"][c] - bal[c]) < 1e-12
    assert het["p_value"] > 0.01
    # a plain vector is one cell and reproduces the classic Welch
    flat = {c: [float(x) for x in rng.normal(size=5) + i] for i, c in enumerate("ABC")}
    one = heterogeneity_test(flat, n_perm=10, rng=np.random.default_rng(0))
    wrapped = heterogeneity_test({c: {"T": v} for c, v in flat.items()}, n_perm=10,
                                 rng=np.random.default_rng(0))
    assert abs(one["statistic"] - wrapped["statistic"]) < 1e-12


def test_round6_matcher_uses_normalised_labels_and_checks_each_strip_step():
    """Round-6 code review, P1/P2: "soba" is not Brazil's Sobá, "japanese
    curry" is not India's curry, and "a bowl of fresh tomme" finds its entry."""
    import json
    import os
    from src.labels import VocabMatcher
    from src.pipeline import ESTABLISHED, classify_tags
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab = json.load(open(os.path.join(here, "data", "cspace_cuisine_vocab.json")))["vocab"]
    M = VocabMatcher(vocab)
    lab, inv = M.match("soba")
    assert inv and lab == "soba"
    st, _ = classify_tags([("C", "Japan", lab)], "Japan", vocab["Japan"])
    assert st == [ESTABLISHED]
    assert M.match("Pâté")[0] == M.match("pâté (pâtisserie)")[0] == "pate"
    assert M.match("Japanese curry") == ("japanese curry", False)
    assert M.match("japanese ramen") == ("ramen", True)
    fresh = [v for v in vocab["France"] if v.lower().startswith("fresh ")]
    if fresh:
        k = M.match(fresh[0])[0]
        assert M.match("a bowl of " + fresh[0]) == (k, True)
    # without a by-country vocabulary the demonym rule is off
    assert VocabMatcher(["Ramen"]).match("japanese ramen") == ("japanese ramen", False)


def test_round6_a2_enters_holm_with_the_conservative_bootstrap_p():
    """Round-6 code review, P2: Welch is liberal under skew; A2's Holm p is
    max(Welch, conservative bootstrap)."""
    ep = {"equivalence": {"country_means": {"A": 0.0, "B": 0.3}},
          "heterogeneity_decision": {"verdict": "heterogeneous"},
          "heterogeneity": {"p_value": 0.001, "p_value_bootstrap_conservative": 0.03},
          "design": {"complete_design": True},
          "country_intervals": {"intervals": {"A": [-0.1, 0.05], "B": [0.2, 0.4]}}}
    a2 = analyze.a2_rule(ep)
    assert a2["holm_p"] == 0.03 and a2["welch_p"] == 0.001
    r = _res(a3_p=0.9)
    r["a1__pickscore"]["raw"]["a2"] = a2
    c = analyze.claims(r)
    assert abs(c["A2"]["p_holm"] - 0.06) < 1e-12        # 2 x .03, not 2 x .001


def test_round7_edge_cases():
    """Round-7 code review, P2: A2 without a bootstrap p is unavailable;
    raw tags get one normalised label; exact agreement needs every country."""
    from src.pipeline import ESTABLISHED, _gap_test, classify_tags
    ep = {"equivalence": {"country_means": {"A": 0.0, "B": 0.3}},
          "heterogeneity_decision": {"verdict": "heterogeneous"},
          "heterogeneity": {"p_value": 0.001},
          "design": {"complete_design": True},
          "country_intervals": {"intervals": {"A": [-0.1, 0.05], "B": [0.2, 0.4]}}}
    a2 = analyze.a2_rule(ep)
    assert a2["holm_p"] is None
    r = _res(a3_p=0.9)
    r["a1__pickscore"]["raw"]["a2"] = a2
    assert analyze.claims(r)["A2"]["p_holm"] == 1.0
    st, lab = classify_tags([("C", "France", "Praline"), ("C", "France", "praline")],
                            "France", ["Praline", "praline"])
    assert st == [ESTABLISHED] * 2 and lab[0] == lab[1]
    g = _gap_test([{"country": "A", "g": 0.0}] * 2, "g", ["A", "B"])
    assert g["verdict"] == "non_estimable" and not g["exact_agreement"]
    from src.pipeline import _normalised_keys
    assert _normalised_keys({"Praline": 1.0, "praline": 3.0, "Pâté": 2.0}) == \
        {"praline": 2.0, "pate": 2.0}


def test_amendment6_reader_files_are_separate(tmp_path):
    """A sensitivity reader reads and writes `<gen>__<reader>.jsonl`; the
    pre-registered Qwen reader keeps the plain name, and its provenance has no
    reader key, so existing outputs still merge."""
    import json as _json
    from src.store import CachedTagger, append_jsonl, reader_file
    root = str(tmp_path)
    assert reader_file(root, "tags", "g").endswith("tags/g.jsonl")
    assert reader_file(root, "tags", "g", "qwen").endswith("tags/g.jsonl")
    assert reader_file(root, "verify", "g", "pixtral").endswith("verify/g__pixtral.jsonl")
    vocab = {"Brazil": ["Feijoada"], "Japan": ["Sushi"]}
    append_jsonl(reader_file(root, "tags", "g"),
                 [{"image_id": "i", "reply": _json.dumps({"dish": "sushi", "country": "Japan"})}])
    append_jsonl(reader_file(root, "tags", "g", "siglip"),
                 [{"image_id": "i", "reply": _json.dumps({"dish": "Feijoada", "country": "Brazil"})}])
    assert CachedTagger(root, "g", vocab).tag_one("i")[2] == "sushi"
    assert CachedTagger(root, "g", vocab, reader="siglip").tag_one("i")[2] == "feijoada"


def test_dose_summary_reports_each_secondary_k():
    """Experiment audit: the k = 32 curve was computed and then dropped."""
    import analyze
    rows = [{"dose_response_matched": [
        {"k": 16, "non_estimable": False, "log_contrast": -0.10 + 0.01 * i},
        {"k": 32, "non_estimable": False, "log_contrast": -0.05 + 0.01 * i},
        {"k": 48, "non_estimable": True}]} for i in range(6)]
    out = analyze.dose_summary(rows)
    assert out["32"]["n_estimable"] == 6 and out["48"]["n_estimable"] == 0
    assert "mean_log_contrast" not in out["48"]
    lo, hi = out["32"]["ci"]
    assert lo <= out["32"]["mean_log_contrast"] <= hi < 0


def test_dose_summary_compares_k_on_one_estimand():
    """Experiment audit run 02: k = 32 was log(sel / arithmetic null mean) while
    k = 16 was log sel - mean log null.  Both now come from one sweep, and the
    secondary k is reported as a paired difference from the primary k."""
    import analyze
    import numpy as np
    from src.analysis import dose_response
    labels = [f"d{i % 12}" for i in range(64)]
    scores = np.array([-(i % 12) + 0.001 * i for i in range(64)])  # favour low dish ids
    rows = dose_response(labels, scores, [16, 32],
                         lambda kk, r: r.choice(64, kk, replace=False),
                         n_null=50, rng=np.random.default_rng(0), eval_size=8)
    for d in rows:
        assert np.isfinite(d["log_contrast"])
    out = analyze.dose_summary([{"dose_response_matched": rows}] * 3)
    v = out["32"]["vs_primary_k"]
    assert v["k_ref"] == 16 and v["n_pools"] == 3
    assert abs(v["mean_diff"] - (rows[1]["log_contrast"] - rows[0]["log_contrast"])) < 1e-12
    assert "vs_primary_k" not in out["16"]


def test_reader_robust_needs_all_three_registered_readers():
    """Experiment audit: two readers must not yield "yes"."""
    import sensitivity
    assert sensitivity._robust_cell(False, ["qwen"]) == "no"
    assert sensitivity._robust_cell(True, ["qwen", "siglip"]).startswith("pending")
    assert sensitivity._robust_cell(True, ["qwen", "pixtral", "siglip"]) == "yes"


def test_sensitivity_reader_outputs_carry_no_claims(monkeypatch):
    """Experiment audit run 02: a SigLIP report said "Pre-registered claims"
    and "claimed"; amendment 6 reserves claims for the Qwen reader."""
    import analyze
    import make_report
    fake = {"cell": "c", "A1": {"claim": False}, "A2": {"claim": False},
            "A3": {"claim": True, "family": "stk"}}
    monkeypatch.setattr(analyze, "claims", lambda res, scorer: dict(fake))
    qwen = analyze.rebuild_claims({"a1__pickscore": {}, "provenance": {}})
    assert qwen["claims__pickscore"]["A3"]["claim"] is True
    sig = analyze.rebuild_claims({"a1__pickscore": {}, "scorers": ["pickscore"],
                                  "provenance": {"reader": "siglip"}})
    c = sig["claims__pickscore"]
    assert c["confirmatory"] is False and c["A3"]["claim"] is False
    assert c["A3"]["rule_met_sensitivity_only"] is True
    table = make_report.table_claims({"sdxl": sig})
    assert "claimed" not in table and "sensitivity reader, not a claim" in table


def test_a3_cross_reader_table_uses_the_frozen_configuration():
    """Post hoc: the configuration frozen on the first reader is read from
    every reader's `curves_vs_topk`; a CI touching 0 under any reader, or a
    missing reader, is not reader-robust."""
    import sensitivity

    def d(m, lo, hi):
        return {"non_estimable": False, "mean": m, "ci": [lo, hi]}

    def a3(tag_ci_lo, param=4):
        return {"a3__pickscore": {
            "selector": "pickscore", "judge": "imagereward",
            "chosen_on_tuning": {"stk": {"param": param}},
            "curves_vs_topk": {"stk|4": {"delta": d(0.04, tag_ci_lo, 0.08),
                                         "emb_delta": d(0.07, 0.05, 0.09)},
                               "stk|12": {"delta": d(-0.5, -0.6, -0.4),
                                          "emb_delta": d(0.0, -0.1, 0.1)}}}}
    names = ["qwen", "pixtral", "siglip"]
    # the other readers froze a different parameter; the table must ignore it
    res = {("qwen", "sdxl"): a3(0.01), ("pixtral", "sdxl"): a3(0.02, 12),
           ("siglip", "sdxl"): a3(0.03, 12)}
    table, ok = sensitivity.a3_cross_reader_table(res, names)
    assert ok[("sdxl", "pickscore", "stk")] is True
    assert "stk (4)" in table and "-39.3%" not in table
    res[("siglip", "sdxl")] = a3(-0.001)
    assert sensitivity.a3_cross_reader_table(res, names)[1][("sdxl", "pickscore", "stk")] is False
    _, ok2 = sensitivity.a3_cross_reader_table(res, names[:2])
    assert ok2[("sdxl", "pickscore", "stk")] is False
