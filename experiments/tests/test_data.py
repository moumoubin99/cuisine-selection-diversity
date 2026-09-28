import pytest
from src.data import (COUNTRIES, DIVERSITY_DOMAINS, DEMONYM,
                      build_underspecified_prompts, cspace_vocabulary,
                      artifact_frequencies, Prompt)

def test_every_country_has_a_demonym():
    assert set(DEMONYM) == set(COUNTRIES)

def test_prompts_never_name_an_artifact():
    """The whole point: a diversity prompt must leave the artifact free."""
    for p in build_underspecified_prompts():
        assert p.text.startswith("An image of ")
        # no specific dish/landmark/artwork name appears
        assert not any(tok in p.text.lower()
                       for tok in ["carne", "pizza", "sushi", "eiffel", "taj"])

def test_prompt_set_shape():
    ps = build_underspecified_prompts()
    # 3 domains x (8 countries x 2 templates + 1 global)
    assert len(ps) == 3 * (8 * 2 + 1)
    assert len({p.prompt_id for p in ps}) == len(ps)

def test_global_prompts_name_no_country():
    for p in build_underspecified_prompts():
        if p.country == "GLOBAL":
            assert not any(c.lower() in p.text.lower() for c in COUNTRIES)
            assert not any(d.lower() in p.text.lower() for d in DEMONYM.values())

def test_single_domain_selection():
    ps = build_underspecified_prompts(domains=["cuisine"])
    assert {p.domain for p in ps} == {"cuisine"}
    assert len(ps) == 17

def test_cspace_helpers_on_a_stub():
    cspace = [
        {"name": "moqueca", "country": "Brazil", "domain": "cuisine"},
        {"name": "moqueca", "country": "Brazil", "domain": "cuisine"},
        {"name": "acaraje", "country": "Brazil", "domain": "cuisine"},
        {"name": "pizza", "country": "Italy", "domain": "cuisine"},
        {"name": "None", "country": "Brazil", "domain": "cuisine"},
    ]
    assert cspace_vocabulary(cspace, "Brazil", "cuisine") == ["acaraje", "moqueca"]
    assert artifact_frequencies(cspace, "Brazil", "cuisine") == {"moqueca": 2, "acaraje": 1}


def test_artifact_frequencies_are_degenerate_on_real_cspace_stub():
    """Pins the documented reason frequencies are not used as the proxy."""
    from src.data import artifact_frequencies
    cspace = [{"name": f"a{i}", "country": "X", "domain": "cuisine"} for i in range(50)]
    f = artifact_frequencies(cspace, "X", "cuisine")
    assert set(f.values()) == {1}


def test_kb_prominence_uses_properties_and_title():
    from src.data import kb_prominence
    cspace = [
        {"name": "rich", "country": "X", "domain": "cuisine", "title": "Rich",
         "P31": "['Q1']", "P279": "['Q2']", "P495": "['Q3']", "P17": "['Q4']", "P361": "['Q5']"},
        {"name": "bare", "country": "X", "domain": "cuisine", "title": "None",
         "P31": "[]", "P279": "[]", "P495": "[]", "P17": "[]", "P361": "[]"},
    ]
    p = kb_prominence(cspace, "X", "cuisine")
    assert p["rich"] == pytest.approx(1.0)
    assert p["bare"] == pytest.approx(0.0)
