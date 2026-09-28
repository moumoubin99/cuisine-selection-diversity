"""Config validation runs on CPU, before any weight is downloaded."""
import pytest
import yaml

from src.analysis import MIN_POOLS_FOR_EQUIVALENCE
from src.config import ConfigError, load_config

BASE = dict(seed=0, domain="cuisine",
            countries=["Brazil", "France", "India", "Italy", "Japan",
                       "Nigeria", "Turkey", "United States"],
            templates=["A", "B"], n=64, n_pools=4, k=16, m_auth=8)


def _write(tmp_path, **over):
    cfg = dict(BASE)
    cfg.update(over)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_the_shipped_pilot_config_validates():
    cfg = load_config("configs/pilot.yaml")
    b = cfg.budget()
    assert b["prompt_strings"] == 16
    assert b["images_total"] == 8192
    assert b["pools_per_country"] == 8


def test_k_larger_than_the_pool_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="exceeds pool size"):
        load_config(_write(tmp_path, k=128))


def test_rarefaction_count_larger_than_k_is_rejected(tmp_path):
    """m > k makes every pool non-estimable, hours into the run."""
    with pytest.raises(ConfigError, match="cannot be larger"):
        load_config(_write(tmp_path, m_auth=32))


def test_too_few_countries_for_the_equivalence_verdict_is_rejected(tmp_path):
    few = BASE["countries"][:MIN_POOLS_FOR_EQUIVALENCE - 1]
    with pytest.raises(ConfigError, match="equivalence verdict"):
        load_config(_write(tmp_path, countries=few))


def test_a_design_too_small_for_the_heterogeneity_test_warns(tmp_path):
    cfg = load_config(_write(tmp_path, templates=["A"], n_pools=2))
    assert any("heterogeneity" in w for w in cfg["warnings"])


def test_a_config_without_known_label_pools_warns_that_the_claim_is_ungated(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert any("known_label" in w for w in cfg["warnings"])


def test_a_human_audit_block_is_refused(tmp_path):
    # No human-subject annotation in this study: a leftover block from the
    # pre-registration must not silently reintroduce one.
    with pytest.raises(ConfigError, match="human-subject"):
        load_config(_write(tmp_path, human_audit={"pools_per_country_per_model": 2}))


def test_missing_keys_and_missing_file_are_errors(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"domain": "cuisine"}))
    with pytest.raises(ConfigError, match="missing required keys"):
        load_config(str(p))
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "nope.yaml"))
