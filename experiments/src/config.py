"""Run configuration: parsed and VALIDATED, never assumed.

The code review found `run.py --config` never opened the file it was handed --
it raised the GPU blocker first, so a typo in `configs/pilot.yaml` would have
been discovered only on the rented machine, after the weights had downloaded.
Everything that can be checked without a GPU is checked here, on CPU, before
anything is loaded.
"""

from __future__ import annotations

import os

__all__ = ["RunConfig", "load_config", "ConfigError"]


class ConfigError(ValueError):
    pass


_REQUIRED = ("domain", "countries", "templates", "n", "n_pools", "k", "m_auth")


class RunConfig(dict):
    """A validated config.  Attribute access for the fields that are required."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:                      # pragma: no cover - defensive
            raise AttributeError(name) from exc

    @property
    def n_prompts(self):
        return len(self["countries"]) * len(self["templates"])

    @property
    def n_images_per_model(self):
        return self.n_prompts * self["n_pools"] * self["n"]

    def budget(self):
        """What this config actually costs, in the units the plan is written in."""
        models = len(self.get("generators", []) or [{}])
        return {
            "prompt_strings": self.n_prompts,
            "images_per_model": self.n_images_per_model,
            "models": models,
            "images_total": self.n_images_per_model * models,
            "pools_per_country": len(self["templates"]) * self["n_pools"],
            "known_label_images": (
                (self.get("known_label") or {}).get("pools_per_country", 0)
                * len(self["countries"]) * self["n"] * models),
            "tuning_images": (
                (self.get("tune") or {}).get("pools_per_country", 0)
                * len(self["countries"]) * self["n"] * models),
            "human_annotated_images": 0,
        }


def load_config(path):
    try:
        import yaml
    except ImportError as exc:                       # pragma: no cover
        raise ConfigError("pyyaml is required to read a run config") from exc
    if not os.path.exists(path):
        raise ConfigError(f"config not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ConfigError(f"{path}: missing required keys {missing}")
    for key in ("n", "n_pools", "k", "m_auth"):
        if not isinstance(raw[key], int) or raw[key] < 1:
            raise ConfigError(f"{path}: {key} must be a positive integer")
    if raw["k"] > raw["n"]:
        raise ConfigError(f"{path}: k={raw['k']} exceeds pool size n={raw['n']}")
    if raw["m_auth"] > raw["k"]:
        raise ConfigError(
            f"{path}: m_auth={raw['m_auth']} exceeds k={raw['k']}; the common "
            "rarefaction count cannot be larger than the selected set")
    sk = raw.get("secondary_k")
    if sk is not None:
        sks = [sk] if isinstance(sk, int) else list(sk)
        if any(x > raw["n"] for x in sks):
            raise ConfigError(f"{path}: a secondary_k exceeds n={raw['n']}")
        raw["secondary_k"] = sks
    if not raw["countries"]:
        raise ConfigError(f"{path}: countries is empty")
    if len(set(raw["countries"])) != len(raw["countries"]):
        raise ConfigError(f"{path}: duplicate countries")

    # Design checks the analysis will otherwise fail on, hours later.
    from .analysis import (MIN_POOLS_FOR_EQUIVALENCE,
                           MIN_POOLS_FOR_HETEROGENEITY)
    cfg = RunConfig(raw)
    if len(raw["countries"]) < MIN_POOLS_FOR_EQUIVALENCE:
        raise ConfigError(
            f"{path}: {len(raw['countries'])} countries; the equivalence "
            f"verdict is taken on per-country means and needs at least "
            f"{MIN_POOLS_FOR_EQUIVALENCE}")
    pools_per_country = len(raw["templates"]) * raw["n_pools"]
    cfg["warnings"] = []
    if pools_per_country < MIN_POOLS_FOR_HETEROGENEITY:
        cfg["warnings"].append(
            f"{pools_per_country} pools per country is below the "
            f"{MIN_POOLS_FOR_HETEROGENEITY} at which the heterogeneity test is "
            "calibrated; A4 will be reported as underpowered")
    if raw.get("human_audit"):
        raise ConfigError(
            f"{path}: a human_audit block is present.  This study uses no "
            "human-subject annotation; the R1 check is the known_label block")
    kl = raw.get("known_label") or {}
    if not kl.get("pools_per_country"):
        cfg["warnings"].append(
            "no known_label block: the audit endpoint cannot run, and the "
            "headline claim is gated on it (R1)")
    return cfg
