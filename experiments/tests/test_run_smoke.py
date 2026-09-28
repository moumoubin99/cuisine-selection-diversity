"""The entrypoint itself.

This runs at the design's real size (8 countries, 4 pools per prompt) rather
than a reduced one, because two of the five controls are statements about
*precision* -- `sub_margin_world_is_equivalent` and `null_world_is_equivalent`
both require the interval to fit inside the pre-registered margin -- and a
shrunken design cannot satisfy them no matter how correct the code is.  The
second test pins that down instead of hiding it.
"""
import json
import os

import numpy as np
import pytest

import run
from src.pipeline import run_audit


def test_smoke_entrypoint_passes_all_controls(tmp_path):
    rc = run.smoke(n=96, k=16, m_auth=8, n_pools=4, n_null=40,
                   out_dir=str(tmp_path), seed=0)
    ok = json.load(open(os.path.join(tmp_path, "controls.json")))
    assert ok["passed"], ok["checks"]
    assert rc == 0
    for w in run.WORLDS:
        assert os.path.exists(os.path.join(tmp_path, f"{w}.json"))
    assert len(ok["checks"]) >= 9


def test_a_smaller_design_cannot_certify_equivalence_and_says_so():
    """Measured, and reported as `inconclusive` rather than as a null result.

    At 3 pools per prompt with k=12 the null world's interval is wider than the
    10% margin, so the truthful verdict is `inconclusive`.  An implementation
    that returned `equivalent` here would be claiming a precision the design
    does not have -- which is the specific failure the Phase-4 reviewer's R6
    and the code review's finding on the bootstrap bounds were both about.
    """
    cfg = run.WORLDS["null_world"]
    gen, scorer, tagger, prompts, vocab, proto = run._world(
        cfg["taste"], cfg["merge"], seed=0, noise=cfg["noise"])
    out = run_audit(gen, scorer, tagger, prompts, vocab, n=64, k=12, m_auth=6,
                    n_pools=3, seed=0, n_null=30)["conditional"]
    assert out["equivalence"]["verdict"] == "inconclusive"
    lo, hi = out["equivalence"]["ci"]
    assert (hi - lo) > 2 * abs(np.log(0.90)) * 0.5


def test_real_backend_path_preflights_then_refuses_loudly(tmp_path):
    with pytest.raises(SystemExit) as e:
        run.real("configs/pilot.yaml", out_dir=str(tmp_path))
    msg = str(e.value)
    assert "not a measurement" in msg
    assert "Unavailable backends" in msg
    report = json.load(open(os.path.join(tmp_path, "preflight.json")))
    assert report["ready_to_run"] is False
    assert report["budget"]["images_total"] == 8192
    assert len(report["unavailable_backends"]) == 3


def test_preflight_reports_annotation_coverage(tmp_path):
    import csv
    from src.annotations import image_ids_for_pool
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(open("configs/pilot.yaml").read())
    rep0 = run.preflight(str(cfg_path))
    sdxl = rep0["generator_ids"]["stabilityai/stable-diffusion-xl-base-1.0"]
    ann = tmp_path / "a.csv"
    with open(ann, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "country", "artifact"])
        for iid in image_ids_for_pool(sdxl, "cuisine|Brazil|A", 0, 64):
            w.writerow([iid, "Brazil", "feijoada"])
    rep = run.preflight(str(cfg_path), str(ann))
    assert rep["annotations"]["fully_annotated_pools_total"] == 1
    assert rep["annotations"]["n_rows"] == 64
    # Coverage is reported PER GENERATOR, and the second model has none of it:
    # the same 64 rows would have counted for both before the generator became
    # part of the join key.
    assert rep["annotations"]["fully_annotated_pools"] == {
        "stabilityai/stable-diffusion-xl-base-1.0": 1,
        "black-forest-labs/FLUX.1-schnell": 0}
    assert any("FLUX" in w for w in rep["warnings"])
