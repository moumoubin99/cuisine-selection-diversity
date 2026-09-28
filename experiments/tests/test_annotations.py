"""The human-audit join.

Every test here corresponds to a way the join can be silently wrong and still
produce plausible numbers -- which is worse than producing none.
"""
import csv

import numpy as np
import pytest

from src.annotations import (audit_labels_for_pool, image_ids_for_pool,
                             load_annotations, write_annotation_template)
from src.backends import (MockGenerator, MockGeoTagger, MockScorer,
                          generator_fingerprint)
from src.data import Prompt
from src.pipeline import run_pool

PROMPT = Prompt("cuisine|Nigeria|A", "An image of a traditional dish from Nigeria",
                "Nigeria", "cuisine", "A")
VOCAB = {"Nigeria": [f"n{i}" for i in range(40)]}
GID = MockGenerator(VOCAB, seed=0).generator_id


def _world(seed=0):
    g = MockGenerator(VOCAB, seed=seed)
    w = g.artifact_embedding("Nigeria", "n0")
    w = w / np.linalg.norm(w)
    return (g, MockScorer(w, taste_weight=1.0, noise=0.05, seed=seed + 1),
            MockGeoTagger({}, seed=seed + 2))


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "country", "artifact", "adjudicated"])
        for r in rows:
            w.writerow(r)
    return str(path)


def test_image_ids_are_determined_by_the_tuple_that_makes_the_images():
    a = image_ids_for_pool(GID, "cuisine|Nigeria|A", 7, 4)
    assert a == image_ids_for_pool(GID, "cuisine|Nigeria|A", 7, 4)
    assert a != image_ids_for_pool(GID, "cuisine|Nigeria|A", 8, 4)
    assert a != image_ids_for_pool(GID, "cuisine|Italy|A", 7, 4)
    assert len(set(a)) == 4


def test_two_generators_never_share_an_image_id():
    """Round-2 P1: the join key omitted the generator entirely.

    SDXL pool 7 of a prompt and FLUX pool 7 of the same prompt produced
    identical IDs, so labels adjudicated on one model's images were accepted
    for the other's as `human_adjudicated` at 100% coverage -- in exactly the
    two-generator replication the design calls for.
    """
    sdxl = generator_fingerprint("stabilityai/stable-diffusion-xl-base-1.0",
                                 {"steps": 30, "guidance_scale": 7.0})
    flux = generator_fingerprint("black-forest-labs/FLUX.1-schnell",
                                 {"steps": 4, "guidance_scale": 0.0})
    a = image_ids_for_pool(sdxl, "cuisine|Nigeria|A", 7, 16)
    b = image_ids_for_pool(flux, "cuisine|Nigeria|A", 7, 16)
    assert not (set(a) & set(b))
    # ... and the CONFIGURATION is part of it, not only the checkpoint name.
    same_weights_other_steps = generator_fingerprint(
        "stabilityai/stable-diffusion-xl-base-1.0",
        {"steps": 50, "guidance_scale": 7.0})
    assert same_weights_other_steps != sdxl
    assert not (set(a) & set(image_ids_for_pool(
        same_weights_other_steps, "cuisine|Nigeria|A", 7, 16)))


def test_one_generators_labels_are_not_accepted_for_another(tmp_path):
    """The same failure, end to end through `run_pool`."""
    g1 = MockGenerator(VOCAB, seed=0)
    g2 = MockGenerator(VOCAB, seed=99)               # a different generator
    w = g1.artifact_embedding("Nigeria", "n0"); w = w / np.linalg.norm(w)
    s_, t_ = MockScorer(w, taste_weight=1.0, seed=1), MockGeoTagger({}, seed=2)
    ids = image_ids_for_pool(g1.generator_id, PROMPT.prompt_id, 11, 32)
    path = _write(tmp_path / "g1.csv", [(i, "Nigeria", "one_dish", "yes") for i in ids])
    ann = load_annotations(path)
    ok = run_pool(g1, s_, t_, PROMPT, n=32, k=8, m_auth=4,
                  cspace_vocab=VOCAB["Nigeria"], pool_seed=11, n_null=20,
                  annotations=ann)
    assert ok["audit_coverage"]["complete"]
    wrong = run_pool(g2, s_, t_, PROMPT, n=32, k=8, m_auth=4,
                     cspace_vocab=VOCAB["Nigeria"], pool_seed=11, n_null=20,
                     annotations=ann)
    assert wrong["audit_coverage"]["n_annotated"] == 0
    assert wrong["audit"]["non_estimable"]


def test_a_generator_with_no_identity_cannot_be_audited():
    class _Anon:
        def generate(self, text, n, seed, country=None):
            return [object() for _ in range(n)]

    with pytest.raises(TypeError, match="generator_id"):
        run_pool(_Anon(), None, None, PROMPT, n=4, k=2, m_auth=2,
                 cspace_vocab=VOCAB["Nigeria"], pool_seed=1, n_null=2)


def test_a_blank_adjudication_field_is_not_an_adjudicated_label(tmp_path):
    """Round-2 P2: `(row.get("adjudicated") or "yes")` read blank as yes.

    The untouched blank template therefore loaded as a full pool of completed
    `unresolved` annotations, and a half-finished sheet reported full coverage
    of labels nobody had agreed on.
    """
    ids = image_ids_for_pool(GID, PROMPT.prompt_id, 5, 4)
    path = _write(tmp_path / "blank.csv",
                  [(ids[0], "Nigeria", "n0", "yes")]
                  + [(i, "Nigeria", "draft_guess", "") for i in ids[1:]])
    ann = load_annotations(path)
    assert set(ann) == {ids[0]}
    labels, cov = audit_labels_for_pool(GID, PROMPT.prompt_id, 5, 4, ann)
    assert labels is None and cov["n_annotated"] == 1


def test_the_untouched_template_loads_as_zero_annotations(tmp_path):
    ids = image_ids_for_pool(GID, PROMPT.prompt_id, 5, 4)
    path = write_annotation_template(
        str(tmp_path / "t.csv"),
        [(i, PROMPT.prompt_id, "Nigeria", f"/img/{i}.png") for i in ids])
    assert load_annotations(path) == {}


def test_a_file_without_an_adjudicated_column_is_taken_as_final(tmp_path):
    """No column at all declares no adjudication workflow; blank does not."""
    ids = image_ids_for_pool(GID, PROMPT.prompt_id, 5, 2)
    path = tmp_path / "final.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "country", "artifact"])
        for i in ids:
            w.writerow([i, "Nigeria", "n0"])
    assert len(load_annotations(str(path))) == 2


def test_a_partially_annotated_pool_is_not_audited(tmp_path):
    """The failure mode this blocks: annotating only the high-scoring images.

    That makes the audit endpoint agree with the machine endpoint by
    construction, which is exactly the artefact the audit exists to detect.
    """
    ids = image_ids_for_pool(GID, PROMPT.prompt_id, 5, 8)
    path = _write(tmp_path / "half.csv",
                  [(i, "Nigeria", "n0", "yes") for i in ids[:4]])
    labels, cov = audit_labels_for_pool(GID, PROMPT.prompt_id, 5, 8, load_annotations(path))
    assert labels is None
    assert cov["n_annotated"] == 4 and not cov["complete"]
    assert cov["missing_examples"][0] == ids[4]


def test_unadjudicated_rows_are_not_labels(tmp_path):
    ids = image_ids_for_pool(GID, PROMPT.prompt_id, 5, 4)
    path = _write(tmp_path / "raw.csv",
                  [(ids[0], "Nigeria", "n0", "yes")]
                  + [(i, "Nigeria", "n1", "no") for i in ids[1:]])
    ann = load_annotations(path)
    assert set(ann) == {ids[0]}


def test_duplicate_and_missing_columns_are_errors(tmp_path):
    ids = image_ids_for_pool(GID, PROMPT.prompt_id, 5, 2)
    dup = _write(tmp_path / "dup.csv", [(ids[0], "Nigeria", "n0", "yes"),
                                        (ids[0], "Nigeria", "n1", "yes")])
    with pytest.raises(ValueError, match="duplicate"):
        load_annotations(dup)
    bad = tmp_path / "bad.csv"
    bad.write_text("image_id,artifact\nx,y\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_annotations(str(bad))


def test_unresolved_is_a_label_not_a_gap(tmp_path):
    ids = image_ids_for_pool(GID, PROMPT.prompt_id, 5, 4)
    path = _write(tmp_path / "u.csv",
                  [(ids[0], "Nigeria", "unresolved", "yes")]
                  + [(i, "Nigeria", f"n{j}", "yes") for j, i in enumerate(ids[1:])])
    labels, cov = audit_labels_for_pool(GID, PROMPT.prompt_id, 5, 4, load_annotations(path))
    assert cov["complete"] and cov["n_unresolved"] == 1 and labels[0] is None


def test_run_pool_uses_human_labels_when_they_are_supplied(tmp_path):
    """The audit endpoint must read the CSV, not the planted ground truth."""
    g, s, t = _world()
    n, seed = 32, 11
    ids = image_ids_for_pool(GID, PROMPT.prompt_id, seed, n)
    # Every human label is the SAME artifact: whatever the generator planted,
    # an audit that reads this file has to report a fully collapsed set.
    path = _write(tmp_path / "flat.csv", [(i, "Nigeria", "one_dish", "yes") for i in ids])
    row = run_pool(g, s, t, PROMPT, n=n, k=8, m_auth=4, cspace_vocab=VOCAB["Nigeria"],
                   pool_seed=seed, n_null=20, annotations=load_annotations(path))
    assert row["audit_source"] == "human_adjudicated"
    assert row["audit"]["diversity_selected"] == pytest.approx(1.0 / 4, abs=1e-9)
    assert row["audit_coverage"]["complete"]


def test_annotations_for_the_wrong_pool_do_not_silently_match(tmp_path):
    g, s, t = _world()
    ids = image_ids_for_pool(GID, PROMPT.prompt_id, 999, 32)      # a different pool
    path = _write(tmp_path / "wrong.csv", [(i, "Nigeria", "x", "yes") for i in ids])
    row = run_pool(g, s, t, PROMPT, n=32, k=8, m_auth=4, cspace_vocab=VOCAB["Nigeria"],
                   pool_seed=11, n_null=20, annotations=load_annotations(path))
    assert row["audit"]["non_estimable"]
    assert row["audit_coverage"]["n_annotated"] == 0


def test_planted_ground_truth_is_refused_for_images_that_do_not_carry_one():
    class _Plain:
        generator_id = "plain@0000"

        def generate(self, text, n, seed, country=None):
            return [object() for _ in range(n)]

    class _S:
        def score(self, images, text=None):
            return np.arange(len(images), dtype=float)

    class _T:
        def tag(self, images, text=None, examples=None):
            return [("C", "Nigeria", "n0")] * len(images)

    with pytest.raises(TypeError, match="annotation file"):
        run_pool(_Plain(), _S(), _T(), PROMPT, n=8, k=4, m_auth=2,
                 cspace_vocab=VOCAB["Nigeria"], pool_seed=1, n_null=5, audit=True)


def test_annotation_template_does_not_leak_the_score_or_the_condition(tmp_path):
    path = write_annotation_template(
        str(tmp_path / "t.csv"),
        [("id1", PROMPT.prompt_id, "Nigeria", "/img/1.png")])
    header = open(path).readline().strip().split(",")
    assert "score" not in header and "condition" not in header
    assert "rank" not in header and "selected" not in header
    assert header[:4] == ["image_id", "prompt_id", "country_prompted", "image_path"]
