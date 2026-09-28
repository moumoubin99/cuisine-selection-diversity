"""Human-audit labels, matched to images by ID.

The Phase-4 reviewer made the blinded human audit the gate on the headline
claim (R1), and the code review then found that nothing in the code could
actually accept one: `audit=True` read the mock generator's planted label,
which exists only in simulation.  A real audit produces a CSV from two blinded
annotators, and it has to be joined to the exact images that were generated --
not to a re-run, not to a re-ordering, not to "the same prompt".

Four rules are enforced here, because each of them is a way the join can be
silently wrong:

1. **Image IDs are derived, not stored.**  `image_ids_for_pool` reconstructs
   them from (generator_id, prompt_id, pool_seed, index), the same tuple that
   determines the images themselves, so an annotation file can never be matched
   against a pool it was not produced from.
2. **The generator is part of the key.**  The round-2 code review's P1: the key
   used to be (prompt_id, pool_seed, index) only, so SDXL pool 7 of
   `cuisine|Nigeria|A` and FLUX pool 7 of the same prompt produced *identical*
   IDs.  Labels adjudicated on the SDXL images were accepted for the FLUX ones
   as `human_adjudicated` with 100% coverage -- silently auditing one model
   with another model's annotations, in exactly the two-generator replication
   the design calls for.  `generator_id` must therefore identify the weights
   AND the generation configuration (steps, guidance, sampler, size, dtype):
   see `MockGenerator.generator_id` and `load_generator`.
3. **Partial coverage is not silently tolerated.**  A pool whose annotations
   cover only the high-scoring images would make the audit endpoint agree with
   the machine endpoint by construction -- exactly the artefact the audit is
   there to detect.  A pool is audited only if EVERY image in it is annotated.
4. **Unresolved is a label, not a gap; blank is neither.**  An annotator pair
   that could not agree records `unresolved`; it maps to the same status the
   tagger's failure maps to, and is counted.  A row whose `adjudicated` column
   is *blank* is not a label at all -- it is a sheet nobody has finished.  An
   earlier version read a blank field as "yes" (`or "yes"` on a missing value),
   so the untouched blank template loaded as a complete set of `unresolved`
   annotations and a half-finished sheet reported full coverage.
"""

from __future__ import annotations

import csv
import os

__all__ = ["image_ids_for_pool", "load_annotations", "audit_labels_for_pool",
           "write_annotation_template", "UNRESOLVED_LABEL", "ID_SEP"]

UNRESOLVED_LABEL = "unresolved"

#: Field separator inside an image ID.  No component may contain it, or two
#: different (generator, prompt, pool) triples could render to the same string.
ID_SEP = "#"

_TRUE = ("yes", "y", "true", "1")


def _component(value, what):
    s = str(value).strip()
    if not s:
        raise ValueError(f"empty {what} in an image ID")
    if ID_SEP in s:
        raise ValueError(
            f"{what} {s!r} contains {ID_SEP!r}, the image-ID separator; IDs "
            "would no longer be uniquely decodable")
    return s


def image_ids_for_pool(generator_id, prompt_id, pool_seed, n):
    """Stable IDs for the images of one pool.

    Deterministic in the tuple that also determines the images -- generator
    (weights *and* generation configuration), prompt, pool seed, index -- so
    the join key cannot drift from what was generated, and two generators can
    never be handed each other's labels.
    """
    gid = _component(generator_id, "generator_id")
    pid = _component(prompt_id, "prompt_id")
    return [f"{gid}{ID_SEP}{pid}{ID_SEP}{int(pool_seed)}{ID_SEP}{i:04d}"
            for i in range(int(n))]


def load_annotations(path):
    """Read an adjudicated annotation CSV.

    Required columns: image_id, country, artifact.  Optional: adjudicated
    ('yes'/'no'), annotator_a, annotator_b, notes.

    If the `adjudicated` column is present, a row is accepted only when it
    holds an explicit affirmative; blank and 'no' are both rejected, because an
    unfinished row and an un-adjudicated disagreement are equally not labels.
    A file with no `adjudicated` column at all declares no adjudication
    workflow and every row is taken as final.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"annotation file not found: {path}")
    out = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = {"image_id", "country", "artifact"} - fields
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        has_adj = "adjudicated" in fields
        for i, row in enumerate(reader, start=2):
            iid = (row["image_id"] or "").strip()
            if not iid:
                raise ValueError(f"{path}:{i}: empty image_id")
            if iid in out:
                raise ValueError(f"{path}:{i}: duplicate image_id {iid!r}")
            if has_adj and (row.get("adjudicated") or "").strip().lower() not in _TRUE:
                continue
            out[iid] = ((row["country"] or "").strip(),
                        (row["artifact"] or UNRESOLVED_LABEL).strip() or UNRESOLVED_LABEL)
    return out


def audit_labels_for_pool(generator_id, prompt_id, pool_seed, n, annotations):
    """(labels, coverage) for one pool, or (None, coverage) if incomplete.

    `labels[i]` is the adjudicated artifact name, or None where the annotators
    could not resolve the image.  Returning None for a partially covered pool
    is deliberate: see rule 3 in the module docstring.
    """
    ids = image_ids_for_pool(generator_id, prompt_id, pool_seed, n)
    have = [i for i, iid in enumerate(ids) if iid in annotations]
    coverage = {"n_images": len(ids), "n_annotated": len(have),
                "fraction": len(have) / float(len(ids)) if ids else 0.0,
                "complete": len(have) == len(ids),
                "generator_id": str(generator_id)}
    if not coverage["complete"]:
        coverage["missing_examples"] = [iid for iid in ids
                                        if iid not in annotations][:3]
        return None, coverage
    labels = []
    for iid in ids:
        country, artifact = annotations[iid]
        labels.append(None if artifact == UNRESOLVED_LABEL else artifact)
    coverage["n_unresolved"] = sum(1 for x in labels if x is None)
    coverage["countries"] = sorted({annotations[i][0] for i in ids})
    return labels, coverage


def write_annotation_template(path, rows):
    """Emit the blank sheet the annotators fill in.

    `rows` are (image_id, prompt_id, country, image_path) tuples.  Score,
    condition and the CSpace vocabulary are deliberately NOT columns: the
    protocol is free-text identity first, vocabulary mapping afterwards, and an
    annotator who can see the score is no longer blinded.

    Every `adjudicated` cell is left blank, and a blank cell is not a label:
    the untouched sheet loads as zero annotations, not as a pool of
    `unresolved` ones.
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "prompt_id", "country_prompted", "image_path",
                    "annotator_a", "annotator_b", "adjudicated", "country",
                    "artifact", "notes"])
        for iid, pid, country, img in rows:
            w.writerow([iid, pid, country, img, "", "", "", "", "", ""])
    return path
