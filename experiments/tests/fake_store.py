"""A synthetic store in the on-disk layout of `run_gpu.py`, for CPU tests.

Every pool has a hidden dish per image.  In main and tuning pools the scorer
prefers the country's first few dishes (so top-k is homogenising by
construction); in known-label pools the hidden dish is the prompted one.  The
tagger reports the hidden dish with probability `tag_acc`, else "unknown".
"""
import json
import os

import numpy as np

from src.store import append_jsonl, gen_tag, generator_config, load_vocab_file, plan_pools

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = os.path.join(HERE, "data", "cspace_cuisine_vocab.json")


def small_config(tmp, n=32, k=16, n_pools=2, n_null=100):
    import yaml
    with open(os.path.join(HERE, "configs", "pilot.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    cfg.update(n=n, k=k, n_pools=n_pools, n_null=n_null, secondary_k=None)
    cfg["generators"] = cfg["generators"][:1]
    p = os.path.join(tmp, "cfg.yaml")
    with open(p, "w") as fh:
        yaml.safe_dump(cfg, fh)
    return p


def build(root, cfg, tag_acc=0.9, head=4, bias=2.0, seed=0, dim=16):
    rng = np.random.default_rng(seed)
    full, known = load_vocab_file(VOCAB, cfg["known_label"]["min_kb_prominence"])
    g = cfg["generators"][0]
    tag = gen_tag(g)
    gid = "fake:" + json.dumps(generator_config(g, cfg), sort_keys=True)
    man, sc_p, sc_i, tags, ver, ids, E = [], [], [], [], [], [], []
    basis = {}
    for spec in plan_pools(cfg, known):
        c = spec.prompt.country
        vocab = full[c][:24]
        for idx, gp in enumerate(spec.gen_prompts):
            iid = f"{tag}-{spec.pool_seed}-{idx}"
            if spec.kind == "known":
                dish = spec.dishes[idx]
            else:
                dish = vocab[min(int(rng.zipf(1.3)) - 1, len(vocab) - 1)]
            q = rng.normal() + (bias if dish in vocab[:head] else 0.0)
            man.append({"image_id": iid, "generator_id": gid, "kind": spec.kind,
                        "prompt_id": spec.prompt.prompt_id, "country": c,
                        "template": spec.prompt.template, "pool_seed": spec.pool_seed,
                        "index": idx, "image_seed": idx, "gen_prompt": gp,
                        "select_text": spec.prompt.text,
                        "dish": dish if spec.kind == "known" else None,
                        "path": f"images/{tag}/{spec.pool_seed}/{idx}.jpg"})
            sc_p.append({"image_id": iid, "score": q, "text": spec.prompt.text})
            sc_i.append({"image_id": iid, "score": q + 0.5 * rng.normal(),
                         "text": spec.prompt.text})
            ok = rng.random() < tag_acc
            tags.append({"image_id": iid, "reply": "", "dish": dish if ok else None,
                         "country": c if ok else None})
            if spec.kind == "known":
                kept = bool(rng.random() < 0.85)
                ver.append({"image_id": iid, "dish": dish, "kept": kept,
                            "p_yes": 0.9, "p_yes_decoy": 0.6 if kept else 0.9})
            if dish not in basis:
                basis[dish] = rng.normal(size=dim)
            ids.append(iid)
            E.append(basis[dish] + 0.3 * rng.normal(size=dim))
    for sub in ("manifest", "scores", "tags", "verify", "embed"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    append_jsonl(os.path.join(root, "manifest", f"{tag}.jsonl"), man)
    append_jsonl(os.path.join(root, "scores", f"{tag}__pickscore.jsonl"), sc_p)
    append_jsonl(os.path.join(root, "scores", f"{tag}__imagereward.jsonl"), sc_i)
    append_jsonl(os.path.join(root, "tags", f"{tag}.jsonl"), tags)
    append_jsonl(os.path.join(root, "verify", f"{tag}.jsonl"), ver)
    np.savez(os.path.join(root, "embed", f"{tag}__dinov2.npz"),
             ids=np.asarray(ids), E=np.asarray(E, dtype=np.float32))
    return tag
