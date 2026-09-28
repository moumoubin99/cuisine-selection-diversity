#!/usr/bin/env python3
"""GPU stages of the real run.  Each stage is resumable and writes to the store.

    python run_gpu.py generate --gen sdxl   [--kinds main,known,tune] [--limit-pools N]
    python run_gpu.py score    --gen sdxl   --scorers pickscore,imagereward
    python run_gpu.py tag      --gen sdxl
    python run_gpu.py verify   --gen sdxl       # known-label pools only
    python run_gpu.py tag      --gen sdxl --reader pixtral   # amendment 6 readers
    python run_gpu.py embed    --gen sdxl [--embedder siglip]
    python run_gpu.py bench    --gen sdxl --n 16

Order matters for one reason: tags are produced from images alone and the tag
stage never opens a score file, so the tagger stays blind to the selector even
though both live in the same store.  `analyze.py` does everything after this on
CPU, through the same `pipeline.run_audit` the smoke controls validate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.annotations import image_ids_for_pool
from src.backends import generator_fingerprint
from src.config import load_config
from src.store import (append_jsonl, gen_tag, generator_config, load_vocab_file,
                       plan_pools, read_jsonl, reader_file)

HERE = os.path.dirname(os.path.abspath(__file__))


def _setup(a):
    cfg = load_config(a.config)
    kl = cfg.get("known_label") or {}
    full, known = load_vocab_file(a.vocab, kl.get("min_kb_prominence"))
    gens = {gen_tag(g): g for g in cfg["generators"]}
    g = gens[a.gen]
    gid = generator_fingerprint(g["id"], generator_config(g, cfg))
    plan = plan_pools(cfg, known)
    return cfg, g, gid, plan, full


def _log(*x):
    print(time.strftime("[%H:%M:%S]"), *x, flush=True)


def _manifest(a):
    return os.path.join(a.root, "manifest", f"{a.gen}.jsonl")


def cmd_generate(a):
    cfg, g, gid, plan, _ = _setup(a)
    from src.gpu_backends import load_diffusers_generator
    kinds = set(a.kinds.split(","))
    done = {}
    for r in read_jsonl(_manifest(a)):
        if r["generator_id"] != gid:
            raise SystemExit(f"manifest {_manifest(a)} holds images of {r['generator_id']}, "
                             f"config now fingerprints {gid}: settings changed; use a new root")
        done[r["pool_seed"]] = done.get(r["pool_seed"], 0) + 1
    todo = [p for p in plan if p.kind in kinds and done.get(p.pool_seed, 0) < cfg["n"]]
    if a.limit_pools:
        todo = todo[:a.limit_pools]
    _log(f"{a.gen}: {len(todo)} pools to generate ({gid})")
    if not todo:
        return 0
    gc = dict(g)
    if a.batch:
        gc["batch"] = a.batch
    gen = load_diffusers_generator(gc, a.models_dir)
    t0, n_img = time.time(), 0
    for p in todo:
        if done.get(p.pool_seed, 0):
            raise SystemExit(f"pool {p.pool_seed} is partially written; remove its "
                             "manifest rows before resuming")
        ts = time.time()
        images, seeds = gen.generate_pool(p.gen_prompts, p.pool_seed)
        ids = image_ids_for_pool(gid, p.prompt.prompt_id, p.pool_seed, cfg["n"])
        d = os.path.join("images", a.gen, str(p.pool_seed))
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
        rows = []
        for i, (im, s) in enumerate(zip(images, seeds)):
            rel = os.path.join(d, f"{i:04d}.jpg")
            im.save(os.path.join(a.root, rel), quality=95)
            rows.append({"image_id": ids[i], "generator_id": gid, "kind": p.kind,
                         "prompt_id": p.prompt.prompt_id, "country": p.prompt.country,
                         "template": p.prompt.template, "pool_seed": p.pool_seed,
                         "index": i, "image_seed": s, "gen_prompt": p.gen_prompts[i],
                         "select_text": p.prompt.text,
                         "dish": p.dishes[i] if p.dishes else None, "path": rel})
        append_jsonl(_manifest(a), rows)
        n_img += len(rows)
        el = time.time() - t0
        _log(f"pool {p.kind}/{p.prompt.prompt_id}/{p.pool_seed}: {time.time()-ts:.1f}s; "
             f"{n_img} imgs, {el/n_img:.2f} s/img")
    return 0


def _images(a, kinds=None):
    rows = read_jsonl(_manifest(a))
    if kinds:
        rows = [r for r in rows if r["kind"] in kinds]
    return rows


def cmd_score(a):
    rows = _images(a)
    for name in a.scorers.split(","):
        out = os.path.join(a.root, "scores", f"{a.gen}__{name}.jsonl")
        have = {r["image_id"] for r in read_jsonl(out)}
        todo = [r for r in rows if r["image_id"] not in have]
        _log(f"{name}: {len(todo)} images to score")
        if not todo:
            continue
        from src.gpu_backends import load_real_scorer
        sc = load_real_scorer(name, models_dir=a.models_dir)
        by_text = {}
        for r in todo:
            by_text.setdefault(r["select_text"], []).append(r)
        for text, rs in by_text.items():
            for s in range(0, len(rs), 256):
                chunk = rs[s:s + 256]
                v = sc.score([os.path.join(a.root, r["path"]) for r in chunk], text)
                append_jsonl(out, [{"image_id": r["image_id"], "score": float(x),
                                    "text": text} for r, x in zip(chunk, v)])
            _log(f"{name}: scored {len(rs)} for {text!r}")
        del sc
        _free()
    return 0


def _free():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def cmd_tag(a):
    rows = _images(a)
    out = reader_file(a.root, "tags", a.gen, a.reader)
    have = {r["image_id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["image_id"] not in have]
    _log(f"tag[{a.reader}]: {len(todo)} images")
    if not todo:
        return 0
    from src.gpu_backends import load_reader
    from src.labels import parse_tagger_json
    vlm = load_reader(a.reader, a.models_dir, vocab_by_country=_setup(a)[4])
    t0 = time.time()
    for s in range(0, len(todo), 128):
        chunk = todo[s:s + 128]
        replies = vlm.tag([os.path.join(a.root, r["path"]) for r in chunk], batch=a.vlm_batch)
        rows_out = []
        for r, rep in zip(chunk, replies):
            dish, country = parse_tagger_json(rep)
            rows_out.append({"image_id": r["image_id"], "reply": rep,
                             "dish": dish, "country": country})
        append_jsonl(out, rows_out)
        _log(f"tag: {s+len(chunk)}/{len(todo)}  {(time.time()-t0)/(s+len(chunk)):.2f} s/img")
    return 0


def verifier_options(row, pool_dishes, fallback):
    """The forced-choice options for one known-label image: the prompted dish
    and three decoys at a position fixed by the image ID, then "none of these".
    Decoys come from the same pool's dishes (the hardest plausible confusions),
    topped up from the country's known-label vocabulary."""
    import zlib
    rng = np.random.default_rng(zlib.crc32(row["image_id"].encode()))
    cand = sorted({d for d in pool_dishes if d != row["dish"]})
    if len(cand) < 3:
        cand += [d for d in sorted(fallback) if d != row["dish"] and d not in cand]
    decoys = list(rng.choice(cand, 3, replace=False))
    pos = int(rng.integers(4))
    opts = decoys[:pos] + [row["dish"]] + decoys[pos:]
    return opts + ["none of these"], pos, decoys[0]


def cmd_verify(a):
    rows = _images(a, kinds={"known"})
    out = reader_file(a.root, "verify", a.gen, a.reader)
    have = {r["image_id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["image_id"] not in have]
    _log(f"verify[{a.reader}]: {len(todo)} images")
    if not todo:
        return 0
    _, known = load_vocab_file(a.vocab, _setup(a)[0]["known_label"]["min_kb_prominence"])
    by_pool = {}
    for r in rows:
        by_pool.setdefault(r["pool_seed"], set()).add(r["dish"])
    from src.gpu_backends import load_reader
    vlm = load_reader(a.reader, a.models_dir)
    for s in range(0, len(todo), 128):
        chunk = todo[s:s + 128]
        paths = [os.path.join(a.root, r["path"]) for r in chunk]
        mc = [verifier_options(r, by_pool[r["pool_seed"]], known[r["country"]]) for r in chunk]
        probs = vlm.choose(paths, [m[0] for m in mc], batch=a.vlm_batch)
        py = vlm.verify(paths, [r["dish"] for r in chunk], [r["country"] for r in chunk],
                        batch=a.vlm_batch)
        pd = vlm.verify(paths, [m[2] for m in mc], [r["country"] for r in chunk],
                        batch=a.vlm_batch)
        append_jsonl(out, [{"image_id": r["image_id"], "dish": r["dish"],
                            "options": m[0], "true_pos": m[1],
                            "p_choice": [float(v) for v in pr],
                            "chosen": int(np.argmax(pr)),
                            "kept": bool(int(np.argmax(pr)) == m[1]),
                            "p_yes": float(y), "decoy": m[2], "p_yes_decoy": float(d)}
                           for r, m, pr, y, d in zip(chunk, mc, probs, py, pd)])
        _log(f"verify: {s+len(chunk)}/{len(todo)}")
    return 0


def cmd_embed(a):
    rows = _images(a)
    path = os.path.join(a.root, "embed", f"{a.gen}__{a.embedder}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if set(z["ids"].tolist()) >= {r["image_id"] for r in rows}:
            _log("embed: up to date")
            return 0
    from src.gpu_backends import load_embedder
    emb = load_embedder(a.embedder, models_dir=a.models_dir)
    E = emb.embed([os.path.join(a.root, r["path"]) for r in rows])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, ids=np.array([r["image_id"] for r in rows]), E=E.astype(np.float32))
    _log(f"embed: {E.shape}")
    return 0


def cmd_bench(a):
    """Throughput of each model on a handful of images, written to the store."""
    cfg, g, gid, plan, _ = _setup(a)
    import torch
    from src.gpu_backends import load_diffusers_generator
    gc = dict(g)
    if a.batch:
        gc["batch"] = a.batch
    res = {"generator": g["id"], "generator_id": gid,
           "gpu": torch.cuda.get_device_name(0)}
    t = time.time()
    gen = load_diffusers_generator(gc, a.models_dir)
    res["load_s"] = time.time() - t
    p = plan[0]
    gen.generate_pool(p.gen_prompts[:gen.batch], p.pool_seed)       # warm-up
    torch.cuda.synchronize()
    t = time.time()
    ims, _ = gen.generate_pool(p.gen_prompts[:a.n], p.pool_seed + 7)
    torch.cuda.synchronize()
    res["s_per_image"] = (time.time() - t) / a.n
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    d = os.path.join(a.root, "bench", a.gen)
    os.makedirs(d, exist_ok=True)
    for i, im in enumerate(ims[:4]):
        im.save(os.path.join(d, f"{i}.jpg"), quality=90)
    with open(os.path.join(d, "bench.json"), "w") as f:
        json.dump(res, f, indent=2)
    _log(json.dumps(res))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["generate", "score", "tag", "verify", "embed", "bench"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "pilot.yaml"))
    ap.add_argument("--vocab", default=os.path.join(HERE, "data", "cspace_cuisine_vocab.json"))
    ap.add_argument("--root", default=os.environ.get("ARIS_STORE", "/root/autodl-tmp/store"))
    ap.add_argument("--models-dir", default=os.environ.get("ARIS_MODELS", "/root/autodl-tmp/models"))
    ap.add_argument("--kinds", default="main,known,tune")
    ap.add_argument("--limit-pools", type=int, default=0)
    ap.add_argument("--scorers", default="pickscore,imagereward")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--vlm-batch", type=int, default=16)
    ap.add_argument("--reader", default="qwen", choices=["qwen", "pixtral", "siglip"],
                    help="qwen is pre-registered; the others are amendment 6 sensitivity readers")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--embedder", default="dinov2", choices=["dinov2", "siglip"],
                    help="siglip is a second, selector-independent feature space")
    a = ap.parse_args(argv)
    return {"generate": cmd_generate, "score": cmd_score, "tag": cmd_tag,
            "verify": cmd_verify, "embed": cmd_embed, "bench": cmd_bench}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
