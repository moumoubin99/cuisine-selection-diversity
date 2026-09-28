"""Real models.  Imported only by `run_gpu.py`, on the machine with the GPU.

Every class here takes PIL images or file paths and returns plain numbers or
strings; nothing downstream of `src/store.py` imports torch.

  SDXLGenerator / FluxGenerator   batched, one torch.Generator per image, the
                                  seeds of `backends.image_seeds_for_pool`
  PickScorer                      yuvalkirstain/PickScore_v1 on the LAION
                                  CLIP-ViT-H-14 processor (the published recipe)
  ImageRewardScorer               THUDM/ImageReward, loaded from the local
                                  snapshot so nothing is fetched at run time
  QwenVL                          Qwen2.5-VL-7B-Instruct: the open tagger and the
                                  closed yes/no verifier.  The tagger prompt has
                                  NO vocabulary list and never sees a score.
  PixtralVL / SiglipReader        second and third readers (PREREG amendment 6):
                                  Pixtral-12B with the same prompts, and SigLIP
                                  zero-shot over the CSpace vocabulary
  DinoEmbedder                    facebook/dinov2-base CLS features, the MMR /
                                  frontier similarity space (never CLIP, which is
                                  the selector's own space)
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from .backends import image_seeds_for_pool

_DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _resolve(repo_id, models_dir=None):
    """A local directory for `repo_id` if one exists, else the repo id (HF cache)."""
    if models_dir:
        p = os.path.join(models_dir, repo_id.split("/")[-1])
        if os.path.isdir(p):
            return p
    return repo_id


# ------------------------------------------------------------------ generators

class _DiffusersGenerator:
    batch = 4

    def _call(self, prompts, gens):
        raise NotImplementedError

    def generate_pool(self, gen_prompts, pool_seed, device="cuda"):
        """All images of one pool, in index order, with their per-image seeds."""
        n = len(gen_prompts)
        seeds = image_seeds_for_pool(pool_seed, n)
        out = []
        for s in range(0, n, self.batch):
            ps = list(gen_prompts[s:s + self.batch])
            gens = [torch.Generator(device=device).manual_seed(int(x))
                    for x in seeds[s:s + self.batch]]
            out.extend(self._call(ps, gens))
        return out, [int(x) for x in seeds]


class SDXLGenerator(_DiffusersGenerator):
    def __init__(self, cfg, models_dir=None, device="cuda"):
        from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,
                               StableDiffusionXLPipeline)
        dt = _DT[cfg.get("dtype", "float16")]
        vae = None
        if cfg.get("vae"):
            vae = AutoencoderKL.from_pretrained(_resolve(cfg["vae"], models_dir),
                                                torch_dtype=dt)
        kw = {"vae": vae} if vae is not None else {}
        if cfg.get("unet"):
            # E15: a preference-fine-tuned UNet (Diffusion-DPO) in the stock pipeline
            from diffusers import UNet2DConditionModel
            kw["unet"] = UNet2DConditionModel.from_pretrained(
                _resolve(cfg["unet"], models_dir), subfolder="unet", torch_dtype=dt)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            _resolve(cfg["id"], models_dir), torch_dtype=dt, variant="fp16",
            use_safetensors=True, **kw).to(device)
        if cfg.get("sampler") == "DPMSolverMultistep":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config)
        elif cfg.get("sampler"):
            raise ValueError(f"unsupported sampler {cfg['sampler']!r}")
        self.pipe.set_progress_bar_config(disable=True)
        self.cfg = cfg
        self.batch = int(cfg.get("batch", 8))

    def _call(self, prompts, gens):
        c = self.cfg
        return self.pipe(prompt=prompts, negative_prompt=[c.get("negative_prompt", "")] * len(prompts),
                         num_inference_steps=c["steps"], guidance_scale=c["guidance_scale"],
                         height=c["height"], width=c["width"], generator=gens).images


class FluxGenerator(_DiffusersGenerator):
    def __init__(self, cfg, models_dir=None, device="cuda"):
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(
            _resolve(cfg["id"], models_dir),
            torch_dtype=_DT[cfg.get("dtype", "bfloat16")]).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.cfg = cfg
        self.batch = int(cfg.get("batch", 4))

    def _call(self, prompts, gens):
        c = self.cfg
        return self.pipe(prompt=prompts, num_inference_steps=c["steps"],
                         guidance_scale=c["guidance_scale"], height=c["height"],
                         width=c["width"], max_sequence_length=c.get("max_sequence_length", 256),
                         generator=gens).images


class PixArtSigmaGenerator(_DiffusersGenerator):
    """PixArt-Sigma (amendment 7): DiT with a T5-XXL text encoder."""

    def __init__(self, cfg, models_dir=None, device="cuda"):
        from diffusers import PixArtSigmaPipeline
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            _resolve(cfg["id"], models_dir),
            torch_dtype=_DT[cfg.get("dtype", "float16")]).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.cfg = cfg
        self.batch = int(cfg.get("batch", 8))

    def _call(self, prompts, gens):
        c = self.cfg
        return self.pipe(prompt=prompts, negative_prompt=c.get("negative_prompt", ""),
                         num_inference_steps=c["steps"], guidance_scale=c["guidance_scale"],
                         height=c["height"], width=c["width"],
                         max_sequence_length=c.get("max_sequence_length", 300),
                         generator=gens).images


class SD3Generator(_DiffusersGenerator):
    """Stable Diffusion 3.5 Medium (amendment 7): MMDiT, CLIP-L/G + T5-XXL."""

    def __init__(self, cfg, models_dir=None, device="cuda"):
        from diffusers import StableDiffusion3Pipeline
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            _resolve(cfg["id"], models_dir),
            torch_dtype=_DT[cfg.get("dtype", "bfloat16")]).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.cfg = cfg
        self.batch = int(cfg.get("batch", 4))

    def _call(self, prompts, gens):
        c = self.cfg
        return self.pipe(prompt=prompts,
                         negative_prompt=[c.get("negative_prompt", "")] * len(prompts),
                         num_inference_steps=c["steps"], guidance_scale=c["guidance_scale"],
                         height=c["height"], width=c["width"],
                         max_sequence_length=c.get("max_sequence_length", 256),
                         generator=gens).images


def load_diffusers_generator(cfg, models_dir=None, device="cuda"):
    if "pixart" in cfg["id"].lower():
        return PixArtSigmaGenerator(cfg, models_dir, device)
    if "stable-diffusion-3" in cfg["id"].lower():
        return SD3Generator(cfg, models_dir, device)
    if "xl" in cfg["id"].lower():
        return SDXLGenerator(cfg, models_dir, device)
    if "flux" in cfg["id"].lower():
        return FluxGenerator(cfg, models_dir, device)
    raise ValueError(f"no generator backend for {cfg['id']!r}")


# ------------------------------------------------------------------ scorers

def _open(x):
    return x if isinstance(x, Image.Image) else Image.open(x).convert("RGB")


class PickScorer:
    name = "pickscore"

    def __init__(self, device="cuda", models_dir=None):
        from transformers import AutoModel, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(
            _resolve("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", models_dir))
        self.model = AutoModel.from_pretrained(
            _resolve("yuvalkirstain/PickScore_v1", models_dir)).eval().to(device)
        self.device = device

    @torch.no_grad()
    def score(self, images, text, batch=32):
        t = self.proc(text=[text], padding=True, truncation=True, max_length=77,
                      return_tensors="pt").to(self.device)
        te = self.model.get_text_features(**t)
        te = te / te.norm(dim=-1, keepdim=True)
        out = []
        for s in range(0, len(images), batch):
            ims = [_open(x) for x in images[s:s + batch]]
            v = self.proc(images=ims, return_tensors="pt").to(self.device)
            ie = self.model.get_image_features(**v)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            out.append((self.model.logit_scale.exp() * ie @ te.T)[:, 0].float().cpu().numpy())
        return np.concatenate(out)


class ImageRewardScorer:
    name = "imagereward"

    def __init__(self, device="cuda", models_dir=None):
        import ImageReward as RM
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("THUDM/ImageReward", "ImageReward.pt")
        med = hf_hub_download("THUDM/ImageReward", "med_config.json")
        self.model = RM.load(ckpt, device=device, med_config=med)
        self.model.eval()

    @torch.no_grad()
    def score(self, images, text, batch=32):
        out = []
        for s in range(0, len(images), batch):
            ims = [_open(x) for x in images[s:s + batch]]
            r = self.model.score(text, ims)
            out.extend(r if isinstance(r, list) else [r])
        return np.asarray(out, dtype=np.float64)


class LaionAesScorer:
    """LAION-Aesthetics predictor v2 (amendment 7): the MLP of
    christophschuhmann/improved-aesthetic-predictor
    (sac+logos+ava1-l14-linearMSE.pth) on L2-normalised OpenAI CLIP ViT-L/14
    image embeddings.  Prompt-free: `text` is ignored, it is an aesthetic
    score, not a preference or alignment score."""
    name = "laion_aes"

    def __init__(self, device="cuda", models_dir=None):
        import torch.nn as nn
        from huggingface_hub import hf_hub_download
        from transformers import CLIPModel, CLIPProcessor
        p = _resolve("openai/clip-vit-large-patch14", models_dir)
        self.proc = CLIPProcessor.from_pretrained(p)
        self.clip = CLIPModel.from_pretrained(p).eval().to(device)
        self.mlp = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128),
                                 nn.Dropout(0.2), nn.Linear(128, 64), nn.Dropout(0.1),
                                 nn.Linear(64, 16), nn.Linear(16, 1))
        sd = torch.load(hf_hub_download("camenduru/improved-aesthetic-predictor",
                                        "sac+logos+ava1-l14-linearMSE.pth"), map_location="cpu")
        self.mlp.load_state_dict({k.replace("layers.", ""): v for k, v in sd.items()})
        self.mlp.eval().to(device)
        self.device = device

    @torch.no_grad()
    def score(self, images, text=None, batch=64):
        out = []
        for s in range(0, len(images), batch):
            v = self.proc(images=[_open(x) for x in images[s:s + batch]],
                          return_tensors="pt").to(self.device)
            e = self.clip.get_image_features(**v).float()
            e = e / e.norm(dim=-1, keepdim=True)
            out.append(self.mlp(e)[:, 0].cpu().numpy())
        return np.concatenate(out).astype(np.float64)


class HPSv21Scorer:
    """HPS v2.1 (amendment 7): open_clip ViT-H-14 fine-tuned on HPD v2
    (xswu/HPSv2, HPS_v2.1_compressed.pt); score = cosine(image, prompt), as in
    the reference implementation."""
    name = "hpsv21"

    def __init__(self, device="cuda", models_dir=None):
        import open_clip
        from huggingface_hub import hf_hub_download
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=None, precision="amp", device=device)
        # same file (sha256 c57a38fb...) mirrored on ModelScope; prefer a local copy
        local = os.path.join(models_dir or os.environ.get("ARIS_MODELS", ""),
                             "HPSv2", "HPS_v2.1_compressed.pt")
        path = local if os.path.exists(local) else hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.tok = open_clip.get_tokenizer("ViT-H-14")
        self.device = device

    @torch.no_grad()
    def score(self, images, text, batch=32):
        t = self.tok([text]).to(self.device)
        with torch.autocast("cuda"):
            te = self.model.encode_text(t)
        te = te.float() / te.float().norm(dim=-1, keepdim=True)
        out = []
        for s in range(0, len(images), batch):
            x = torch.stack([self.pre(_open(i)) for i in images[s:s + batch]]).to(self.device)
            with torch.autocast("cuda"):
                ie = self.model.encode_image(x)
            ie = ie.float() / ie.float().norm(dim=-1, keepdim=True)
            out.append((ie @ te.T)[:, 0].cpu().numpy())
        return np.concatenate(out).astype(np.float64)


def load_real_scorer(name, device="cuda", models_dir=None):
    if name == "laion_aes":
        return LaionAesScorer(device, models_dir)
    if name == "hpsv21":
        return HPSv21Scorer(device, models_dir)
    if name == "pickscore":
        return PickScorer(device, models_dir)
    if name == "imagereward":
        return ImageRewardScorer(device, models_dir)
    raise ValueError(f"no scorer backend for {name!r}")


# ------------------------------------------------------------------ VLM

TAGGER_PROMPT = (
    "This image was produced by a text-to-image model and is meant to show a dish "
    "or food item.  Identify it.\n"
    "Reply with JSON only, in this form:\n"
    '{"dish": "<the most specific name of the dish, as it is usually written in '
    'English or its romanized native name>", "country": "<the country whose '
    'cuisine it most likely comes from>"}\n'
    'If no identifiable dish or food is shown, reply {"dish": "unknown", "country": "unknown"}.')

VERIFIER_PROMPT = ("Does this image show {dish}, a dish or food from {country}? "
                   "Answer with a single word, yes or no.")

# The yes/no verifier is yes-biased: on Brazilian SDXL images it answered
# P(yes) = 0.71 and 0.88 to two Nigerian dishes.  Keeping images on it would keep
# nearly everything.  The KEEP decision is therefore a forced choice among the
# prompted dish, three decoys from the same pool, and "none of these"; the
# yes/no probabilities are recorded for the true dish AND one decoy, so the
# bias is reported, not hidden.
VERIFIER_MC_PROMPT = ("Which dish does this image show?\n{options}\n"
                      "Answer with a single letter.")
MC_LETTERS = "ABCDE"


class QwenVL:
    """Qwen2.5-VL-7B-Instruct, used twice: open tagging and closed verification.

    `tag` has no `scores` parameter and never will: the tagger must be blind to
    the selector.  Images are resized to at most `max_pixels` (448^2 by
    default) -- identification does not need 1024^2, and the cap keeps a batch
    of 16 inside memory next to nothing else.
    """

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda",
                 models_dir=None, max_pixels=448 * 448):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        path = _resolve(model_id, models_dir)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path, min_pixels=256 * 28 * 28,
                                                  max_pixels=max_pixels)
        self.proc.tokenizer.padding_side = "left"
        self.device = device
        tok = self.proc.tokenizer
        self._yes = sorted({tok.encode(w, add_special_tokens=False)[0]
                            for w in ("yes", "Yes", " yes", " Yes")})
        self._no = sorted({tok.encode(w, add_special_tokens=False)[0]
                           for w in ("no", "No", " no", " No")})

    def _inputs(self, images, texts):
        msgs = [[{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": t}]}]
                for t in texts]
        chat = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in msgs]
        ims = [_open(x) for x in images]
        return self.proc(text=chat, images=ims, padding=True,
                         return_tensors="pt").to(self.device)

    @torch.no_grad()
    def tag(self, images, batch=16, max_new_tokens=48):
        out = []
        for s in range(0, len(images), batch):
            ims = images[s:s + batch]
            x = self._inputs(ims, [TAGGER_PROMPT] * len(ims))
            g = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
            g = g[:, x["input_ids"].shape[1]:]
            out.extend(self.proc.batch_decode(g, skip_special_tokens=True))
        return out

    @torch.no_grad()
    def verify(self, images, dishes, countries, batch=16):
        """P(yes) for "does this image show {dish}?", from the first-token logits."""
        out = []
        for s in range(0, len(images), batch):
            texts = [VERIFIER_PROMPT.format(dish=d, country=c)
                     for d, c in zip(dishes[s:s + batch], countries[s:s + batch])]
            x = self._inputs(images[s:s + batch], texts)
            logits = self.model(**x).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            y = torch.logsumexp(lp[:, self._yes], dim=-1)
            n = torch.logsumexp(lp[:, self._no], dim=-1)
            out.extend(torch.sigmoid(y - n).cpu().numpy().tolist())
        return out


    @torch.no_grad()
    def choose(self, images, options, batch=16):
        """Forced choice.  `options[i]` is a list of 5 strings (last: none of
        these); returns the softmax over the five letter tokens, per image."""
        tok = self.proc.tokenizer
        ids = [sorted({tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)})
               for L in MC_LETTERS]
        out = []
        for s in range(0, len(images), batch):
            texts = [VERIFIER_MC_PROMPT.format(options="\n".join(
                f"{L}) {o}" for L, o in zip(MC_LETTERS, opts)))
                for opts in options[s:s + batch]]
            x = self._inputs(images[s:s + batch], texts)
            lp = torch.log_softmax(self.model(**x).logits[:, -1, :].float(), dim=-1)
            z = torch.stack([torch.logsumexp(lp[:, t], dim=-1) for t in ids], dim=-1)
            out.extend(torch.softmax(z, dim=-1).cpu().numpy().tolist())
        return out


class PixtralVL(QwenVL):
    """Pixtral-12B, the second reader of PREREG_AMENDMENT.md amendment 6.

    Same prompts, forced-choice options and greedy decoding as `QwenVL`; only
    the model and its input packing differ.  Images are shrunk to at most 768
    pixels on the long side.  The amendment first set 448 to match the Qwen
    cap, but at 448 Pixtral answered "unknown" on 3 of 4 smoke images that it
    described as plated food, so the open tag would have measured abstention;
    the change is logged in amendment 6 before any study image was read.
    """

    def __init__(self, model_id="mistral-community/pixtral-12b", device="cuda",
                 models_dir=None, max_side=768):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        path = _resolve(model_id, models_dir)
        # eager: transformers 4.51 has no SDPA path for the Pixtral vision tower
        self.model = LlavaForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        self.proc.tokenizer.padding_side = "left"
        if self.proc.tokenizer.pad_token is None:
            self.proc.tokenizer.pad_token = self.proc.tokenizer.eos_token
        self.device = device
        self.max_side = max_side
        tok = self.proc.tokenizer
        self._yes = sorted({tok.encode(w, add_special_tokens=False)[0]
                            for w in ("yes", "Yes", " yes", " Yes")})
        self._no = sorted({tok.encode(w, add_special_tokens=False)[0]
                           for w in ("no", "No", " no", " No")})

    def _inputs(self, images, texts):
        msgs = [[{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "content": t, "text": t}]}]
                for t in texts]
        chat = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in msgs]
        ims = []
        for x in images:
            im = _open(x).copy()
            im.thumbnail((self.max_side, self.max_side))
            ims.append([im])
        x = self.proc(text=chat, images=ims, padding=True, return_tensors="pt")
        return {k: (v.to(self.device, torch.bfloat16) if k == "pixel_values" else v.to(self.device))
                for k, v in x.items()}


class SiglipReader:
    """SigLIP so400m as a closed-vocabulary, non-generative third reader
    (amendment 6).  `tag` returns the reply string the open tagger would, so
    `parse_tagger_json` and the whole replay path are unchanged; `choose` has
    no abstain option, so "none of these" gets probability 0."""

    TEXT = "a photo of {dish}, a dish."

    def __init__(self, vocab_by_country=None, model_id="google/siglip-so400m-patch14-384",
                 device="cuda", models_dir=None):
        from transformers import AutoModel, AutoProcessor
        path = _resolve(model_id, models_dir)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        self.device = device
        self._names, self._countries, self._T = [], [], None
        if vocab_by_country:
            seen = set()
            for c in sorted(vocab_by_country):
                for d in vocab_by_country[c]:
                    if d not in seen:
                        seen.add(d)
                        self._names.append(d)
                        self._countries.append(c)
            self._T = self._text([self.TEXT.format(dish=d) for d in self._names])

    @torch.no_grad()
    def _text(self, texts, batch=256):
        out = []
        for s in range(0, len(texts), batch):
            t = self.proc(text=texts[s:s + batch], padding="max_length", max_length=64,
                          truncation=True, return_tensors="pt").to(self.device)
            e = self.model.get_text_features(**t).float()
            out.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(out)

    @torch.no_grad()
    def _image(self, images):
        v = self.proc(images=[_open(x) for x in images], return_tensors="pt")
        e = self.model.get_image_features(
            pixel_values=v["pixel_values"].to(self.device, torch.float16)).float()
        return e / e.norm(dim=-1, keepdim=True)

    def _logits(self, I, T):
        return I @ T.T * self.model.logit_scale.exp().float() + self.model.logit_bias.float()

    @torch.no_grad()
    def tag(self, images, batch=64, **_):
        import json as _json
        out = []
        for s in range(0, len(images), batch):
            idx = (self._image(images[s:s + batch]) @ self._T.T).argmax(dim=-1).tolist()
            out.extend(_json.dumps({"dish": self._names[i], "country": self._countries[i]})
                       for i in idx)
        return out

    @torch.no_grad()
    def verify(self, images, dishes, countries, batch=64):
        out = []
        for s in range(0, len(images), batch):
            I = self._image(images[s:s + batch])
            T = self._text([self.TEXT.format(dish=d) for d in dishes[s:s + batch]])
            out.extend(torch.sigmoid((self._logits(I, T)).diagonal()).cpu().numpy().tolist())
        return out

    @torch.no_grad()
    def choose(self, images, options, batch=64):
        out = []
        for s in range(0, len(images), batch):
            I = self._image(images[s:s + batch])
            for i, opts in enumerate(options[s:s + batch]):
                T = self._text([self.TEXT.format(dish=o) for o in opts[:-1]])
                p = torch.softmax(self._logits(I[i:i + 1], T)[0], dim=-1).cpu().numpy()
                out.append(p.tolist() + [0.0])
        return out


READERS = ("qwen", "pixtral", "siglip")


def load_reader(name, models_dir=None, vocab_by_country=None, device="cuda"):
    if name == "qwen":
        return QwenVL(models_dir=models_dir, device=device)
    if name == "pixtral":
        return PixtralVL(models_dir=models_dir, device=device)
    if name == "siglip":
        return SiglipReader(vocab_by_country, models_dir=models_dir, device=device)
    raise ValueError(f"no reader {name!r}")


class DinoEmbedder:
    def __init__(self, device="cuda", models_dir=None):
        from transformers import AutoImageProcessor, AutoModel
        p = _resolve("facebook/dinov2-base", models_dir)
        self.proc = AutoImageProcessor.from_pretrained(p)
        self.model = AutoModel.from_pretrained(p).eval().to(device)
        self.device = device

    @torch.no_grad()
    def embed(self, images, batch=64):
        out = []
        for s in range(0, len(images), batch):
            ims = [_open(x) for x in images[s:s + batch]]
            v = self.proc(images=ims, return_tensors="pt").to(self.device)
            out.append(self.model(**v).pooler_output.float().cpu().numpy())
        return np.concatenate(out)


class SiglipEmbedder:
    """SigLIP so400m image embeddings: a second label-free feature space,
    independent of the DINOv2 space the selectors use."""

    def __init__(self, device="cuda", models_dir=None):
        self._r = SiglipReader(None, device=device, models_dir=models_dir)

    def embed(self, images, batch=64):
        out = []
        for s in range(0, len(images), batch):
            out.append(self._r._image(images[s:s + batch]).cpu().numpy())
        return np.concatenate(out)


def load_embedder(name, models_dir=None, device="cuda"):
    if name == "dinov2":
        return DinoEmbedder(device=device, models_dir=models_dir)
    if name == "siglip":
        return SiglipEmbedder(device=device, models_dir=models_dir)
    raise ValueError(f"no embedder {name!r}")
