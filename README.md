# Preference-score best-of-N selection narrows the diversity of generated cuisine images

This repository accompanies the Scientific Reports manuscript of the same title. It contains the experiment code, selected stored measurements, analysis outputs, and figure scripts. The study evaluates generated cuisine images with automated readers and image embeddings. It includes **no human evaluation or new human annotation**. The paper distinguishes pre-registered analyses from sensitivity and post hoc analyses; the files here retain that distinction.

## What is included

| Path | Purpose |
| --- | --- |
| `experiments/src/`, `experiments/run*.py`, `experiments/analyze.py`, `experiments/extra/` | Generation, scoring, automated reading, main analysis, and extension analyses. |
| `experiments/configs/` and `experiments/data/cspace_cuisine_vocab.json` | Study configurations and the cuisine vocabulary extracted from CUBE-CSpace. The JSON records its source and source checksum. |
| `experiments/results/store_mirror/` | Main SDXL and FLUX.1-schnell replay store: image manifests, preference scores, three readers' labels and verification records, and DINOv2 embeddings. Its own `SHA256SUMS.txt` covers this store. |
| `experiments/results/store_mirror_a7/` | Additional generators, scorers, readers, and derived WorldCuisines photograph records used in the amended and post hoc analyses. Photograph files are omitted. |
| `experiments/results/store_mirror_e15/` | Stored measurements for the Diffusion-DPO comparison; the paired comparison also uses the SDXL records in the main store. Benchmark files are omitted. |
| `experiments/results/store_siglip/` | SigLIP embeddings for the second embedding-space analysis. |
| Other selected JSON files under `experiments/results/` | Saved main, sensitivity, and post hoc analysis summaries corresponding to the manuscript and Supplementary Information. These are outputs, not substitutes for the underlying replay stores. |
| `paper_kbs/figures/scripts/` | Scientific Reports figure-production scripts retained at their original relative path. |

The main replay uses `store_mirror/`. The amended and post hoc findings draw on additional stores and scripts; for example, the extension code combines the main and `store_mirror_a7/` stores, the Diffusion-DPO comparison uses `store_mirror_e15/`, and the second embedding-space analysis uses `store_siglip/`. The real-photograph CPU analyses read the derived manifest, scores, reader outputs, and embeddings in `store_mirror_a7/realimg/`; they do not require the photograph bytes. The stored JSON summaries provide the reported outputs for comparison.

**Generated images are not in this repository.** The main experiment's images were approximately 3.8 GB and are not part of the local replay bundle. Manifests record prompts, seeds, and generator fingerprints so that a suitably provisioned system can attempt regeneration with the public checkpoints and matching settings. Regeneration requires GPU resources and model downloads; exact pixel reproduction can depend on the software and accelerator environment. The stored scores, reader outputs, and embeddings allow the CPU analyses to operate without regenerating images.

Figure scripts that place image thumbnails, including the Figure 1 hero, need those omitted images. Their plotted statistics and table inputs are included in the JSON results.

## Check the files

From the repository root, verify the release manifest before using the data:

```bash
sha256sum -c SHA256SUMS.txt
```

The main store also carries its original transfer checksums:

```bash
cd experiments/results/store_mirror
sha256sum -c SHA256SUMS.txt
```

## Replay the main CPU analysis

Run from `experiments/` on Linux or WSL. A Python environment needs the packages in `requirements.txt`. The GPU generation, model scoring, and real-photograph preparation stages have additional model and data dependencies. The requirements are not a frozen environment, so small numeric differences across environments are possible.

```bash
cd experiments
python -m pip install -r requirements.txt
python analyze.py \
  --root results/store_mirror \
  --config configs/pilot.yaml \
  --vocab data/cspace_cuisine_vocab.json \
  --out results/replay_qwen
python make_report.py --in results/replay_qwen --out results/replay_qwen/report
```

The default reader is Qwen2.5-VL. To repeat the main-store sensitivity analyses for the other automated readers, give each its own output directory:

```bash
python analyze.py --root results/store_mirror --config configs/pilot.yaml \
  --vocab data/cspace_cuisine_vocab.json --reader pixtral --out results/replay_pixtral
python analyze.py --root results/store_mirror --config configs/pilot.yaml \
  --vocab data/cspace_cuisine_vocab.json --reader siglip --out results/replay_siglip
```

`analyze.py` records hashes of the configuration, vocabulary, relevant store files, and analysis code with its outputs, and refuses to overwrite outputs of different provenance without an explicit replacement option. Use new output directories when comparing a release with saved results.

## Release verification

In a fresh Linux/WSL environment with Python 3.12.3, NumPy 2.5.3, PyYAML 6.0.3, Matplotlib 3.11.2, and pytest 9.1.1, all **180 tests passed**. The root checksum manifest verified all **125 listed data files**. The main Qwen CPU analysis and `make_report.py` completed. For both SDXL and FLUX.1-schnell, every saved output field except `provenance` matched the corresponding `experiments/results/real/*.json` file exactly; the provenance metadata differs between runs. GPU generation and the extension analyses were not rerun for this release.

## Sources and rights

The cuisine vocabulary is derived from [CUBE-CSpace](https://github.com/google-deepmind/cube) (Kannen et al., *Beyond Aesthetics: Cultural Competence in Text-to-Image Models*). The CUBE dataset is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Attribute CUBE and follow that license when using the derived vocabulary. The original CUBE data and model checkpoints are not redistributed here.

The real-photograph checks use [WorldCuisines `food-kb`](https://huggingface.co/datasets/worldcuisines/food-kb). Its photographs have individual source and license information. **No WorldCuisines photograph files or server snapshot are redistributed here.** The derived per-photograph records should be interpreted with their dataset source; use the original dataset and each photograph's license for any image reuse. Public model checkpoints and weights remain with their original providers and licenses.

This repository's publication does not replace the terms attached to those third-party sources. No blanket license for all repository contents is implied by this README.
