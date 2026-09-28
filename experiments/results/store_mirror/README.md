# Store mirror (replay bundle)

This is a copy of the server's `store/` directory, made after experiment audit run 02, so that the
reported analyses can be replayed without GPU access.

| Path | Contents |
|---|---|
| `manifest/{gen}.jsonl` | one record per image (prompt, pool, seed, generator fingerprint); 5,632 per generator |
| `scores/{gen}__{pickscore,imagereward}.jsonl` | per-image scorer outputs |
| `tags/{gen}.jsonl`, `tags/{gen}__{pixtral,siglip}.jsonl` | per-image dish tags: Qwen2.5-VL (pre-registered), Pixtral-12B, SigLIP so400m |
| `verify/{gen}.jsonl`, `verify/{gen}__{pixtral,siglip}.jsonl` | per-image forced-choice verifier records for the known-label pools |
| `embed/{gen}__dinov2.npz` | DINOv2-base embeddings used by STK, MMR and DPP and by the label-free Vendi |

`{gen}` is `sdxl` or `flux-schnell`. `SHA256SUMS.txt` lists a checksum for every file. The
embedding checksums were checked against the server copies after transfer.

**Images are not included** (about 3.8 GB). They are regenerable from the manifest seeds with
`run_gpu.py`, using the generator configurations fingerprinted in each manifest record.
