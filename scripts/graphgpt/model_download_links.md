# GraphGPT model weights — direct download links

Download in a browser or with IDM, wget, curl, etc., then copy to the matching paths on your server.

---

## 1. Vicuna-7B-v1.5 (~13.5 GB)

Save under: `gfm-toolbox-main/ckpts/graphgpt/checkpoints/vicuna-7b-v1.5/`

### China mirror (hf-mirror, recommended)

| File | URL |
|------|-----|
| Shard 1 (~5.5 GB) | https://hf-mirror.com/lmsys/vicuna-7b-v1.5/resolve/main/pytorch_model-00001-of-00002.bin |
| Shard 2 (~7.9 GB) | https://hf-mirror.com/lmsys/vicuna-7b-v1.5/resolve/main/pytorch_model-00002-of-00002.bin |
| Index | https://hf-mirror.com/lmsys/vicuna-7b-v1.5/resolve/main/pytorch_model.bin.index.json |
| Config | https://hf-mirror.com/lmsys/vicuna-7b-v1.5/resolve/main/config.json |
| Generation config | https://hf-mirror.com/lmsys/vicuna-7b-v1.5/resolve/main/generation_config.json |
| Special tokens | https://hf-mirror.com/lmsys/vicuna-7b-v1.5/resolve/main/special_tokens_map.json |
| Tokenizer | https://hf-mirror.com/lmsys/vicuna-7b-v1.5/resolve/main/tokenizer.model |
| Tokenizer config | https://hf-mirror.com/lmsys/vicuna-7b-v1.5/resolve/main/tokenizer_config.json |

### Official (Hugging Face)

| File | URL |
|------|-----|
| Shard 1 | https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/pytorch_model-00001-of-00002.bin |
| Shard 2 | https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/pytorch_model-00002-of-00002.bin |
| Index | https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/pytorch_model.bin.index.json |
| Config | https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/config.json |
| Generation config | https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/generation_config.json |
| Special tokens | https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/special_tokens_map.json |
| Tokenizer | https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/tokenizer.model |
| Tokenizer config | https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/tokenizer_config.json |

---

## 2. Graph encoder Arxiv-PubMed-GraphCLIP-GT (~254 MB)

Save under: `gfm-toolbox-main/ckpts/graphgpt/checkpoints/clip_gt_arxiv/`

### China mirror (recommended)

| File | URL |
|------|-----|
| Weights (pkl) | https://hf-mirror.com/Jiabin99/Arxiv-PubMed-GraphCLIP-GT/resolve/main/clip_gt_arxiv_pub.pkl |
| Config | https://hf-mirror.com/Jiabin99/Arxiv-PubMed-GraphCLIP-GT/resolve/main/config.json |

### Official

| File | URL |
|------|-----|
| Weights (pkl) | https://huggingface.co/Jiabin99/Arxiv-PubMed-GraphCLIP-GT/resolve/main/clip_gt_arxiv_pub.pkl |
| Config | https://huggingface.co/Jiabin99/Arxiv-PubMed-GraphCLIP-GT/resolve/main/config.json |

---

## 3. “ChatGPT” vs full GraphGPT

- **OpenAI ChatGPT**: there is **no** public full weight release; this doc and repo **cannot** provide a “full ChatGPT” direct link.
- **Paper GraphGPT (after two-stage graph instruction tuning)**: Hugging Face repo **`Jiabin99/GraphGPT-7B-mix-all`** (~7B, similar size to full Vicuna). **Vicuna above is only the base; for a ready-to-use GraphGPT checkpoint, use this repo.**

Save under: `gfm-toolbox-main/ckpts/graphgpt/checkpoints/GraphGPT-7B-mix-all/` (folder name can differ if paths in inference scripts match).

### Recommended: full repo download (sharded weights)

```bash
pip install -U huggingface_hub
# In China you can set:
export HF_ENDPOINT=https://hf-mirror.com

cd GraphGPT-main   # project root
huggingface-cli download Jiabin99/GraphGPT-7B-mix-all --local-dir ./ckpts/graphgpt/checkpoints/GraphGPT-7B-mix-all
```

### China mirror direct links (`Jiabin99/GraphGPT-7B-mix-all`, save all into one directory)

| File | Approx. size | URL |
|------|--------------|-----|
| Weight shard 1 | ~9.2 GB | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/pytorch_model-00001-of-00003.bin |
| Weight shard 2 | ~9.2 GB | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/pytorch_model-00002-of-00003.bin |
| Weight shard 3 | ~6.7 GB | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/pytorch_model-00003-of-00003.bin |
| Weight index | small | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/pytorch_model.bin.index.json |
| Graph projector | ~502 MB | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/graph_projector.bin |
| Model config | small | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/config.json |
| Generation config | small | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/generation_config.json |
| Tokenizer | small | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/tokenizer.model |
| Tokenizer config | small | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/tokenizer_config.json |
| Special tokens | small | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/special_tokens_map.json |
| added_tokens | small | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/added_tokens.json |
| Trainer state (optional) | ~5 MB | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/trainer_state.json |
| training_args (optional) | small | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/training_args.bin |
| README (optional) | small | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/README.md |
| .gitattributes (optional) | small | https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/resolve/main/.gitattributes |

**Total ~26 GB+** (three `pytorch_model` shards + `graph_projector` + configs). For inference you need at minimum: all three shards + index + `graph_projector.bin` + config + generation_config + tokenizer files + special_tokens + added_tokens.

Browse files: https://hf-mirror.com/Jiabin99/GraphGPT-7B-mix-all/tree/main

---

## After uploading to the server

1. Place Vicuna’s 8 files in: `ckpts/graphgpt/checkpoints/vicuna-7b-v1.5/`
2. Place the graph encoder’s 2 files in: `ckpts/graphgpt/checkpoints/clip_gt_arxiv/`
3. **(Optional)** For full GraphGPT: download `GraphGPT-7B-mix-all` into `ckpts/graphgpt/checkpoints/GraphGPT-7B-mix-all/`
4. From project root run: `python scripts/prepare_models.py --check-only` (checks Vicuna + graph encoder only; verify GraphGPT shards yourself)

If `tokenizer.json` is missing, add it from:

- Mirror: https://hf-mirror.com/lmsys/vicuna-7b-v1.5/resolve/main/tokenizer.json
- Official: https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/tokenizer.json
